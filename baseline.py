"""
Baseline OpenAI Inference Script
==================================

Runs a GPT-4o agent against all three OpenEnv Data Cleaning tasks
and prints a final JSON summary with task scores, step counts, and
reward trajectories.

Usage::

    # Windows PowerShell
    $env:OPENAI_API_KEY = "sk-..."
    python baseline.py

    # Demo mode (no API key needed — uses rule-based fallback agent)
    python baseline.py --demo

Design choices:
  - temperature=0 for deterministic, reproducible runs
  - max_steps respects each task's configured limit
  - Malformed JSON from the model is caught gracefully (penalty, no crash)
  - All three tasks are run sequentially; results are aggregated
  - Compatible with openai SDK v1.x and v2.x

Output schema::

    {
      "results": [
        {
          "task_id": "fix_missing_price",
          "difficulty": "easy",
          "steps": 8,
          "final_grade": 0.982,
          "final_reward": 0.74,
          "success": true,
          "reward_trajectory": [0.19, 0.15, 0.12, ...]
        },
        ...
      ],
      "summary": {
        "mean_grade": 0.91,
        "mean_steps": 14,
        "tasks_succeeded": 3
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("openenv.baseline")

# Guard against missing dependency *before* importing openai
try:
    from openai import OpenAI
    import openai as _openai_module
    _OPENAI_VERSION = tuple(int(x) for x in _openai_module.__version__.split(".")[:2])
except ImportError:
    logger.error("openai package not found. Install with: pip install openai")
    sys.exit(1)

from src.env import DataCleaningEnv, ALLOWED_TASK_IDS
from src.models import Observation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_RETRIES = 3

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert data cleaning agent operating inside an OpenEnv environment.

    At each step you receive an observation JSON describing the current dataset
    defects. You must respond with a SINGLE valid JSON action to fix one or more
    defects. Do NOT include any explanation outside the JSON block.

    Available operations:
      - impute_missing        : Fill null values. params: {strategy: median|mean|mode|constant, value: <value>}
      - replace_sentinel      : Replace sentinel values (e.g. -1). params: {sentinel: -1, strategy: median}
      - deduplicate           : Remove duplicate rows. params: {keep: first|last}
      - standardize_phone     : Normalize phone to E.164. params: {} (applies to 'phone' column)
      - lowercase_email       : Convert emails to lowercase. params: {}
      - clamp_range           : Clamp numeric column. params: {min_val: 18, max_val: 100}
      - fix_date_order        : Fix DOB > admission_date. params: {mode: swap|null_dob}
      - fix_icd10             : Fix ICD-10 code format. params: {}
      - fix_negative_dosage   : Make dosages positive. params: {}
      - fill_mandatory        : Fill blank mandatory fields. params: {value: "UNKNOWN"}
      - validate              : Check current grade (no-op).
      - export                : Signal you are done.

    Response format (strict JSON, no markdown):
    {
      "operation": "<operation_name>",
      "target_columns": ["col1"],
      "params": {}
    }

    Strategy:
    1. Read total_defects and each defect count in the observation.
    2. Pick the operation that reduces the most defects.
    3. When total_defects == 0 or defect_reduction_pct >= 95, use export.
    """
).strip()


# ---------------------------------------------------------------------------
# Rule-based demo agent (no API key required)
# ---------------------------------------------------------------------------

def _rule_based_action(obs: Observation) -> Dict[str, Any]:
    """
    Deterministic rule-based agent used in --demo mode.

    Picks the operation that addresses the largest defect count.
    No API calls — fully offline.
    """
    if obs.total_defects == 0 or obs.defect_reduction_pct >= 95.0:
        return {"operation": "export", "target_columns": [], "params": {}}

    candidates = [
        (obs.missing_count,           {"operation": "impute_missing",      "target_columns": [],                          "params": {"strategy": "median"}}),
        (obs.sentinel_count,          {"operation": "replace_sentinel",     "target_columns": [],                          "params": {"sentinel": -1, "strategy": "median"}}),
        (obs.duplicate_count,         {"operation": "deduplicate",          "target_columns": [],                          "params": {"keep": "first"}}),
        (obs.phone_format_errors,     {"operation": "standardize_phone",    "target_columns": ["phone"],                   "params": {}}),
        (obs.email_case_errors,       {"operation": "lowercase_email",      "target_columns": ["email"],                   "params": {}}),
        (obs.age_range_violations,    {"operation": "clamp_range",          "target_columns": ["age"],                     "params": {"min_val": 18, "max_val": 100}}),
        (obs.date_order_violations,   {"operation": "fix_date_order",       "target_columns": [],                          "params": {"mode": "swap"}}),
        (obs.icd10_format_errors,     {"operation": "fix_icd10",            "target_columns": ["icd10_code"],              "params": {}}),
        (obs.negative_dosage_count,   {"operation": "fix_negative_dosage",  "target_columns": ["dosage_mg"],               "params": {}}),
        (obs.blank_mandatory_count,   {"operation": "fill_mandatory",       "target_columns": ["patient_id", "diagnosis"], "params": {"value": "UNKNOWN"}}),
        (obs.cross_column_violations, {"operation": "validate",             "target_columns": [],                          "params": {}}),
    ]

    candidates.sort(key=lambda x: x[0], reverse=True)
    for count, action in candidates:
        if count > 0:
            return action

    return {"operation": "export", "target_columns": [], "params": {}}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_task(
    env: DataCleaningEnv,
    client: Optional[OpenAI],
    task_id: str,
    max_steps: int,
    demo_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run the agent on a single task until done or max_steps.

    Parameters
    ----------
    env : DataCleaningEnv
        The environment instance.
    client : OpenAI or None
        Authenticated OpenAI client. None when demo_mode=True.
    task_id : str
        Task identifier.
    max_steps : int
        Maximum steps before forced termination.
    demo_mode : bool
        If True, use the rule-based agent instead of GPT.

    Returns
    -------
    dict
        Result dictionary for this task.
    """
    logger.info("=" * 60)
    logger.info(
        "Starting task: %s (max_steps=%d, mode=%s)",
        task_id, max_steps, "DEMO" if demo_mode else "GPT-4o",
    )

    obs: Observation = env.reset(task_id)
    difficulty = env._config.difficulty.value
    reward_trajectory: List[float] = []
    steps = 0
    done = False
    final_grade = 0.0
    cumulative_reward = 0.0

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not done and steps < max_steps:
        obs_summary = {
            "task_id": obs.task_id,
            "step": obs.step_number,
            "max_steps": obs.max_steps,
            "total_defects": obs.total_defects,
            "defect_reduction_pct": obs.defect_reduction_pct,
            "missing_count": obs.missing_count,
            "sentinel_count": obs.sentinel_count,
            "price_below_zero": obs.price_below_zero,
            "duplicate_count": obs.duplicate_count,
            "phone_format_errors": obs.phone_format_errors,
            "email_case_errors": obs.email_case_errors,
            "age_range_violations": obs.age_range_violations,
            "date_order_violations": obs.date_order_violations,
            "icd10_format_errors": obs.icd10_format_errors,
            "negative_dosage_count": obs.negative_dosage_count,
            "blank_mandatory_count": obs.blank_mandatory_count,
            "cross_column_violations": obs.cross_column_violations,
        }

        if demo_mode:
            action_dict = _rule_based_action(obs)
        else:
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n{json.dumps(obs_summary, indent=2)}",
            })
            action_dict = _get_action(client, messages)
            messages.append({
                "role": "assistant",
                "content": json.dumps(action_dict),
            })

        logger.info(
            "Step %d | action=%s cols=%s",
            steps + 1,
            action_dict.get("operation"),
            action_dict.get("target_columns", []),
        )

        obs, reward, done, info = env.step(action_dict)
        cumulative_reward = reward.cumulative
        reward_trajectory.append(round(reward.value, 4))
        steps += 1
        final_grade = info.get("grade", 0.0)

        logger.info(
            "  -> reward=%.4f | grade=%.3f | defects=%d | done=%s",
            reward.value, final_grade, obs.total_defects, done,
        )

    state_snap = env.state()
    success = state_snap.success

    logger.info(
        "Task finished: grade=%.3f | steps=%d | success=%s",
        final_grade, steps, success,
    )

    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "steps": steps,
        "final_grade": round(final_grade, 4),
        "final_reward": round(cumulative_reward, 4),
        "success": success,
        "reward_trajectory": reward_trajectory,
    }


def _get_action(
    client: OpenAI, messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Query the OpenAI API and parse the JSON action response.

    Compatible with openai SDK v1.x and v2.x:
      - v2.x renamed max_tokens -> max_completion_tokens
      - response_format dict syntax works in both versions

    Retries up to MAX_RETRIES times on parse failures.
    """
    for attempt in range(MAX_RETRIES):
        try:
            kwargs: Dict[str, Any] = {
                "model": MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,
                "response_format": {"type": "json_object"},
            }
            # SDK v2.x renamed max_tokens -> max_completion_tokens
            if _OPENAI_VERSION >= (2, 0):
                kwargs["max_completion_tokens"] = 300
            else:
                kwargs["max_tokens"] = 300

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()
            action = json.loads(content)
            action.setdefault("target_columns", [])
            action.setdefault("params", {})
            return action
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning(
                "JSON parse error on attempt %d/%d: %s", attempt + 1, MAX_RETRIES, exc
            )
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            break

    logger.warning("All retries exhausted. Returning fallback 'validate' action.")
    return {"operation": "validate", "target_columns": [], "params": {}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the baseline agent script."""
    demo_mode = "--demo" in sys.argv

    if demo_mode:
        logger.info("Running in DEMO mode (no API key required).")
        client = None
    else:
        # Check for hackathon proxy variables first
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("API_BASE_URL")

        if not api_key:
            logger.error(
                "No API key found. Set API_KEY or OPENAI_API_KEY environmental variables.\n"
                "  PowerShell : $env:OPENAI_API_KEY = 'sk-...'\n"
                "\nOr run without an API key using the demo agent:\n"
                "  python baseline.py --demo"
            )
            sys.exit(1)
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            logger.info("Using LLM proxy: %s", base_url)
            client_kwargs["base_url"] = base_url
            
        client = OpenAI(**client_kwargs)

    env = DataCleaningEnv(seed=42, verbose=False)

    task_configs = {
        "fix_missing_price": 20,
        "normalize_customer_pipeline": 35,
        "validate_medical_records": 50,
    }

    results: List[Dict[str, Any]] = []
    for task_id, max_steps in task_configs.items():
        result = run_task(env, client, task_id, max_steps, demo_mode=demo_mode)
        results.append(result)

    grades = [r["final_grade"] for r in results]
    steps_list = [r["steps"] for r in results]
    succeeded = sum(1 for r in results if r["success"])

    summary = {
        "mean_grade": round(sum(grades) / len(grades), 4),
        "mean_steps": round(sum(steps_list) / len(steps_list), 1),
        "tasks_succeeded": succeeded,
        "total_tasks": len(results),
    }

    output = {"results": results, "summary": summary}
    print("\n" + "=" * 60)
    print("FINAL BASELINE RESULTS")
    print("=" * 60)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
