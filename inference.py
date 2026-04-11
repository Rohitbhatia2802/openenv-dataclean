import asyncio
import os
import json
import textwrap
import logging
from typing import List, Optional, Dict, Any

from openai import OpenAI
from src.env import DataCleaningEnv

# Constants from environment or defaults
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "EMPTY"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK") or os.getenv("TASK_NAME") or "fix_missing_price"
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK") or "openenv-dataclean"
MAX_STEPS_DEFAULT = 10
TEMPERATURE = 0.0
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert data cleaning agent.
    Each turn you receive an observation JSON describing defects in a dataset.
    You must respond with a SINGLE valid JSON action to fix the defects.
    
    Valid actions:
    - {"operation": "impute_missing", "target_columns": ["col1"], "params": {"strategy": "median"}}
    - {"operation": "replace_sentinel", "target_columns": ["col1"], "params": {"sentinel": -1, "strategy": "median"}}
    - {"operation": "deduplicate", "target_columns": [], "params": {"keep": "first"}}
    - {"operation": "standardize_phone", "target_columns": ["phone"], "params": {}}
    - {"operation": "lowercase_email", "target_columns": ["email"], "params": {}}
    - {"operation": "clamp_range", "target_columns": ["age"], "params": {"min_val": 18, "max_val": 100}}
    - {"operation": "export", "target_columns": [], "params": {}}
    
    Provide ONLY the JSON object.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs: Any) -> Dict[str, Any]:
    # Simplify observation for LLM context
    obs_summary = {
        "task_id": obs.task_id,
        "step": obs.step_number,
        "total_defects": obs.total_defects,
        "defect_types": {
            "missing": obs.missing_count,
            "sentinel": obs.sentinel_count,
            "duplicate": obs.duplicate_count,
            "phone_errors": obs.phone_format_errors,
            "email_errors": obs.email_case_errors,
            "age_errors": obs.age_range_violations
        }
    }
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs_summary)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        text = (completion.choices[0].message.content or "").strip()
        return json.loads(text)
    except Exception as exc:
        # Fallback rule-based action
        if obs.total_defects == 0:
            return {"operation": "export", "target_columns": [], "params": {}}
        return {"operation": "impute_missing", "target_columns": [], "params": {"strategy": "median"}}

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # We use the local environment class as defined in this repo
    env = DataCleaningEnv(seed=42)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(TASK_NAME)
        limit = obs.max_steps if obs.max_steps else MAX_STEPS_DEFAULT
        for step in range(1, limit + 1):
            if obs.done:
                break

            action_dict = get_model_action(client, obs)
            action_str = json.dumps(action_dict)

            obs, reward_obj, done, info = env.step(action_dict)

            reward = reward_obj.value
            rewards.append(reward)
            steps_taken = step
            error = info.get("exec_error")

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = info.get("grade", sum(rewards) / limit if limit > 0 else 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
