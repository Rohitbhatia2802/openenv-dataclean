"""
Inference script for OpenEnv Data Cleaning Environment.
This script is required by the hackathon validator at the repository root.
It runs an LLM-based agent using the provided LiteLLM proxy environment variables.
"""

import sys
import json
import os
import textwrap
import logging
from src.env import DataCleaningEnv

# Suppress excessive logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("openenv.inference")

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Install with: pip install openai")
    sys.exit(1)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert data cleaning agent operating inside an OpenEnv environment.
    At each step you receive an observation JSON. You must respond with a SINGLE valid JSON action.
    Response format (strict JSON):
    {
      "operation": "<operation_name>",
      "target_columns": ["col1"],
      "params": {}
    }
    """
).strip()

def get_rule_based_action(obs):
    """Fallback rule-based logic if LLM fails."""
    if obs.total_defects == 0 or obs.defect_reduction_pct >= 95.0:
        return {"operation": "export", "target_columns": [], "params": {}}

    candidates = [
        (obs.missing_count,           "impute_missing",      [],                          {"strategy": "median"}),
        (obs.sentinel_count,          "replace_sentinel",     [],                          {"sentinel": -1, "strategy": "median"}),
        (obs.duplicate_count,         "deduplicate",          [],                          {"keep": "first"}),
        (obs.phone_format_errors,     "standardize_phone",    ["phone"],                   {}),
        (obs.email_case_errors,       "lowercase_email",      ["email"],                   {}),
        (obs.age_range_violations,    "clamp_range",          ["age"],                     {"min_val": 18, "max_val": 100}),
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)
    for count, op, cols, params in candidates:
        if count > 0:
            return {"operation": op, "target_columns": cols, "params": params}
    return {"operation": "export", "target_columns": [], "params": {}}

def get_llm_action(client, obs):
    """Query the LLM proxy for the next action."""
    obs_summary = {
        "task_id": obs.task_id,
        "step": obs.step_number,
        "total_defects": obs.total_defects,
        "missing_count": obs.missing_count,
        "duplicate_count": obs.duplicate_count,
        # Add more fields if needed
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # The proxy usually handles model mapping
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs_summary)}
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return None

def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else "fix_missing_price"
    
    # Initialize OpenAI client with validator's proxy settings
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("API_BASE_URL")
    
    client = None
    if api_key:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
    
    try:
        env = DataCleaningEnv(seed=42)
        obs = env.reset(task_id)
        print(f"[START] task={task_id}", flush=True)
        
        # Run for up to 10 steps
        for _ in range(10):
            action = None
            if client:
                action = get_llm_action(client, obs)
            
            if not action:
                action = get_rule_based_action(obs)
                
            obs, reward, done, info = env.step(action)
            print(f"[STEP] step={obs.step_number} reward={reward.value}", flush=True)
            
            if done:
                break
        
        print(f"[END] task={task_id} score={info.get('grade', 0.0)} steps={obs.step_number}", flush=True)
        
        result = {
            "task_id": task_id,
            "steps_taken": obs.step_number,
            "grade": info.get("grade", 0.0),
            "status": "completed"
        }
        print(json.dumps(result, indent=2), flush=True)
        
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
