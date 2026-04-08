"""
Inference script for OpenEnv Data Cleaning Environment.
This script is required by the hackathon validator at the repository root.
It runs a basic rule-based agent to demonstrate environment functionality.
"""

import sys
import json
from src.env import DataCleaningEnv

def get_action(obs):
    """Simple rule-based logic for inference testing."""
    if obs.total_defects == 0 or obs.defect_reduction_pct >= 95.0:
        return {"operation": "export", "target_columns": [], "params": {}}

    # Priority list of defects to fix
    candidates = [
        (obs.missing_count,           "impute_missing",      [],                          {"strategy": "median"}),
        (obs.sentinel_count,          "replace_sentinel",     [],                          {"sentinel": -1, "strategy": "median"}),
        (obs.duplicate_count,         "deduplicate",          [],                          {"keep": "first"}),
        (obs.phone_format_errors,     "standardize_phone",    ["phone"],                   {}),
        (obs.email_case_errors,       "lowercase_email",      ["email"],                   {}),
        (obs.age_range_violations,    "clamp_range",          ["age"],                     {"min_val": 18, "max_val": 100}),
        (obs.date_order_violations,   "fix_date_order",       [],                          {"mode": "swap"}),
        (obs.icd10_format_errors,     "fix_icd10",            ["icd10_code"],              {}),
        (obs.negative_dosage_count,   "fix_negative_dosage",  ["dosage_mg"],               {}),
        (obs.blank_mandatory_count,   "fill_mandatory",       ["patient_id", "diagnosis"], {"value": "UNKNOWN"}),
    ]

    candidates.sort(key=lambda x: x[0], reverse=True)
    for count, op, cols, params in candidates:
        if count > 0:
            return {"operation": op, "target_columns": cols, "params": params}

    return {"operation": "export", "target_columns": [], "params": {}}

def main():
    # Allow passing task_id as first argument
    task_id = sys.argv[1] if len(sys.argv) > 1 else "fix_missing_price"
    
    try:
        env = DataCleaningEnv(seed=42)
        obs = env.reset(task_id)
        
        # Run for up to 10 steps or until done
        for _ in range(10):
            action = get_action(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        # Output JSON result for the grader
        result = {
            "task_id": task_id,
            "steps_taken": obs.step_number,
            "final_defects": obs.total_defects,
            "grade": info.get("grade", 0.0),
            "status": "completed"
        }
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
