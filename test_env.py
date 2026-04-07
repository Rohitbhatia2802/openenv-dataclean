"""Quick end-to-end verification of all three tasks."""
import sys
from src.env import DataCleaningEnv

def test_easy():
    print("=== Easy Task: fix_missing_price ===")
    e = DataCleaningEnv()
    o = e.reset("fix_missing_price")
    assert o.task_id == "fix_missing_price", "task_id mismatch"
    assert o.total_defects > 0, "Expected defects on reset"
    assert o.missing_count > 0, "Expected missing values"
    assert o.sentinel_count > 0, "Expected sentinel values"
    print(f"  reset OK | defects={o.total_defects} | missing={o.missing_count} | sentinels={o.sentinel_count}")

    # Step 1: impute missing -> should reach threshold (grade=0.95 exactly)
    obs1, reward1, done1, info1 = e.step({"operation": "impute_missing", "target_columns": ["price"], "params": {"strategy": "median"}})
    assert "grade" in info1, f"grade not in info1. Keys: {list(info1.keys())}"
    print(f"  impute_missing | reward={reward1.value:.4f} | grade={info1['grade']:.4f} | done={done1}")

    # If episode ended (grade hit threshold), that's correct behavior
    if done1:
        print(f"  Episode ended after impute (grade={info1['grade']:.4f} >= threshold=0.95) -- CORRECT")
    else:
        # Episode still running: also try replace_sentinel
        obs2, reward2, done2, info2 = e.step({"operation": "replace_sentinel", "target_columns": ["price"], "params": {"sentinel": -1, "strategy": "median"}})
        assert "grade" in info2, f"grade not in info2. Keys: {list(info2.keys())}"
        print(f"  replace_sentinel | reward={reward2.value:.4f} | grade={info2['grade']:.4f}")

    s = e.state()
    print(f"  state OK | shape={s.dataframe_shape} | cumulative={s.cumulative_reward:.4f} | success={s.success}")

    # Test with replace_sentinel on fresh episode
    print("  --- Testing replace_sentinel on fresh episode ---")
    e2 = DataCleaningEnv()
    o2 = e2.reset("fix_missing_price")
    obs_r, reward_r, done_r, info_r = e2.step({"operation": "replace_sentinel", "target_columns": ["price"], "params": {"sentinel": -1, "strategy": "median"}})
    assert "grade" in info_r, f"grade not in info_r. Keys: {list(info_r.keys())}"
    print(f"  replace_sentinel (fresh) | reward={reward_r.value:.4f} | grade={info_r['grade']:.4f} | done={done_r}")

    if not done_r:
        obs_v, reward_v, done_v, info_v = e2.step({"operation": "validate", "target_columns": [], "params": {}})
        assert "grade" in info_v, "grade not in info_v"
        print(f"  validate | reward={reward_v.value:.4f} | grade={info_v['grade']:.4f}")
    print()

def test_medium():
    print("=== Medium Task: normalize_customer_pipeline ===")
    e = DataCleaningEnv()
    o = e.reset("normalize_customer_pipeline")
    assert o.task_id == "normalize_customer_pipeline"
    assert o.duplicate_count > 0, "Expected duplicates"
    assert o.phone_format_errors > 0, "Expected phone errors"
    assert o.email_case_errors > 0, "Expected email case errors"
    assert o.age_range_violations > 0, "Expected age violations"
    print(f"  reset OK | dups={o.duplicate_count} | phones={o.phone_format_errors} | emails={o.email_case_errors} | ages={o.age_range_violations}")

    ops = [
        {"operation": "deduplicate", "target_columns": [], "params": {"keep": "first"}},
        {"operation": "lowercase_email", "target_columns": ["email"], "params": {}},
        {"operation": "standardize_phone", "target_columns": ["phone"], "params": {}},
        {"operation": "clamp_range", "target_columns": ["age"], "params": {"min_val": 18, "max_val": 100}},
    ]
    for action_dict in ops:
        obs, reward, done, info = e.step(action_dict)
        assert "grade" in info, f"grade not in info for {action_dict['operation']}"
        print(f"  {action_dict['operation']} | reward={reward.value:.4f} | grade={info['grade']:.4f} | done={done}")
        if done:
            break

    s = e.state()
    print(f"  state OK | shape={s.dataframe_shape} | steps={s.step_number} | cumulative={s.cumulative_reward:.4f}")
    print()

def test_hard():
    print("=== Hard Task: validate_medical_records ===")
    e = DataCleaningEnv()
    o = e.reset("validate_medical_records")
    assert o.task_id == "validate_medical_records"
    print(f"  reset OK | date_errs={o.date_order_violations} | icd10={o.icd10_format_errors} | neg_dosage={o.negative_dosage_count} | blank={o.blank_mandatory_count} | cross={o.cross_column_violations}")

    ops = [
        {"operation": "fix_date_order", "target_columns": [], "params": {"mode": "swap"}},
        {"operation": "fix_icd10", "target_columns": ["icd10_code"], "params": {}},
        {"operation": "fix_negative_dosage", "target_columns": ["dosage_mg"], "params": {}},
        {"operation": "fill_mandatory", "target_columns": ["patient_id", "diagnosis"], "params": {"value": "UNKNOWN"}},
        {"operation": "validate", "target_columns": [], "params": {}},
    ]
    for action_dict in ops:
        obs, reward, done, info = e.step(action_dict)
        assert "grade" in info, f"grade not in info for {action_dict['operation']}. Keys: {list(info.keys())}"
        print(f"  {action_dict['operation']} | reward={reward.value:.4f} | grade={info['grade']:.4f} | done={done}")
        if done:
            break

    s = e.state()
    print(f"  state OK | shape={s.dataframe_shape} | steps={s.step_number} | cumulative={s.cumulative_reward:.4f} | defects={s.current_defect_count}")
    print()

def test_invalid_action():
    print("=== Invalid Action Handling ===")
    e = DataCleaningEnv()
    e.reset("fix_missing_price")
    obs, reward, done, info = e.step({"operation": "not_a_real_op", "target_columns": [], "params": {}})
    assert reward.value < 0, "Invalid action should return penalty"
    assert "grade" in info, f"grade should be in invalid action info. Keys: {list(info.keys())}"
    print(f"  invalid_action OK | reward={reward.value:.4f} | grade={info['grade']:.4f}")
    print()

def test_already_done():
    print("=== Already Done Handling ===")
    e = DataCleaningEnv()
    e.reset("fix_missing_price")
    e.step({"operation": "export", "target_columns": [], "params": {}})
    obs, reward, done, info = e.step({"operation": "validate", "target_columns": [], "params": {}})
    assert done, "Should remain done"
    assert reward.value == 0.0, "Post-done reward should be 0"
    print(f"  already_done OK | done={done} | reward={reward.value}")
    print()

if __name__ == "__main__":
    try:
        test_easy()
        test_medium()
        test_hard()
        test_invalid_action()
        test_already_done()
        print("=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
