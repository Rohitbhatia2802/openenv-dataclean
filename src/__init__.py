"""
OpenEnv Data Cleaning & Validation Package
==========================================

A production-grade OpenEnv environment simulating real-world enterprise
data cleaning pipelines. Exposes the standard OpenEnv API surface:

    - DataCleaningEnv.reset(task_id) -> Observation
    - DataCleaningEnv.step(action_dict) -> Tuple[Observation, Reward, bool, dict]
    - DataCleaningEnv.state() -> State

Three difficulty tiers are included:
    - easy   : fix_missing_price         (single-column imputation)
    - medium : normalize_customer_pipeline (multi-column CRM normalization)
    - hard   : validate_medical_records  (constraint-heavy EMR validation)

Usage::

    from src.env import DataCleaningEnv

    env = DataCleaningEnv()
    obs = env.reset("fix_missing_price")
    obs, reward, done, info = env.step({
        "operation": "impute_missing",
        "target_columns": ["price"],
        "params": {"strategy": "median"}
    })
    state = env.state()
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openenv-dataclean")
except PackageNotFoundError:
    __version__ = "1.0.0"

__author__ = "OpenEnv Hackathon Submission"
__license__ = "MIT"

__all__ = [
    "DataCleaningEnv",
    "Observation",
    "Action",
    "Reward",
    "State",
]
