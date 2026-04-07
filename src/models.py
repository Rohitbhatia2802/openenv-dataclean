"""
Pydantic v2 models for the OpenEnv Data Cleaning Environment.

All models use strict typing and are serializable to/from JSON to
satisfy the OpenEnv specification validation requirements.

Classes
-------
ColumnStats
    Descriptive statistics for a single DataFrame column.
Observation
    Observation returned by reset() and step().
Action
    Structured action submitted by the agent.
Reward
    Step-level reward with breakdown components.
State
    Full internal state snapshot returned by state().
TaskConfig
    Static configuration for a task (loaded from openenv.yaml).
EpisodeHistory
    Record of all (action, reward) pairs in the current episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OperationType(str, Enum):
    """All legal operations an agent may submit."""
    IMPUTE_MISSING = "impute_missing"
    REPLACE_SENTINEL = "replace_sentinel"
    DEDUPLICATE = "deduplicate"
    STANDARDIZE_PHONE = "standardize_phone"
    LOWERCASE_EMAIL = "lowercase_email"
    CLAMP_RANGE = "clamp_range"
    FIX_DATE_ORDER = "fix_date_order"
    FIX_ICD10 = "fix_icd10"
    FIX_NEGATIVE_DOSAGE = "fix_negative_dosage"
    FILL_MANDATORY = "fill_mandatory"
    DROP_COLUMN = "drop_column"
    VALIDATE = "validate"
    EXPORT = "export"


class ImputeStrategy(str, Enum):
    """Strategies for imputing missing values."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"


class Difficulty(str, Enum):
    """Task difficulty tier."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ColumnStats(BaseModel):
    """Descriptive statistics for a single DataFrame column."""
    column: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Pandas dtype as string")
    null_count: int = Field(..., ge=0, description="Number of null values")
    null_pct: float = Field(..., ge=0.0, le=100.0, description="Null percentage")
    unique_count: int = Field(..., ge=0, description="Number of unique values")
    sample_values: List[Any] = Field(
        default_factory=list,
        description="Up to 5 sample non-null values"
    )

    model_config = {"populate_by_name": True}


class RewardBreakdown(BaseModel):
    """Itemised reward components for interpretability."""
    progress_reward: float = Field(0.0, description="Reward for measurable progress")
    step_penalty: float = Field(0.0, description="Per-step living cost penalty")
    invalid_action_penalty: float = Field(0.0, description="Penalty for illegal ops")
    destructive_penalty: float = Field(0.0, description="Penalty for destructive ops")
    completion_bonus: float = Field(0.0, description="Bonus for task completion")
    total: float = Field(0.0, description="Sum of all components, clamped [0,1]")


class EpisodeStep(BaseModel):
    """Single step record stored in episode history."""
    step_number: int
    action: Dict[str, Any]
    reward: float
    cumulative_reward: float
    done: bool
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Core OpenEnv Models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Observation returned by reset() and step().

    Contains enough signal for an agent to decide the next action
    without needing direct DataFrame access.
    """
    task_id: str = Field(..., description="Active task identifier")
    step_number: int = Field(..., ge=0, description="Steps taken so far")
    max_steps: int = Field(..., gt=0, description="Maximum allowed steps")

    # Defect counts (task-agnostic)
    missing_count: int = Field(0, ge=0, description="Total null cells across monitored columns")
    sentinel_count: int = Field(0, ge=0, description="Cells with sentinel/invalid values")

    # Easy-task specific
    price_below_zero: int = Field(0, ge=0, description="Prices ≤ 0 (Easy task)")

    # Medium-task specific
    duplicate_count: int = Field(0, ge=0, description="Duplicate rows (Medium task)")
    phone_format_errors: int = Field(0, ge=0, description="Malformed phone numbers")
    email_case_errors: int = Field(0, ge=0, description="Non-lowercase email addresses")
    age_range_violations: int = Field(0, ge=0, description="Ages outside [18, 100]")

    # Hard-task specific
    date_order_violations: int = Field(0, ge=0, description="DOB > admission date errors")
    icd10_format_errors: int = Field(0, ge=0, description="Malformed ICD-10 codes")
    negative_dosage_count: int = Field(0, ge=0, description="Negative dosage values")
    blank_mandatory_count: int = Field(0, ge=0, description="Blank mandatory fields")
    cross_column_violations: int = Field(0, ge=0, description="Cross-column constraint violations")

    # Always present
    column_stats: List[ColumnStats] = Field(
        default_factory=list,
        description="Per-column statistics for monitored columns"
    )
    sample_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Up to 5 sample rows from the current DataFrame"
    )
    total_defects: int = Field(0, ge=0, description="Sum of all tracked defect counts")
    defect_reduction_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage reduction from initial defect count"
    )
    done: bool = Field(False, description="Whether the episode has ended")
    message: str = Field("", description="Human-readable status message")

    model_config = {"populate_by_name": True}


class Action(BaseModel):
    """
    Structured action submitted by the agent.

    The agent sends a JSON object; this model validates the schema
    before the environment executes the operation.
    """
    operation: OperationType = Field(..., description="Operation to perform")
    target_columns: List[str] = Field(
        default_factory=list,
        description="Columns to apply the operation to (empty = all monitored)"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters (e.g., strategy, value, min_val)"
    )

    @field_validator("target_columns", mode="before")
    @classmethod
    def coerce_columns(cls, v: Any) -> List[str]:
        """Accept None or a single string in addition to a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    model_config = {"populate_by_name": True, "use_enum_values": True}


class Reward(BaseModel):
    """Step-level reward with itemised breakdown, clamped to [0.0, 1.0]."""
    value: float = Field(..., ge=-1.0, le=1.0, description="Net reward for this step")
    breakdown: RewardBreakdown = Field(..., description="Itemised reward components")
    cumulative: float = Field(0.0, description="Cumulative episode reward so far")

    model_config = {"populate_by_name": True}


class State(BaseModel):
    """
    Full internal state snapshot.

    Returned by state() — richer than Observation, intended for
    debugging, logging, and human review rather than agent consumption.
    """
    task_id: str
    difficulty: Difficulty
    step_number: int
    max_steps: int
    cumulative_reward: float
    initial_defect_count: int
    current_defect_count: int
    defect_reduction_pct: float
    done: bool
    success: bool
    episode_history: List[EpisodeStep]
    dataframe_shape: tuple[int, int]
    dataframe_columns: List[str]
    dataframe_dtypes: Dict[str, str]
    current_observation: Optional[Observation] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class TaskConfig(BaseModel):
    """Static task configuration, parsed from openenv.yaml."""
    id: str
    display_name: str
    difficulty: Difficulty
    description: str
    max_steps: int
    success_threshold: float = Field(..., ge=0.0, le=1.0)
    reward_shaping: str = "dense"

    model_config = {"populate_by_name": True}
