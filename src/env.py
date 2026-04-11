"""
Core DataCleaningEnv — the main OpenEnv environment class.

Implements the three required OpenEnv API methods:
    - reset(task_id: str) -> Observation
    - step(action_dict: dict) -> Tuple[Observation, Reward, bool, dict]
    - state() -> State

Design decisions
----------------
* All DataFrame manipulation is in-memory using pandas / io.StringIO.
* No disk I/O occurs during an episode; datasets are embedded in task modules.
* Actions are validated via Pydantic before execution; malformed JSON returns
  a penalty but never crashes the server.
* Reward shaping is delegated to src.rewards.RewardCalculator.
* Grading is delegated to src.graders.{Easy,Medium,Hard}Grader.
* The environment is stateful but NOT thread-safe; wrap with a lock for
  concurrent server usage.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .models import (
    Action,
    ColumnStats,
    Difficulty,
    EpisodeStep,
    Observation,
    OperationType,
    Reward,
    RewardBreakdown,
    State,
    TaskConfig,
)
from .tasks.easy import EasyTask
from .tasks.medium import MediumTask
from .tasks.hard import HardTask
from .graders import EasyGrader, MediumGrader, HardGrader
from .rewards import RewardCalculator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fix_missing_price": {
        "config": TaskConfig(
            id="fix_missing_price",
            name="Fix Missing Price Values (Easy)",
            difficulty=Difficulty.EASY,
            description=(
                "Impute missing and sentinel price values in a retail "
                "transaction dataset using the column median."
            ),
            max_steps=20,
            pass_threshold=0.70,
            excellent_threshold=0.95,
        ),
        "task_cls": EasyTask,
        "grader_cls": EasyGrader,
    },
    "normalize_customer_pipeline": {
        "config": TaskConfig(
            id="normalize_customer_pipeline",
            name="Normalize Customer Data Pipeline (Medium)",
            difficulty=Difficulty.MEDIUM,
            description=(
                "Deduplicate, standardize phone/email, and clamp age values "
                "in a customer CRM export."
            ),
            max_steps=35,
            pass_threshold=0.60,
            excellent_threshold=0.90,
        ),
        "task_cls": MediumTask,
        "grader_cls": MediumGrader,
    },
    "validate_medical_records": {
        "config": TaskConfig(
            id="validate_medical_records",
            name="Validate Medical Records (Hard)",
            difficulty=Difficulty.HARD,
            description=(
                "Fix date-ordering, ICD-10 codes, negative dosages, and "
                "cross-column constraint violations in a hospital EMR export."
            ),
            max_steps=50,
            pass_threshold=0.55,
            excellent_threshold=0.85,
        ),
        "task_cls": HardTask,
        "grader_cls": HardGrader,
    },
}

ALLOWED_TASK_IDS = list(_TASK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class DataCleaningEnv:
    """
    OpenEnv-compliant Data Cleaning & Validation environment.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default 42).
    verbose : bool, optional
        If True, log each step to stdout (default False).

    Examples
    --------
    >>> env = DataCleaningEnv()
    >>> obs = env.reset("fix_missing_price")
    >>> obs, reward, done, info = env.step({"operation": "impute_missing",
    ...     "target_columns": ["price"], "params": {"strategy": "median"}})
    >>> state = env.state()
    """

    def __init__(self, seed: int = 42, verbose: bool = False) -> None:
        self._seed = seed
        self._verbose = verbose
        np.random.seed(seed)

        # Internal state (populated by reset)
        self._task_id: Optional[str] = None
        self._config: Optional[TaskConfig] = None
        self._task: Optional[Any] = None
        self._grader: Optional[Any] = None
        self._df: Optional[pd.DataFrame] = None
        self._initial_df: Optional[pd.DataFrame] = None
        self._step_number: int = 0
        self._done: bool = False
        self._success: bool = False
        self._cumulative_reward: float = 0.0
        self._episode_history: List[EpisodeStep] = []
        self._initial_defects: int = 0
        self._current_obs: Optional[Observation] = None
        self._reward_calc: Optional[RewardCalculator] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        task_id : str
            One of the registered task IDs (see ALLOWED_TASK_IDS).

        Returns
        -------
        Observation
            Initial observation of the fresh dataset.

        Raises
        ------
        ValueError
            If task_id is not registered.
        """
        if task_id not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {ALLOWED_TASK_IDS}"
            )

        registry = _TASK_REGISTRY[task_id]
        self._task_id = task_id
        self._config = registry["config"]
        self._task = registry["task_cls"](seed=self._seed)
        self._grader = registry["grader_cls"]()
        self._df = self._task.generate_dataset()
        self._initial_df = self._df.copy(deep=True)
        self._step_number = 0
        self._done = False
        self._success = False
        self._cumulative_reward = 0.0
        self._episode_history = []
        self._reward_calc = RewardCalculator(config=self._config)

        self._initial_defects = self._count_defects(self._df)
        obs = self._build_observation(message="Episode started. Dataset ready.")
        self._current_obs = obs

        if self._verbose:
            logger.info(
                "[RESET] task=%s | shape=%s | initial_defects=%d",
                task_id, self._df.shape, self._initial_defects,
            )
        return obs

    def step(
        self, action_dict: Dict[str, Any]
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Parameters
        ----------
        action_dict : dict
            JSON-serialisable dict conforming to the Action schema.
            Malformed input returns a penalty instead of crashing.

        Returns
        -------
        observation : Observation
        reward : Reward
        done : bool
        info : dict
            Auxiliary diagnostic information.
        """
        if self._done:
            obs = self._current_obs or self._build_observation(
                message="Episode already done. Call reset()."
            )
            zero_reward = Reward(
                value=0.0,
                breakdown=RewardBreakdown(),
                cumulative=self._cumulative_reward,
            )
            return obs, zero_reward, True, {"warning": "Episode already done"}

        # --- Parse & validate action ---
        action, parse_error = self._parse_action(action_dict)
        if parse_error:
            penalty = self._reward_calc.invalid_action_penalty()
            self._cumulative_reward = max(
                0.0, self._cumulative_reward + penalty.value
            )
            obs = self._build_observation(
                message=f"Invalid action: {parse_error}"
            )
            self._current_obs = obs
            self._step_number += 1
            self._record_step(action_dict, penalty.value, done=False)
            grade = self._grader.grade(self._df)
            return obs, penalty, False, {"error": parse_error, "grade": grade,
                "defects_before": self._count_defects(self._df),
                "defects_after": self._count_defects(self._df),
                "success": False, "step": self._step_number, "exec_error": parse_error}

        # --- Execute action ---
        defects_before = self._count_defects(self._df)
        exec_msg, exec_error = self._execute_action(action)
        defects_after = self._count_defects(self._df)

        self._step_number += 1

        # --- Compute reward ---
        grade = self._grader.grade(self._df)
        reward = self._reward_calc.compute(
            action=action,
            defects_before=defects_before,
            defects_after=defects_after,
            initial_defects=self._initial_defects,
            grade=grade,
            step_number=self._step_number,
            exec_error=exec_error,
        )
        reward.cumulative = self._cumulative_reward + reward.value
        self._cumulative_reward = max(0.001, min(0.999, reward.cumulative))

        # --- Check termination ---
        self._done = self._is_done(grade, action)
        self._success = grade >= self._config.pass_threshold

        # --- Build observation ---
        if exec_error:
            msg = f"Execution error: {exec_error}"
        elif self._done:
            msg = f"Episode complete! Grade={grade:.3f}, Success={self._success}"
        else:
            msg = exec_msg or f"Completed '{action.operation}'. Grade={grade:.3f}"

        obs = self._build_observation(message=msg)
        self._current_obs = obs
        self._record_step(action_dict, reward.value, done=self._done)

        info: Dict[str, Any] = {
            "grade": grade,
            "defects_before": defects_before,
            "defects_after": defects_after,
            "success": self._success,
            "step": self._step_number,
            "exec_error": exec_error,
        }

        if self._verbose:
            logger.info(
                "[STEP %d] op=%s | defects %d→%d | reward=%.4f | grade=%.3f | done=%s",
                self._step_number, action.operation,
                defects_before, defects_after,
                reward.value, grade, self._done,
            )

        return obs, reward, self._done, info

    def state(self) -> State:
        """
        Return the full internal state snapshot.

        Returns
        -------
        State
            Complete episode state including history, DataFrame metadata,
            and current observation.

        Raises
        ------
        RuntimeError
            If called before reset().
        """
        if self._df is None or self._config is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        current_defects = self._count_defects(self._df)
        reduction_pct = 0.0
        if self._initial_defects > 0:
            reduction_pct = (
                (self._initial_defects - current_defects)
                / self._initial_defects
                * 100.0
            )

        return State(
            task_id=self._task_id,
            difficulty=self._config.difficulty,
            step_number=self._step_number,
            max_steps=self._config.max_steps,
            cumulative_reward=self._cumulative_reward,
            initial_defect_count=self._initial_defects,
            current_defect_count=current_defects,
            defect_reduction_pct=round(reduction_pct, 2),
            done=self._done,
            success=self._success,
            episode_history=self._episode_history,
            dataframe_shape=tuple(self._df.shape),
            dataframe_columns=list(self._df.columns),
            dataframe_dtypes={c: str(t) for c, t in self._df.dtypes.items()},
            current_observation=self._current_obs,
            metadata={
                "seed": self._seed,
                "pass_threshold": self._config.pass_threshold,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_action(
        self, raw: Dict[str, Any]
    ) -> Tuple[Optional[Action], Optional[str]]:
        """Validate raw dict against Action schema."""
        try:
            action = Action.model_validate(raw)
            return action, None
        except ValidationError as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"Unexpected parse error: {exc}"

    def _execute_action(
        self, action: Action
    ) -> Tuple[str, Optional[str]]:
        """
        Dispatch action to the appropriate handler method.

        Returns a (message, error_string) tuple.
        """
        op = action.operation
        cols = action.target_columns or []
        params = action.params or {}

        handlers = {
            OperationType.IMPUTE_MISSING:      self._op_impute_missing,
            OperationType.REPLACE_SENTINEL:    self._op_replace_sentinel,
            OperationType.DEDUPLICATE:         self._op_deduplicate,
            OperationType.STANDARDIZE_PHONE:   self._op_standardize_phone,
            OperationType.LOWERCASE_EMAIL:     self._op_lowercase_email,
            OperationType.CLAMP_RANGE:         self._op_clamp_range,
            OperationType.FIX_DATE_ORDER:      self._op_fix_date_order,
            OperationType.FIX_ICD10:           self._op_fix_icd10,
            OperationType.FIX_NEGATIVE_DOSAGE: self._op_fix_negative_dosage,
            OperationType.FILL_MANDATORY:      self._op_fill_mandatory,
            OperationType.DROP_COLUMN:         self._op_drop_column,
            OperationType.VALIDATE:            self._op_validate,
            OperationType.EXPORT:              self._op_export,
        }

        handler = handlers.get(op)
        if handler is None:
            return "", f"No handler registered for operation '{op}'"

        try:
            msg = handler(cols, params)
            return msg, None
        except Exception as exc:
            logger.exception("Action execution error")
            return "", f"Runtime error in '{op}': {exc}"

    # ------------------------------------------------------------------
    # Operation handlers
    # ------------------------------------------------------------------

    def _op_impute_missing(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Impute null values in target columns using the given strategy.

        Uses pd.to_numeric coercion for mixed object-dtype columns so that
        statistics (median, mean) are computed correctly even when the column
        stores floats as objects (common when NaN is mixed with numeric values).
        """
        strategy = params.get("strategy", "median")
        fill_value = params.get("value", None)
        target = cols if cols else [c for c in self._df.columns if self._df[c].isnull().any()]
        affected = 0
        for col in target:
            if col not in self._df.columns:
                continue
            null_mask = self._df[col].isnull()
            if not null_mask.any():
                continue

            # Try numeric coercion for stats
            numeric_series = pd.to_numeric(self._df[col], errors="coerce")

            if strategy == "median":
                val = numeric_series.median()
            elif strategy == "mean":
                val = numeric_series.mean()
            elif strategy == "mode":
                mode_vals = self._df[col].dropna().mode()
                val = mode_vals.iloc[0] if not mode_vals.empty else None
            elif strategy == "constant":
                val = fill_value
            elif strategy == "forward_fill":
                self._df[col] = self._df[col].ffill()
                affected += int(null_mask.sum())
                continue
            elif strategy == "backward_fill":
                self._df[col] = self._df[col].bfill()
                affected += int(null_mask.sum())
                continue
            else:
                val = numeric_series.median()

            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                self._df.loc[null_mask, col] = val
                affected += int(null_mask.sum())

        return f"Imputed {affected} missing values using strategy='{strategy}'."

    def _op_replace_sentinel(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Replace a sentinel value with median or a supplied constant.

        Works on both numeric and object-typed columns containing numeric
        values mixed with NaN (e.g. price columns stored as object dtype).
        """
        sentinel = params.get("sentinel", -1)
        strategy = params.get("strategy", "median")
        fill_value = params.get("value", None)

        # Fall back to all columns if none specified (coerce to numeric to find candidates)
        if cols:
            target = cols
        else:
            target = []
            for c in self._df.columns:
                numeric = pd.to_numeric(self._df[c], errors="coerce")
                if (numeric == sentinel).any():
                    target.append(c)

        affected = 0
        for col in target:
            if col not in self._df.columns:
                continue
            # Coerce to numeric for reliable comparison (keeps NaN as NaN)
            numeric_col = pd.to_numeric(self._df[col], errors="coerce")
            mask = numeric_col == sentinel
            if not mask.any():
                continue
            if strategy == "median":
                good = numeric_col[~mask].dropna()
                val = float(good.median()) if not good.empty else 0.0
            elif strategy == "mean":
                good = numeric_col[~mask].dropna()
                val = float(good.mean()) if not good.empty else 0.0
            elif strategy == "constant":
                val = fill_value if fill_value is not None else 0
            else:
                good = numeric_col[~mask].dropna()
                val = float(good.median()) if not good.empty else 0.0
            self._df.loc[mask, col] = val
            affected += int(mask.sum())
        return f"Replaced {affected} sentinel ({sentinel}) values."

    def _op_deduplicate(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Drop duplicate rows, optionally keyed on specific columns."""
        subset = cols if cols else None
        keep = params.get("keep", "first")
        before = len(self._df)
        self._df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        removed = before - len(self._df)
        return f"Removed {removed} duplicate rows."

    def _op_standardize_phone(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Normalise phone numbers to E.164 format (+1XXXXXXXXXX)."""
        import re
        target = cols if cols else ["phone"]
        affected = 0
        for col in target:
            if col not in self._df.columns:
                continue
            def _clean(v: Any) -> Any:
                if pd.isna(v):
                    return v
                digits = re.sub(r"\D", "", str(v))
                if len(digits) == 10:
                    return f"+1{digits}"
                elif len(digits) == 11 and digits.startswith("1"):
                    return f"+{digits}"
                return v  # can't normalise — leave as-is
            orig = self._df[col].copy()
            self._df[col] = self._df[col].apply(_clean)
            affected += int((self._df[col] != orig).sum())
        return f"Standardized {affected} phone numbers to E.164 format."

    def _op_lowercase_email(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Convert email columns to lowercase."""
        target = cols if cols else ["email"]
        affected = 0
        for col in target:
            if col not in self._df.columns:
                continue
            mask = self._df[col].notna() & (self._df[col] != self._df[col].str.lower())
            self._df.loc[mask, col] = self._df.loc[mask, col].str.lower()
            affected += int(mask.sum())
        return f"Lowercased {affected} email addresses."

    def _op_clamp_range(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Clamp numeric values to [min_val, max_val]."""
        min_val = params.get("min_val", None)
        max_val = params.get("max_val", None)
        target = cols if cols else list(self._df.select_dtypes(include="number").columns)
        affected = 0
        for col in target:
            if col not in self._df.columns:
                continue
            orig = self._df[col].copy()
            self._df[col] = self._df[col].clip(lower=min_val, upper=max_val)
            affected += int((self._df[col] != orig).sum())
        return f"Clamped {affected} values to [{min_val}, {max_val}]."

    def _op_fix_date_order(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Fix rows where dob > admission_date by swapping or nulling."""
        if "dob" not in self._df.columns or "admission_date" not in self._df.columns:
            return "Columns 'dob'/'admission_date' not found."
        mode = params.get("mode", "swap")
        self._df["dob"] = pd.to_datetime(self._df["dob"], errors="coerce")
        self._df["admission_date"] = pd.to_datetime(
            self._df["admission_date"], errors="coerce"
        )
        mask = self._df["dob"] > self._df["admission_date"]
        count = int(mask.sum())
        if mode == "swap":
            self._df.loc[mask, ["dob", "admission_date"]] = (
                self._df.loc[mask, ["admission_date", "dob"]].values
            )
        elif mode == "null_dob":
            self._df.loc[mask, "dob"] = pd.NaT
        return f"Fixed {count} date-order violations (mode='{mode}')."

    def _op_fix_icd10(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Normalise ICD-10 codes to uppercase + decimal format (e.g. A01.1)."""
        import re
        target = cols if cols else ["icd10_code"]
        affected = 0
        pattern = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,2})?$")
        for col in target:
            if col not in self._df.columns:
                continue
            def _fix(v: Any) -> Any:
                if pd.isna(v):
                    return v
                s = str(v).strip().upper().replace(" ", "")
                # Insert decimal if missing: e.g. A011 -> A01.1
                if re.match(r"^[A-Z]\d{3,4}$", s):
                    s = s[:3] + "." + s[3:]
                return s if pattern.match(s) else v
            orig = self._df[col].copy()
            self._df[col] = self._df[col].apply(_fix)
            affected += int((self._df[col] != orig).sum())
        return f"Fixed {affected} ICD-10 code format errors."

    def _op_fix_negative_dosage(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Replace negative dosage values with their absolute value."""
        target = cols if cols else ["dosage_mg"]
        affected = 0
        for col in target:
            if col not in self._df.columns:
                continue
            mask = pd.to_numeric(self._df[col], errors="coerce") < 0
            self._df.loc[mask, col] = self._df.loc[mask, col].abs()
            affected += int(mask.sum())
        return f"Fixed {affected} negative dosage values."

    def _op_fill_mandatory(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Fill blank mandatory fields with a supplied default value."""
        fill_value = params.get("value", "UNKNOWN")
        target = cols if cols else []
        affected = 0
        for col in target:
            if col not in self._df.columns:
                continue
            mask = self._df[col].isna() | (self._df[col].astype(str).str.strip() == "")
            self._df.loc[mask, col] = fill_value
            affected += int(mask.sum())
        return f"Filled {affected} blank mandatory fields with '{fill_value}'."

    def _op_drop_column(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Drop specified columns (destructive — carries a penalty)."""
        dropped = [c for c in cols if c in self._df.columns]
        self._df.drop(columns=dropped, inplace=True, errors="ignore")
        return f"Dropped columns: {dropped}."

    def _op_validate(self, cols: List[str], params: Dict[str, Any]) -> str:
        """No-op validation step — used to trigger grading observations."""
        grade = self._grader.grade(self._df)
        return f"Validation pass. Current grade={grade:.3f}."

    def _op_export(self, cols: List[str], params: Dict[str, Any]) -> str:
        """Signal episode completion — agent is satisfied with the result."""
        return "Export requested. Episode will end."

    # ------------------------------------------------------------------
    # Observation / defect helpers
    # ------------------------------------------------------------------

    def _build_observation(self, message: str = "") -> Observation:
        """Construct a full Observation from current DataFrame state."""
        df = self._df
        current_defects = self._count_defects(df)
        reduction_pct = 0.0
        if self._initial_defects > 0:
            reduction_pct = (
                (self._initial_defects - current_defects) / self._initial_defects * 100.0
            )

        col_stats = []
        for col in df.columns:
            null_c = int(df[col].isnull().sum())
            unique_c = int(df[col].nunique(dropna=True))
            samples = df[col].dropna().head(5).tolist()
            col_stats.append(
                ColumnStats(
                    column=col,
                    dtype=str(df[col].dtype),
                    null_count=null_c,
                    null_pct=round(null_c / max(len(df), 1) * 100, 2),
                    unique_count=unique_c,
                    sample_values=samples,
                )
            )

        sample_rows = (
            df.head(5)
            .apply(lambda col: col.map(lambda v: "NULL" if pd.isna(v) else v))
            .to_dict(orient="records")
        )


        # Defect counts (task-specific logic via grader)
        g = self._grader
        obs_kwargs: Dict[str, Any] = {
            "task_id": self._task_id,
            "step_number": self._step_number,
            "max_steps": self._config.max_steps,
            "missing_count": int(df.isnull().sum().sum()),
            "sentinel_count": g.sentinel_count(df) if hasattr(g, "sentinel_count") else 0,
            "price_below_zero": g.price_below_zero(df) if hasattr(g, "price_below_zero") else 0,
            "duplicate_count": g.duplicate_count(df) if hasattr(g, "duplicate_count") else 0,
            "phone_format_errors": g.phone_format_errors(df) if hasattr(g, "phone_format_errors") else 0,
            "email_case_errors": g.email_case_errors(df) if hasattr(g, "email_case_errors") else 0,
            "age_range_violations": g.age_range_violations(df) if hasattr(g, "age_range_violations") else 0,
            "date_order_violations": g.date_order_violations(df) if hasattr(g, "date_order_violations") else 0,
            "icd10_format_errors": g.icd10_format_errors(df) if hasattr(g, "icd10_format_errors") else 0,
            "negative_dosage_count": g.negative_dosage_count(df) if hasattr(g, "negative_dosage_count") else 0,
            "blank_mandatory_count": g.blank_mandatory_count(df) if hasattr(g, "blank_mandatory_count") else 0,
            "cross_column_violations": g.cross_column_violations(df) if hasattr(g, "cross_column_violations") else 0,
            "column_stats": col_stats,
            "sample_rows": sample_rows,
            "total_defects": current_defects,
            "defect_reduction_pct": round(reduction_pct, 2),
            "done": self._done,
            "message": message,
        }
        return Observation(**obs_kwargs)

    def _count_defects(self, df: pd.DataFrame) -> int:
        """Return total tracked defect count from the active grader."""
        if self._grader is None:
            return int(df.isnull().sum().sum())
        return self._grader.count_defects(df)

    def _is_done(self, grade: float, action: Action) -> bool:
        """Return True if the episode should terminate."""
        if self._step_number >= self._config.max_steps:
            return True
        if action.operation == OperationType.EXPORT:
            return True
        if grade >= self._config.pass_threshold:
            return True
        return False

    def _record_step(
        self, action_dict: Dict[str, Any], reward_val: float, done: bool
    ) -> None:
        """Append a step record to the episode history."""
        self._episode_history.append(
            EpisodeStep(
                step_number=self._step_number,
                action=action_dict,
                reward=reward_val,
                cumulative_reward=self._cumulative_reward,
                done=done,
                timestamp=time.time(),
            )
        )
