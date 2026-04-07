"""
Dense, Step-Wise Reward Calculator
====================================

Design goals:
  1. Dense signal — every step returns a non-zero reward (progress or penalty)
  2. Partial credit — rewards proportional to defect reduction, not binary
  3. Penalises invalid actions, redundant steps, and destructive operations
  4. Completion bonus when the agent achieves the task success threshold
  5. Strictly clamped to [-1.0, 1.0] per step; cumulative clamped to [0.0, 1.0]

Reward components
-----------------
progress_reward     : +0.0 – +0.30  based on defect reduction fraction
step_penalty        : -0.01          living-cost per step (encourages efficiency)
invalid_penalty     : -0.05          malformed/illegal action submitted
destructive_penalty : -0.10          drop_column used on required columns
completion_bonus    : +0.20          awarded once when grade ≥ success_threshold
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .models import Action, OperationType, Reward, RewardBreakdown, TaskConfig


# ---------------------------------------------------------------------------
# Reward constants (tune-able)
# ---------------------------------------------------------------------------
_STEP_PENALTY = -0.01
_INVALID_PENALTY = -0.05
_DESTRUCTIVE_PENALTY = -0.10
_MAX_PROGRESS_REWARD = 0.30
_COMPLETION_BONUS = 0.20

# Operations that carry a destructive penalty regardless of effect
_DESTRUCTIVE_OPS = {OperationType.DROP_COLUMN}


class RewardCalculator:
    """
    Computes step-wise rewards for the DataCleaningEnv.

    Parameters
    ----------
    config : TaskConfig
        Active task configuration (used for success_threshold).
    """

    def __init__(self, config: TaskConfig) -> None:
        self._config = config
        self._completion_bonus_awarded = False

    def compute(
        self,
        action: Action,
        defects_before: int,
        defects_after: int,
        initial_defects: int,
        grade: float,
        step_number: int,
        exec_error: Optional[str] = None,
    ) -> Reward:
        """
        Compute the reward for a single step.

        Parameters
        ----------
        action : Action
            The validated action that was executed.
        defects_before : int
            Total defect count before the action.
        defects_after : int
            Total defect count after the action.
        initial_defects : int
            Defect count at episode start (used for normalisation).
        grade : float
            Current grader score in [0.0, 1.0].
        step_number : int
            Current step number (1-indexed after increment).
        exec_error : str or None
            Runtime error string if the action failed (else None).

        Returns
        -------
        Reward
            Step reward with itemised breakdown; cumulative not yet set.
        """
        breakdown = RewardBreakdown()

        # --- Step penalty (always) ---
        breakdown.step_penalty = _STEP_PENALTY

        # --- Invalid or errored action ---
        if exec_error:
            breakdown.invalid_action_penalty = _INVALID_PENALTY
            total = breakdown.step_penalty + breakdown.invalid_action_penalty
            return Reward(
                value=float(np.clip(total, -1.0, 1.0)),
                breakdown=breakdown,
                cumulative=0.0,
            )

        # --- Destructive operation ---
        if action.operation in _DESTRUCTIVE_OPS:
            breakdown.destructive_penalty = _DESTRUCTIVE_PENALTY

        # --- Progress reward ---
        progress_reward = self._progress_reward(
            defects_before, defects_after, initial_defects
        )
        breakdown.progress_reward = progress_reward

        # --- Completion bonus (one-time) ---
        if (
            grade >= self._config.success_threshold
            and not self._completion_bonus_awarded
        ):
            breakdown.completion_bonus = _COMPLETION_BONUS
            self._completion_bonus_awarded = True

        # --- Total ---
        total = (
            breakdown.progress_reward
            + breakdown.step_penalty
            + breakdown.invalid_action_penalty
            + breakdown.destructive_penalty
            + breakdown.completion_bonus
        )
        breakdown.total = float(np.clip(total, -1.0, 1.0))

        return Reward(
            value=breakdown.total,
            breakdown=breakdown,
            cumulative=0.0,  # filled in by env.step()
        )

    def invalid_action_penalty(self) -> Reward:
        """
        Return a penalty Reward for a completely invalid action (parse error).

        Used when Action model validation fails before execution.
        """
        breakdown = RewardBreakdown(
            step_penalty=_STEP_PENALTY,
            invalid_action_penalty=_INVALID_PENALTY,
        )
        total = _STEP_PENALTY + _INVALID_PENALTY
        breakdown.total = float(np.clip(total, -1.0, 1.0))
        return Reward(value=breakdown.total, breakdown=breakdown, cumulative=0.0)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _progress_reward(
        self,
        defects_before: int,
        defects_after: int,
        initial_defects: int,
    ) -> float:
        """
        Dense progress reward proportional to the fraction of defects fixed.

        Formula: progress = (before - after) / max(initial, 1)
        Reward   = MAX_PROGRESS_REWARD * progress   (capped at MAX_PROGRESS_REWARD)
        Negative progress (agent introduces new defects) yields 0 (no bonus).
        Redundant step (no change) yields 0.

        Returns
        -------
        float
            Progress reward in [0.0, MAX_PROGRESS_REWARD].
        """
        if initial_defects == 0:
            return 0.0
        delta = defects_before - defects_after
        if delta <= 0:
            return 0.0
        fraction = delta / max(initial_defects, 1)
        reward = _MAX_PROGRESS_REWARD * fraction
        return float(min(reward, _MAX_PROGRESS_REWARD))
