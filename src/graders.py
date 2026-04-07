"""
Deterministic Graders for each OpenEnv task tier.

All graders:
  - Are purely rule-based (pandas / regex / schema validation)
  - Return float ∈ [0.0, 1.0]
  - Are 100% deterministic — no LLM judgment
  - Expose helper methods used by env.py to build observations

Classes
-------
BaseGrader
    Abstract base providing the count_defects interface.
EasyGrader
    Grades the 'fix_missing_price' task.
MediumGrader
    Grades the 'normalize_customer_pipeline' task.
HardGrader
    Grades the 'validate_medical_records' task.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Base grader
# ---------------------------------------------------------------------------

class BaseGrader(ABC):
    """Abstract grader base class."""

    @abstractmethod
    def grade(self, df: pd.DataFrame) -> float:
        """Return a score in [0.0, 1.0] for the current DataFrame state."""
        ...

    @abstractmethod
    def count_defects(self, df: pd.DataFrame) -> int:
        """Return total tracked defect count (used for reward shaping)."""
        ...


# ---------------------------------------------------------------------------
# Easy Grader
# ---------------------------------------------------------------------------

class EasyGrader(BaseGrader):
    """
    Grader for the 'fix_missing_price' (Easy) task.

    Scoring rubric
    --------------
    * 50% weight — no null values in 'price' column
    * 30% weight — no sentinel -1 values in 'price' column
    * 20% weight — all prices > 0 (no zero or negative prices remaining)

    Final score = weighted sum, clamped to [0.0, 1.0].
    """

    def grade(self, df: pd.DataFrame) -> float:
        """
        Grade the DataFrame for the easy task.

        Parameters
        ----------
        df : pd.DataFrame
            Current state of the retail transaction DataFrame.

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if "price" not in df.columns or len(df) == 0:
            return 0.0

        total = len(df)
        price = pd.to_numeric(df["price"], errors="coerce")

        null_c = int(price.isnull().sum())
        sentinel_c = int((price == -1).sum())
        below_zero_c = int((price <= 0).sum())

        null_score = 1.0 - (null_c / total)
        sentinel_score = 1.0 - (sentinel_c / total)
        positive_score = 1.0 - (below_zero_c / total)

        weighted = (
            0.50 * null_score
            + 0.30 * sentinel_score
            + 0.20 * positive_score
        )
        return float(np.clip(weighted, 0.0, 1.0))

    def count_defects(self, df: pd.DataFrame) -> int:
        """Sum of null prices, sentinel -1 prices, and non-positive prices."""
        if "price" not in df.columns:
            return 0
        price = pd.to_numeric(df["price"], errors="coerce")
        return (
            int(price.isnull().sum())
            + int((price == -1).sum())
            + int((price <= 0).sum())
        )

    # --- Observation helper methods ---

    def sentinel_count(self, df: pd.DataFrame) -> int:
        """Number of cells with sentinel value -1 in 'price'."""
        if "price" not in df.columns:
            return 0
        price = pd.to_numeric(df["price"], errors="coerce")
        return int((price == -1).sum())

    def price_below_zero(self, df: pd.DataFrame) -> int:
        """Number of price values ≤ 0."""
        if "price" not in df.columns:
            return 0
        price = pd.to_numeric(df["price"], errors="coerce")
        return int((price <= 0).sum())

    # Stub helpers for unused defect types
    def duplicate_count(self, df: pd.DataFrame) -> int:
        return 0
    def phone_format_errors(self, df: pd.DataFrame) -> int:
        return 0
    def email_case_errors(self, df: pd.DataFrame) -> int:
        return 0
    def age_range_violations(self, df: pd.DataFrame) -> int:
        return 0
    def date_order_violations(self, df: pd.DataFrame) -> int:
        return 0
    def icd10_format_errors(self, df: pd.DataFrame) -> int:
        return 0
    def negative_dosage_count(self, df: pd.DataFrame) -> int:
        return 0
    def blank_mandatory_count(self, df: pd.DataFrame) -> int:
        return 0
    def cross_column_violations(self, df: pd.DataFrame) -> int:
        return 0


# ---------------------------------------------------------------------------
# Medium Grader
# ---------------------------------------------------------------------------

class MediumGrader(BaseGrader):
    """
    Grader for the 'normalize_customer_pipeline' (Medium) task.

    Scoring rubric
    --------------
    * 25% — no duplicate rows (keyed on email + phone)
    * 25% — all phone numbers match E.164 pattern
    * 25% — all emails are lowercase
    * 25% — all ages within [18, 100]

    Final score = mean of four sub-scores, clamped to [0.0, 1.0].
    """

    _E164_RE = re.compile(r"^\+1\d{10}$")
    _AGE_MIN = 18
    _AGE_MAX = 100

    def grade(self, df: pd.DataFrame) -> float:
        """
        Grade the DataFrame for the medium task.

        Parameters
        ----------
        df : pd.DataFrame
            Current state of the customer CRM DataFrame.

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if len(df) == 0:
            return 0.0

        total = len(df)

        # Sub-score 1: duplicates
        dup_c = self.duplicate_count(df)
        dup_score = 1.0 - (dup_c / total)

        # Sub-score 2: phone format
        phone_err = self.phone_format_errors(df)
        phone_n = total if "phone" in df.columns else 1
        phone_score = 1.0 - (phone_err / phone_n)

        # Sub-score 3: email case
        email_err = self.email_case_errors(df)
        email_n = total if "email" in df.columns else 1
        email_score = 1.0 - (email_err / email_n)

        # Sub-score 4: age range
        age_err = self.age_range_violations(df)
        age_n = total if "age" in df.columns else 1
        age_score = 1.0 - (age_err / age_n)

        final = (dup_score + phone_score + email_score + age_score) / 4.0
        return float(np.clip(final, 0.0, 1.0))

    def count_defects(self, df: pd.DataFrame) -> int:
        """Total count across all four defect types."""
        return (
            self.duplicate_count(df)
            + self.phone_format_errors(df)
            + self.email_case_errors(df)
            + self.age_range_violations(df)
        )

    # --- Observation helpers ---

    def duplicate_count(self, df: pd.DataFrame) -> int:
        """Number of fully-duplicate rows."""
        return int(df.duplicated().sum())

    def phone_format_errors(self, df: pd.DataFrame) -> int:
        """Number of phone values not matching E.164 (+1XXXXXXXXXX)."""
        if "phone" not in df.columns:
            return 0
        phones = df["phone"].dropna().astype(str)
        return int((~phones.str.match(self._E164_RE)).sum())

    def email_case_errors(self, df: pd.DataFrame) -> int:
        """Number of email values that are not fully lowercase."""
        if "email" not in df.columns:
            return 0
        emails = df["email"].dropna().astype(str)
        return int((emails != emails.str.lower()).sum())

    def age_range_violations(self, df: pd.DataFrame) -> int:
        """Number of ages outside [18, 100]."""
        if "age" not in df.columns:
            return 0
        ages = pd.to_numeric(df["age"], errors="coerce")
        return int(((ages < self._AGE_MIN) | (ages > self._AGE_MAX)).sum())

    # Stubs for unused defects
    def sentinel_count(self, df: pd.DataFrame) -> int:
        return 0
    def price_below_zero(self, df: pd.DataFrame) -> int:
        return 0
    def date_order_violations(self, df: pd.DataFrame) -> int:
        return 0
    def icd10_format_errors(self, df: pd.DataFrame) -> int:
        return 0
    def negative_dosage_count(self, df: pd.DataFrame) -> int:
        return 0
    def blank_mandatory_count(self, df: pd.DataFrame) -> int:
        return 0
    def cross_column_violations(self, df: pd.DataFrame) -> int:
        return 0


# ---------------------------------------------------------------------------
# Hard Grader
# ---------------------------------------------------------------------------

class HardGrader(BaseGrader):
    """
    Grader for the 'validate_medical_records' (Hard) task.

    Scoring rubric (5 equal weights of 20% each)
    ----------------------------------------------
    * 20% — no date-order violations (dob ≤ admission_date)
    * 20% — all ICD-10 codes match "[A-Z]\\d{2}\\.\\d{1,2}" pattern
    * 20% — no negative dosage_mg values
    * 20% — no blank mandatory fields (patient_id, diagnosis)
    * 20% — no cross-column constraint violations (discharge ≥ admission)

    Final score = mean of five sub-scores, clamped to [0.0, 1.0].
    """

    _ICD10_RE = re.compile(r"^[A-Z]\d{2}\.\d{1,2}$")
    _MANDATORY_COLS = ["patient_id", "diagnosis"]

    def grade(self, df: pd.DataFrame) -> float:
        """
        Grade the DataFrame for the hard task.

        Parameters
        ----------
        df : pd.DataFrame
            Current state of the hospital EMR DataFrame.

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if len(df) == 0:
            return 0.0

        total = len(df)

        date_err = self.date_order_violations(df)
        date_score = 1.0 - (date_err / total)

        icd_err = self.icd10_format_errors(df)
        icd_score = 1.0 - (icd_err / total)

        neg_err = self.negative_dosage_count(df)
        neg_score = 1.0 - (neg_err / total)

        blank_err = self.blank_mandatory_count(df)
        # blank counts across multiple mandatory columns — normalise per-cell
        mandatory_cells = total * len(self._MANDATORY_COLS)
        blank_score = 1.0 - (blank_err / max(mandatory_cells, 1))

        cross_err = self.cross_column_violations(df)
        cross_score = 1.0 - (cross_err / total)

        final = (
            date_score + icd_score + neg_score + blank_score + cross_score
        ) / 5.0
        return float(np.clip(final, 0.0, 1.0))

    def count_defects(self, df: pd.DataFrame) -> int:
        """Total count across all five defect types."""
        return (
            self.date_order_violations(df)
            + self.icd10_format_errors(df)
            + self.negative_dosage_count(df)
            + self.blank_mandatory_count(df)
            + self.cross_column_violations(df)
        )

    # --- Observation helpers ---

    def date_order_violations(self, df: pd.DataFrame) -> int:
        """Rows where dob > admission_date."""
        if "dob" not in df.columns or "admission_date" not in df.columns:
            return 0
        dob = pd.to_datetime(df["dob"], errors="coerce")
        adm = pd.to_datetime(df["admission_date"], errors="coerce")
        mask = dob.notna() & adm.notna() & (dob > adm)
        return int(mask.sum())

    def icd10_format_errors(self, df: pd.DataFrame) -> int:
        """Rows where icd10_code does not match [A-Z]\\d{2}\\.\\d{1,2}."""
        if "icd10_code" not in df.columns:
            return 0
        codes = df["icd10_code"].fillna("").astype(str)
        return int((~codes.str.match(self._ICD10_RE)).sum())

    def negative_dosage_count(self, df: pd.DataFrame) -> int:
        """Rows with dosage_mg < 0."""
        if "dosage_mg" not in df.columns:
            return 0
        dosage = pd.to_numeric(df["dosage_mg"], errors="coerce")
        return int((dosage < 0).sum())

    def blank_mandatory_count(self, df: pd.DataFrame) -> int:
        """Total blank/null cells across mandatory columns."""
        count = 0
        for col in self._MANDATORY_COLS:
            if col not in df.columns:
                continue
            blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
            count += int(blank.sum())
        return count

    def cross_column_violations(self, df: pd.DataFrame) -> int:
        """Rows where discharge_date < admission_date."""
        if "discharge_date" not in df.columns or "admission_date" not in df.columns:
            return 0
        disc = pd.to_datetime(df["discharge_date"], errors="coerce")
        adm = pd.to_datetime(df["admission_date"], errors="coerce")
        mask = disc.notna() & adm.notna() & (disc < adm)
        return int(mask.sum())

    # Stubs for unused defects
    def sentinel_count(self, df: pd.DataFrame) -> int:
        return 0
    def price_below_zero(self, df: pd.DataFrame) -> int:
        return 0
    def duplicate_count(self, df: pd.DataFrame) -> int:
        return 0
    def phone_format_errors(self, df: pd.DataFrame) -> int:
        return 0
    def email_case_errors(self, df: pd.DataFrame) -> int:
        return 0
    def age_range_violations(self, df: pd.DataFrame) -> int:
        return 0
