"""
Hard Task: Validate Medical Records
=====================================

Dataset: Hospital EMR export (~400 rows)
Defects (5 types):
  1. Date-order violations: dob > admission_date (patient born after admission)
  2. ICD-10 code format errors: missing decimal, wrong length, lowercase
  3. Negative dosage values (dosage_mg < 0)
  4. Blank mandatory fields: patient_id, diagnosis
  5. Cross-column constraint violations: discharge_date < admission_date

Target: Resolve all 5 defect types; minimise destructive edits.

Step budget is tight (50) requiring prioritised, efficient operations.

All data is generated deterministically. No external downloads at runtime.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Valid ICD-10 code pool (real codes in simplified format)
# ---------------------------------------------------------------------------
_VALID_ICD10_CODES = [
    "A01.1", "B02.0", "C34.1", "D50.0", "E11.9",
    "F32.1", "G43.9", "H10.0", "I21.0", "J18.9",
    "K21.0", "L03.0", "M79.3", "N39.0", "O80.0",
    "P07.1", "Q21.0", "R05.0", "S52.5", "T14.0",
    "U07.1", "V01.0", "W50.0", "X00.0", "Z00.0",
]

_MALFORMED_ICD10 = [
    "a011",   # lowercase + missing decimal
    "B020",   # missing decimal
    "c34",    # too short + lowercase
    "D5000",  # too long
    "e119",   # lowercase + missing decimal
    "F321",   # missing decimal
    "g439",   # lowercase + missing decimal
    "H100",   # missing decimal
    "i210",   # lowercase + missing decimal
    "J189",   # missing decimal
]


class HardTask:
    """
    Generates the hospital EMR dataset for the Hard difficulty task.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_rows : int
        Number of records to generate (default 400).
    """

    DIAGNOSES = [
        "Pneumonia", "Type 2 Diabetes", "Appendicitis",
        "Hypertension", "Fracture", "Sepsis", "Stroke",
        "Heart Failure", "COPD", "Renal Failure",
    ]

    WARDS = ["ICU", "General", "Pediatrics", "Oncology", "Surgery", "Cardiology"]

    def __init__(self, seed: int = 42, n_rows: int = 400) -> None:
        self._seed = seed
        self._n_rows = n_rows

    def generate_dataset(self) -> pd.DataFrame:
        """
        Build the hospital EMR DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: patient_id, dob, admission_date, discharge_date,
                     diagnosis, icd10_code, dosage_mg, ward
        """
        rng = np.random.default_rng(self._seed)

        # Base clean dates
        base_date = pd.Timestamp("1960-01-01")
        dob_offsets = rng.integers(0, 50 * 365, size=self._n_rows)  # 1960–2010
        dobs = [base_date + pd.Timedelta(days=int(d)) for d in dob_offsets]

        adm_base = pd.Timestamp("2022-01-01")
        adm_offsets = rng.integers(0, 2 * 365, size=self._n_rows)
        admissions = [adm_base + pd.Timedelta(days=int(d)) for d in adm_offsets]

        # Discharge = admission + 1..14 days
        los = rng.integers(1, 15, size=self._n_rows)
        discharges = [
            adm + pd.Timedelta(days=int(stay))
            for adm, stay in zip(admissions, los)
        ]

        # Dosages (mg)
        dosages = np.round(rng.uniform(10.0, 500.0, size=self._n_rows), 2)

        # ICD-10 codes (clean)
        icd_codes = rng.choice(_VALID_ICD10_CODES, size=self._n_rows)

        diagnoses = rng.choice(self.DIAGNOSES, size=self._n_rows)
        wards = rng.choice(self.WARDS, size=self._n_rows)

        df = pd.DataFrame(
            {
                "patient_id": [f"PAT-{i:06d}" for i in range(self._n_rows)],
                "dob": [d.strftime("%Y-%m-%d") for d in dobs],
                "admission_date": [d.strftime("%Y-%m-%d") for d in admissions],
                "discharge_date": [d.strftime("%Y-%m-%d") for d in discharges],
                "diagnosis": diagnoses,
                "icd10_code": icd_codes,
                "dosage_mg": dosages,
                "ward": wards,
            }
        )

        # ---------------------------------------------------------------
        # Inject defects
        # ---------------------------------------------------------------

        # 1. Date-order violations: swap dob and admission_date (~5%)
        n_date_ord = int(self._n_rows * 0.05)
        date_ord_idx = rng.choice(self._n_rows, size=n_date_ord, replace=False)
        for i in date_ord_idx:
            df.loc[i, "dob"], df.loc[i, "admission_date"] = (
                df.loc[i, "admission_date"],
                df.loc[i, "dob"],
            )

        # 2. ICD-10 format errors (~12%)
        n_icd = int(self._n_rows * 0.12)
        icd_idx = rng.choice(self._n_rows, size=n_icd, replace=False)
        bad_icd = rng.choice(_MALFORMED_ICD10, size=n_icd)
        df.loc[icd_idx, "icd10_code"] = bad_icd

        # 3. Negative dosages (~8%)
        n_neg = int(self._n_rows * 0.08)
        neg_idx = rng.choice(self._n_rows, size=n_neg, replace=False)
        df.loc[neg_idx, "dosage_mg"] = -df.loc[neg_idx, "dosage_mg"]

        # 4. Blank mandatory fields (~6%)
        n_blank = int(self._n_rows * 0.06)
        blank_pid_idx = rng.choice(self._n_rows, size=n_blank // 2, replace=False)
        blank_diag_idx = rng.choice(
            self._n_rows, size=n_blank - n_blank // 2, replace=False
        )
        df.loc[blank_pid_idx, "patient_id"] = ""
        df.loc[blank_diag_idx, "diagnosis"] = np.nan

        # 5. Cross-column constraints: discharge < admission (~4%)
        n_cross = int(self._n_rows * 0.04)
        cross_idx = rng.choice(self._n_rows, size=n_cross, replace=False)
        for i in cross_idx:
            adm = pd.to_datetime(df.loc[i, "admission_date"])
            df.loc[i, "discharge_date"] = (adm - pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )

        return df

    def describe(self) -> dict[str, Any]:
        """Return a human-readable task description dictionary."""
        return {
            "task": "validate_medical_records",
            "difficulty": "hard",
            "defect_types": [
                "date_order_violations",
                "icd10_format_errors",
                "negative_dosage",
                "blank_mandatory_fields",
                "cross_column_constraint_violations",
            ],
            "operations_needed": [
                "fix_date_order",
                "fix_icd10",
                "fix_negative_dosage",
                "fill_mandatory",
                "validate",
            ],
        }
