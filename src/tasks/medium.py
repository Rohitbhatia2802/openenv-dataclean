"""
Medium Task: Normalize Customer Data Pipeline
==============================================

Dataset: Customer CRM export (~300 rows)
Defects:
  1. Duplicate rows (~8% of records)
  2. Phone numbers in mixed formats (parenthetical, dashes, dots, raw 10-digit)
  3. Email addresses with mixed case (not all lowercase)
  4. Age values outside valid range [18, 100] (negative ages, ages > 120)

Target: Clean all four defect types to pass the medium grader.

All data is generated deterministically via np.random.seed(42).
No external downloads required at runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MediumTask:
    """
    Generates the Customer CRM dataset for the Medium difficulty task.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_rows : int
        Number of base rows before duplicates are inserted (default 280).
    """

    FIRST_NAMES = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace",
        "Heidi", "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj",
        "Olivia", "Peggy", "Pluto", "Romeo", "Sybil", "Trent",
    ]
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis",
        "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas",
        "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia",
    ]
    DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "proton.me"]

    def __init__(self, seed: int = 42, n_rows: int = 280) -> None:
        self._seed = seed
        self._n_rows = n_rows

    def generate_dataset(self) -> pd.DataFrame:
        """
        Build the customer CRM DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: customer_id, first_name, last_name, email, phone, age,
                     signup_date, plan
        """
        rng = np.random.default_rng(self._seed)

        first = rng.choice(self.FIRST_NAMES, size=self._n_rows)
        last = rng.choice(self.LAST_NAMES, size=self._n_rows)
        domain = rng.choice(self.DOMAINS, size=self._n_rows)

        # Emails — intentionally mixed case
        emails_raw = [
            f"{f.lower()}.{l.lower()}@{d}" for f, l, d in zip(first, last, domain)
        ]
        # Corrupt ~30% to have uppercase
        corrupt_email_idx = rng.choice(
            self._n_rows, size=int(self._n_rows * 0.30), replace=False
        )
        for i in corrupt_email_idx:
            emails_raw[i] = emails_raw[i].upper()

        # Phones — mixed formats
        area_codes = rng.integers(200, 999, size=self._n_rows)
        prefixes = rng.integers(200, 999, size=self._n_rows)
        lines = rng.integers(1000, 9999, size=self._n_rows)
        phone_format = rng.choice(4, size=self._n_rows)  # 0-3 different formats
        phones = []
        for i in range(self._n_rows):
            a, p, l, fmt = area_codes[i], prefixes[i], lines[i], phone_format[i]
            if fmt == 0:
                phones.append(f"({a}) {p}-{l}")
            elif fmt == 1:
                phones.append(f"{a}-{p}-{l}")
            elif fmt == 2:
                phones.append(f"{a}.{p}.{l}")
            else:
                phones.append(f"{a}{p}{l}")  # raw 10-digit

        # Ages — mostly valid with some outliers
        ages = rng.integers(18, 80, size=self._n_rows).astype(float)
        n_invalid_age = int(self._n_rows * 0.06)
        invalid_age_idx = rng.choice(self._n_rows, size=n_invalid_age, replace=False)
        for i in invalid_age_idx:
            ages[i] = rng.choice([-5, 0, 8, 130, 150, 200])

        # Signup dates
        base_ts = pd.Timestamp("2020-01-01")
        offsets_d = rng.integers(0, 4 * 365, size=self._n_rows)
        signup_dates = [
            (base_ts + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
            for d in offsets_d
        ]

        plans = rng.choice(["free", "basic", "pro", "enterprise"], size=self._n_rows)

        df = pd.DataFrame(
            {
                "customer_id": [f"CUST-{i:06d}" for i in range(self._n_rows)],
                "first_name": first,
                "last_name": last,
                "email": emails_raw,
                "phone": phones,
                "age": ages,
                "signup_date": signup_dates,
                "plan": plans,
            }
        )

        # Insert duplicates (~8%) — keep ALL columns identical including customer_id
        # so df.duplicated() and deduplicate both work correctly.
        n_dups = int(self._n_rows * 0.08)
        dup_idx = rng.choice(self._n_rows, size=n_dups, replace=False)
        dup_rows = df.iloc[dup_idx].copy()   # exact copies including customer_id
        df = pd.concat([df, dup_rows], ignore_index=True)
        # Shuffle to spread duplicates throughout the dataset
        df = df.sample(frac=1, random_state=self._seed).reset_index(drop=True)
        # NOTE: we intentionally do NOT re-assign customer_id so duplicates remain
        # detectable via df.duplicated().

        return df

    def describe(self) -> dict[str, Any]:
        """Return a human-readable task description dictionary."""
        return {
            "task": "normalize_customer_pipeline",
            "difficulty": "medium",
            "defect_types": [
                "duplicate_rows",
                "mixed_case_email",
                "non_e164_phone",
                "out_of_range_age",
            ],
            "operations_needed": [
                "deduplicate",
                "lowercase_email",
                "standardize_phone",
                "clamp_range",
            ],
        }
