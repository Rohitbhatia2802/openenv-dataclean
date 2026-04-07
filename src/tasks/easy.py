"""
Easy Task: Fix Missing Price Values
====================================

Dataset: Retail transaction log (~200 rows)
Defects:
  1. ~20% of 'price' column is NaN
  2. ~10% of 'price' column is set to sentinel -1
  3. Prices of 0 are also invalid

Target: Impute all missing/sentinel/zero prices with the column median
of valid (> 0) prices.

All data is generated deterministically via np.random.seed(42).
No external downloads required at runtime.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd


class EasyTask:
    """
    Generates the retail transaction dataset for the Easy difficulty task.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility (default 42).
    n_rows : int
        Number of rows to generate (default 200).
    """

    CATEGORIES = [
        "Electronics", "Clothing", "Groceries",
        "Books", "Sports", "Home", "Toys",
    ]

    BASE_PRICES: dict[str, float] = {
        "Electronics": 299.99,
        "Clothing": 49.99,
        "Groceries": 12.50,
        "Books": 19.99,
        "Sports": 89.99,
        "Home": 74.99,
        "Toys": 34.99,
    }

    def __init__(self, seed: int = 42, n_rows: int = 200) -> None:
        self._seed = seed
        self._n_rows = n_rows

    def generate_dataset(self) -> pd.DataFrame:
        """
        Build the retail transaction DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: transaction_id, category, quantity, price, timestamp
        """
        rng = np.random.default_rng(self._seed)

        categories = rng.choice(self.CATEGORIES, size=self._n_rows)
        quantities = rng.integers(1, 10, size=self._n_rows)

        # Build prices with Gaussian noise
        base = np.array([self.BASE_PRICES[c] for c in categories])
        noise = rng.normal(0, base * 0.1)
        prices = np.round(base + noise, 2)
        prices = np.clip(prices, 5.0, None)  # floor at $5

        # Introduce defects
        n_nulls = int(self._n_rows * 0.20)          # 20% NaN
        n_sentinels = int(self._n_rows * 0.10)       # 10% sentinel -1

        null_idx = rng.choice(self._n_rows, size=n_nulls, replace=False)
        remaining = np.setdiff1d(np.arange(self._n_rows), null_idx)
        sentinel_idx = rng.choice(remaining, size=n_sentinels, replace=False)

        price_series = prices.astype(object)
        price_series[null_idx] = np.nan
        price_series[sentinel_idx] = -1.0

        # Timestamps
        base_ts = pd.Timestamp("2024-01-01")
        offsets = rng.integers(0, 365 * 24 * 3600, size=self._n_rows)
        timestamps = [
            (base_ts + pd.Timedelta(seconds=int(o))).strftime("%Y-%m-%d %H:%M:%S")
            for o in offsets
        ]

        df = pd.DataFrame(
            {
                "transaction_id": [f"TXN-{i:05d}" for i in range(self._n_rows)],
                "category": categories,
                "quantity": quantities.astype(int),
                "price": price_series,
                "timestamp": timestamps,
            }
        )

        return df

    def describe(self) -> dict[str, Any]:
        """Return a human-readable task description dictionary."""
        return {
            "task": "fix_missing_price",
            "difficulty": "easy",
            "rows": self._n_rows,
            "target_column": "price",
            "defect_types": ["null_values", "sentinel_-1"],
            "fix_strategy": "replace with column median of valid prices",
        }
