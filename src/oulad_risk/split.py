"""Group-aware data splitting and the student-overlap leakage audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class OverlapReport:
    """Result of auditing a train/test split for students present on both sides."""

    n_train_students: int
    n_test_students: int
    n_shared_students: int
    n_test_rows_leaked: int
    n_test_rows: int

    @property
    def leaked_row_share(self) -> float:
        """Share of test rows whose student also appears in the training set."""
        return self.n_test_rows_leaked / self.n_test_rows if self.n_test_rows else 0.0

    @property
    def is_clean(self) -> bool:
        return self.n_shared_students == 0


def group_split(
    groups: pd.Series, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices so that every student (group) lands entirely on one side."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx


def naive_stratified_split(
    y: pd.Series, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """The common but flawed row-level split, kept only to *demonstrate* group leakage."""
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )
    return train_idx, test_idx


def audit_overlap(groups: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray) -> OverlapReport:
    """Count students that appear in both partitions of a split."""
    train_ids = set(groups.iloc[train_idx])
    test_groups = groups.iloc[test_idx]
    shared = train_ids.intersection(test_groups)
    return OverlapReport(
        n_train_students=len(train_ids),
        n_test_students=int(test_groups.nunique()),
        n_shared_students=len(shared),
        n_test_rows_leaked=int(test_groups.isin(shared).sum()),
        n_test_rows=len(test_groups),
    )
