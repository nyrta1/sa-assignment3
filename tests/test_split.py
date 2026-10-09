import pandas as pd
import pytest

from oulad_risk.features import build_features
from oulad_risk.split import audit_overlap, group_split, naive_stratified_split


@pytest.fixture(scope="module")
def fs(tables):
    return build_features(tables)


def test_group_split_has_zero_student_overlap(fs):
    tr, te = group_split(fs.groups, 0.2, 42)
    report = audit_overlap(fs.groups, tr, te)
    assert report.is_clean
    assert report.leaked_row_share == 0.0
    assert len(set(tr) | set(te)) == len(fs)


def test_naive_split_leaks_students(fs):
    """Students in two modules end up on both sides of a row-level split."""
    report = audit_overlap(fs.groups, *naive_stratified_split(fs.y, 0.2, 42))
    assert report.n_shared_students > 0


def test_split_is_deterministic(fs):
    a = group_split(fs.groups, 0.2, 7)
    b = group_split(fs.groups, 0.2, 7)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_audit_counts_by_hand():
    groups = pd.Series([1, 1, 2, 3, 3, 4])
    rep = audit_overlap(groups, train_idx=[0, 2, 3], test_idx=[1, 4, 5])
    assert rep.n_shared_students == 2  # students 1 and 3
    assert rep.n_test_rows_leaked == 2
    assert rep.leaked_row_share == pytest.approx(2 / 3)
