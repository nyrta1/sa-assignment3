"""Feature tests - most importantly, the temporal-leakage guarantees."""

import numpy as np
import pandas as pd
import pytest

from oulad_risk.config import FORBIDDEN_FEATURES
from oulad_risk.features import build_features, make_target


def test_target_mapping():
    y = make_target(pd.Series(["Pass", "Fail", "Withdrawn", "Distinction"]))
    assert y.tolist() == [0, 1, 1, 0]


def test_target_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Unexpected"):
        make_target(pd.Series(["Pass", "Dropped"]))


def test_shapes_and_no_forbidden_columns(tables):
    fs = build_features(tables, cutoff_day=30)
    assert len(fs.X) == len(fs.y) == len(fs.groups)
    assert not FORBIDDEN_FEATURES.intersection(fs.X.columns)
    assert set(fs.y.unique()) <= {0, 1}


def test_future_records_do_not_change_features(tables):
    """Core temporal-leakage test: mutate everything AFTER the cut-off -> identical X."""
    cutoff = 30
    base = build_features(tables, cutoff_day=cutoff).X

    mutated = {k: v.copy() for k, v in tables.items()}
    vle = mutated["studentVle"]
    vle.loc[vle["date"] > cutoff, "sum_click"] *= 100
    sa = mutated["studentAssessment"]
    sa.loc[sa["date_submitted"] > cutoff, "score"] = 0
    after = build_features(mutated, cutoff_day=cutoff).X

    pd.testing.assert_frame_equal(base, after)


def test_past_records_do_change_features(tables):
    """Sanity check of the test above: mutating data BEFORE the cut-off must matter."""
    mutated = {k: v.copy() for k, v in tables.items()}
    vle = mutated["studentVle"]
    vle.loc[vle["date"] <= 30, "sum_click"] *= 100
    assert not build_features(tables).X.equals(build_features(mutated).X)


def test_withdrawn_before_cutoff_are_excluded(tables):
    reg = tables["studentRegistration"]
    n_gone = int(reg["date_unregistration"].le(30).sum())
    kept = build_features(tables, 30, exclude_withdrawn_before_cutoff=True)
    all_rows = build_features(tables, 30, exclude_withdrawn_before_cutoff=False)
    assert len(all_rows) - len(kept) == n_gone > 0


def test_counts_are_non_negative_and_rates_bounded(tables):
    X = build_features(tables).X
    for col in ["clicks_total", "active_days", "n_assess_submitted", "n_late_submissions"]:
        assert (X[col] >= 0).all()
    rate = X["submission_rate"].dropna()
    assert ((rate >= 0) & (rate <= 1)).all()
    assert (X["active_days"] <= 31).all()


def test_works_with_student_info_only(tables):
    fs = build_features({"studentInfo": tables["studentInfo"]})
    assert "clicks_total" not in fs.X.columns
    assert np.isfinite(fs.X[fs.numeric].to_numpy(dtype=float)).all()
