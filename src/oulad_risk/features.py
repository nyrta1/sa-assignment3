"""Leakage-safe feature engineering.

Every behavioural feature is computed **only from records dated on or before the cut-off
day** (day 0 = official module start). This is the core temporal-leakage guarantee of the
package and is verified by ``tests/test_features.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oulad_risk.config import (
    AT_RISK_OUTCOMES,
    CATEGORICAL_FEATURES,
    ENROLMENT_KEYS,
    FORBIDDEN_FEATURES,
)

logger = logging.getLogger(__name__)

ASSESSMENT_FEATURES = [
    "n_assess_due",
    "n_assess_submitted",
    "submission_rate",
    "mean_score",
    "min_score",
    "weighted_score",
    "n_late_submissions",
]
VLE_FEATURES = [
    "clicks_total",
    "clicks_pre_start",
    "clicks_last_7d",
    "active_days",
    "distinct_sites",
    "clicks_per_active_day",
]
REGISTRATION_FEATURES = ["date_registration"]
BASE_NUMERIC_FEATURES = ["num_of_prev_attempts", "studied_credits"]


@dataclass
class FeatureSet:
    """Model-ready data: features ``X``, binary target ``y`` and grouping ids ``groups``."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    categorical: list[str]
    numeric: list[str]

    def __len__(self) -> int:
        return len(self.y)


def make_target(final_result: pd.Series) -> pd.Series:
    """Binary target: 1 if the outcome is Fail or Withdrawn, else 0."""
    unknown = set(final_result.dropna().unique()) - AT_RISK_OUTCOMES - {"Pass", "Distinction"}
    if unknown:
        raise ValueError(f"Unexpected final_result values: {sorted(unknown)}")
    return final_result.isin(AT_RISK_OUTCOMES).astype("int8").rename("at_risk")


def _normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("code_module", "code_presentation"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def assessment_features(
    assessments: pd.DataFrame, student_assessment: pd.DataFrame, cutoff_day: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assessment features using only assessments due AND submitted by the cut-off.

    Returns per-enrolment aggregates and, separately, the number of assessments due by the
    cut-off per module presentation (needed to compute the submission rate).
    """
    due = assessments.loc[assessments["date"].le(cutoff_day)]  # NaN exam dates drop out here
    due = _normalise_keys(due)
    n_due = (
        due.groupby(["code_module", "code_presentation"])
        .size()
        .rename("n_assess_due")
        .reset_index()
    )

    sub = student_assessment.merge(due, on="id_assessment", how="inner")
    sub = sub.loc[sub["date_submitted"].le(cutoff_day)].copy()
    sub["is_late"] = (sub["date_submitted"] > sub["date"]).astype(int)
    sub["weighted"] = sub["score"] * sub["weight"] / 100.0

    agg = (
        sub.groupby(ENROLMENT_KEYS)
        .agg(
            n_assess_submitted=("id_assessment", "nunique"),
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            weighted_score=("weighted", "sum"),
            n_late_submissions=("is_late", "sum"),
        )
        .reset_index()
    )
    return agg, n_due


def vle_features(student_vle: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Per-enrolment VLE engagement features from clicks on or before the cut-off day."""
    vle = student_vle.loc[student_vle["date"].le(cutoff_day)]
    grouped = vle.groupby(ENROLMENT_KEYS, observed=True)
    out = grouped.agg(
        clicks_total=("sum_click", "sum"),
        distinct_sites=("id_site", "nunique"),
    )
    out["clicks_pre_start"] = (
        vle.loc[vle["date"] < 0].groupby(ENROLMENT_KEYS, observed=True)["sum_click"].sum()
    )
    out["clicks_last_7d"] = (
        vle.loc[vle["date"] > cutoff_day - 7]
        .groupby(ENROLMENT_KEYS, observed=True)["sum_click"]
        .sum()
    )
    out["active_days"] = (
        vle.loc[vle["date"] >= 0].groupby(ENROLMENT_KEYS, observed=True)["date"].nunique()
    )
    out = out.fillna(0)
    out["clicks_per_active_day"] = out["clicks_total"] / out["active_days"].replace(0, np.nan)
    return _normalise_keys(out.reset_index())


def build_features(
    tables: dict[str, pd.DataFrame],
    cutoff_day: int = 30,
    exclude_withdrawn_before_cutoff: bool = True,
) -> FeatureSet:
    """Join all available tables into one leakage-safe feature matrix.

    Parameters
    ----------
    tables:
        Output of :func:`oulad_risk.data.load_tables`. Only ``studentInfo`` is required.
    cutoff_day:
        Prediction moment in days after module start; no later information is used.
    exclude_withdrawn_before_cutoff:
        Drop enrolments that had already unregistered by the cut-off. They need no
        prediction and keeping them would let the model "predict" an observed event.
    """
    info = _normalise_keys(tables["studentInfo"])
    df = info.copy()
    numeric = list(BASE_NUMERIC_FEATURES)

    if "studentRegistration" in tables:
        reg = _normalise_keys(tables["studentRegistration"])
        df = df.merge(reg, on=ENROLMENT_KEYS, how="left")
        numeric += REGISTRATION_FEATURES
        if exclude_withdrawn_before_cutoff:
            gone = df["date_unregistration"].le(cutoff_day)
            logger.info("Excluding %d enrolments unregistered by day %d", gone.sum(), cutoff_day)
            df = df.loc[~gone]

    if "assessments" in tables and "studentAssessment" in tables:
        agg, n_due = assessment_features(
            tables["assessments"], tables["studentAssessment"], cutoff_day
        )
        df = df.merge(agg, on=ENROLMENT_KEYS, how="left")
        df = df.merge(n_due, on=["code_module", "code_presentation"], how="left")
        for col in ("n_assess_due", "n_assess_submitted", "n_late_submissions", "weighted_score"):
            df[col] = df[col].fillna(0)
        df["submission_rate"] = df["n_assess_submitted"] / df["n_assess_due"].replace(0, np.nan)
        numeric += ASSESSMENT_FEATURES

    if "studentVle" in tables:
        vle = vle_features(tables["studentVle"], cutoff_day)
        df = df.merge(vle, on=ENROLMENT_KEYS, how="left")
        count_cols = [c for c in VLE_FEATURES if c != "clicks_per_active_day"]
        df[count_cols] = df[count_cols].fillna(0)
        numeric += VLE_FEATURES

    df["imd_band"] = df["imd_band"].fillna("Unknown")
    categorical = list(CATEGORICAL_FEATURES)
    X = df[categorical + numeric].reset_index(drop=True)
    X[categorical] = X[categorical].astype(str)

    leaked = FORBIDDEN_FEATURES.intersection(X.columns)
    if leaked:  # defensive: should be impossible by construction
        raise RuntimeError(f"Forbidden columns reached the feature matrix: {sorted(leaked)}")

    y = make_target(df["final_result"]).reset_index(drop=True)
    groups = df["id_student"].reset_index(drop=True).rename("id_student")
    logger.info("Feature matrix: %d rows x %d columns, at-risk rate %.3f", *X.shape, y.mean())
    return FeatureSet(X=X, y=y, groups=groups, categorical=categorical, numeric=numeric)
