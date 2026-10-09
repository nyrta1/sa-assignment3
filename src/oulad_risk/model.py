"""Model construction, decision-threshold tuning and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _estimator(name: str, random_state: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "XGBoost is optional. Install it with: pip install 'oulad-risk[xgboost]'"
            ) from exc
        return XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name!r}")


def build_pipeline(
    categorical: list[str], numeric: list[str], model: str = "rf", random_state: int = 42
) -> Pipeline:
    """Preprocessing + classifier in one scikit-learn ``Pipeline``.

    Keeping preprocessing inside the pipeline means imputation medians and scaling
    statistics are learned from the training fold only (no preprocessing leakage).
    """
    numeric_steps: list = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if model == "logreg":
        numeric_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
            ("num", Pipeline(numeric_steps), numeric),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("clf", _estimator(model, random_state))])


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Return the probability threshold that maximises F1 on a *validation* set."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]) if len(thresholds) else 0.5


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Standard binary-classification metrics at a given decision threshold."""
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def feature_importance(pipeline: Pipeline, top: int = 15) -> pd.Series:
    """Model-native importances (trees) or absolute coefficients (logistic regression)."""
    clf = pipeline.named_steps["clf"]
    names = pipeline.named_steps["pre"].get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    else:
        values = np.abs(clf.coef_).ravel()
    return pd.Series(values, index=names).sort_values(ascending=False).head(top)
