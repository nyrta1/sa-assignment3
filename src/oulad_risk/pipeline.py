"""End-to-end experiment: features -> group split -> audit -> train -> tune -> evaluate."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from oulad_risk import __version__
from oulad_risk.config import ExperimentConfig
from oulad_risk.features import FeatureSet, build_features
from oulad_risk.model import build_pipeline, evaluate, feature_importance, tune_threshold
from oulad_risk.split import audit_overlap, group_split, naive_stratified_split
from oulad_risk.viz import plot_confusion, plot_importance, plot_roc

logger = logging.getLogger(__name__)


class LeakageError(RuntimeError):
    """Raised when a split that must be student-disjoint is not."""


def run_experiment(
    tables: dict[str, pd.DataFrame], config: ExperimentConfig, out_dir: str | Path | None = None
) -> dict:
    """Run the full experiment and (optionally) write metrics and figures to *out_dir*."""
    fs: FeatureSet = build_features(
        tables, config.cutoff_day, config.exclude_withdrawn_before_cutoff
    )
    rs = config.random_state

    # 1. Outer split by student, then audit it. A non-empty overlap is a hard failure.
    train_idx, test_idx = group_split(fs.groups, config.test_size, rs)
    audit = audit_overlap(fs.groups, train_idx, test_idx)
    if not audit.is_clean:
        raise LeakageError(f"{audit.n_shared_students} students occur in train and test")

    # 2. For comparison only: how much a naive row-level split would have leaked.
    naive = audit_overlap(fs.groups, *naive_stratified_split(fs.y, config.test_size, rs))

    # 3. Inner group split of the training set -> threshold tuned on validation, not test.
    tr_groups = fs.groups.iloc[train_idx].reset_index(drop=True)
    inner_tr, inner_val = group_split(tr_groups, config.val_size, rs)
    X_tr, y_tr = fs.X.iloc[train_idx], fs.y.iloc[train_idx]
    model = build_pipeline(fs.categorical, fs.numeric, config.model, rs)
    model.fit(X_tr.iloc[inner_tr], y_tr.iloc[inner_tr])
    val_proba = model.predict_proba(X_tr.iloc[inner_val])[:, 1]
    threshold = tune_threshold(y_tr.iloc[inner_val].to_numpy(), val_proba)

    # 4. Refit on the whole training set and evaluate once on the untouched test set.
    model.fit(X_tr, y_tr)
    y_te = fs.y.iloc[test_idx].to_numpy()
    proba = model.predict_proba(fs.X.iloc[test_idx])[:, 1]
    train_proba = model.predict_proba(X_tr)[:, 1]

    results = {
        "meta": {
            "package_version": __version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "tables_used": sorted(tables),
        },
        "config": config.to_dict(),
        "data": {
            "n_enrolments": len(fs),
            "n_students": int(fs.groups.nunique()),
            "n_features": fs.X.shape[1],
            "at_risk_rate": round(float(fs.y.mean()), 4),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        },
        "leakage_audit": {
            "group_split": {**asdict(audit), "leaked_row_share": audit.leaked_row_share},
            "naive_split": {**asdict(naive), "leaked_row_share": round(naive.leaked_row_share, 4)},
        },
        "metrics": {
            "test_default_threshold": evaluate(y_te, proba, 0.5),
            "test_tuned_threshold": evaluate(y_te, proba, threshold),
            "train_tuned_threshold": evaluate(y_tr.to_numpy(), train_proba, threshold),
        },
        "top_features": {k: round(float(v), 5) for k, v in feature_importance(model).items()},
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(results, indent=2))
        plot_roc(y_te, proba, out / "roc_curve.png", config.model.upper())
        plot_importance(feature_importance(model), out / "feature_importance.png")
        plot_confusion(y_te, (proba >= threshold).astype(int), out / "confusion_matrix.png")
        logger.info("Results written to %s", out.resolve())
    return results
