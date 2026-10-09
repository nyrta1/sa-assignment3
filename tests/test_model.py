import numpy as np
import pytest

from oulad_risk.config import ExperimentConfig
from oulad_risk.features import build_features
from oulad_risk.model import build_pipeline, evaluate, feature_importance, tune_threshold


def test_evaluate_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == m["f1"] == m["roc_auc"] == 1.0


def test_tune_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    t = tune_threshold(y, np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8]))
    assert 0.3 < t <= 0.6


@pytest.mark.parametrize("name", ["rf", "logreg"])
def test_pipeline_learns_signal(tables, name):
    fs = build_features(tables)
    model = build_pipeline(fs.categorical, fs.numeric, name, random_state=0)
    model.fit(fs.X, fs.y)
    proba = model.predict_proba(fs.X)[:, 1]
    assert evaluate(fs.y.to_numpy(), proba)["roc_auc"] > 0.7
    imp = feature_importance(model, top=5)
    assert len(imp) == 5 and (imp >= 0).all()


@pytest.mark.parametrize(
    "kwargs", [{"cutoff_day": -1}, {"model": "svm"}, {"test_size": 1.5}, {"val_size": 0}]
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)
