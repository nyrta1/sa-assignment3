"""End-to-end tests of the experiment runner and the CLI."""

import json

import pytest

from oulad_risk.cli import main
from oulad_risk.config import ExperimentConfig
from oulad_risk.pipeline import run_experiment


def test_run_experiment_writes_artifacts(tables, tmp_path):
    res = run_experiment(tables, ExperimentConfig(model="logreg"), tmp_path)
    for f in ["metrics.json", "roc_curve.png", "feature_importance.png", "confusion_matrix.png"]:
        assert (tmp_path / f).stat().st_size > 0
    assert res["leakage_audit"]["group_split"]["n_shared_students"] == 0
    assert res["metrics"]["test_tuned_threshold"]["roc_auc"] > 0.6
    assert json.loads((tmp_path / "metrics.json").read_text())["config"]["model"] == "logreg"


def test_results_are_reproducible(tables):
    cfg = ExperimentConfig(model="rf", random_state=3)
    a = run_experiment(tables, cfg)["metrics"]
    b = run_experiment(tables, cfg)["metrics"]
    assert a == b


def test_cli_synth_then_run(tmp_path, capsys):
    data = tmp_path / "data"
    assert main(["synth", "--out-dir", str(data), "--students", "300"]) == 0
    assert main(["run", "--data-dir", str(data), "--out-dir", str(tmp_path / "rep"),
                 "--model", "logreg", "--min-auc", "0.6"]) == 0  # fmt: skip
    assert '"roc_auc"' in capsys.readouterr().out


def test_cli_quality_gate_fails(data_dir, tmp_path):
    assert main(["run", "--data-dir", str(data_dir), "--out-dir", str(tmp_path),
                 "--model", "logreg", "--min-auc", "0.999"]) == 1  # fmt: skip


def test_cli_audit(data_dir, capsys):
    assert main(["audit", "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "group (student) split" in out and "shared students=     0" in out


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        main([])
