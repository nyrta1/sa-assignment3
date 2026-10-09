"""Publication-style figures saved as PNG files (headless; safe to run in CI)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed - required for GitHub Actions runners
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import RocCurveDisplay, confusion_matrix  # noqa: E402

_STYLE = {"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False}


def plot_roc(y_true: np.ndarray, proba: np.ndarray, path: Path, label: str) -> Path:
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        RocCurveDisplay.from_predictions(y_true, proba, name=label, ax=ax)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="Chance")
        ax.set_title("ROC curve - held-out students")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_importance(importance: pd.Series, path: Path) -> Path:
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        importance.iloc[::-1].plot.barh(ax=ax, color="#2b6cb0")
        ax.set_xlabel("Importance")
        ax.set_title("Top features")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_confusion(y_true: np.ndarray, pred: np.ndarray, path: Path) -> Path:
    cm = confusion_matrix(y_true, pred)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")  # fmt: skip
        ax.set_xticks([0, 1], ["Not at risk", "At risk"])
        ax.set_yticks([0, 1], ["Not at risk", "At risk"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix (tuned threshold)")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path
