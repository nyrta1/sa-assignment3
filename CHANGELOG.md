# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/),
versioning: [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-09
### Added
- Schema-validated loading of the five OULAD tables (`data.py`).
- Leakage-safe day-30 feature engineering: demographic, registration, assessment and VLE features.
- Student-disjoint `GroupShuffleSplit` with an automatic overlap audit that fails the run on leakage.
- Random Forest, Logistic Regression and optional XGBoost models in one scikit-learn `Pipeline`.
- Decision-threshold tuning on an inner validation split; ROC, importance and confusion figures.
- `oulad-risk` CLI (`run`, `audit`, `synth`) with an ROC-AUC quality gate.
- Synthetic OULAD-shaped data generator for tests and CI.
- 32 pytest tests (~98 % coverage); GitHub Actions CI (lint, test matrix, smoke, build) and release workflow.
- Reference results on the full OULAD in `results/`.
