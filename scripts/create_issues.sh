#!/usr/bin/env bash
# Creates the project backlog as GitHub issues. Requires the GitHub CLI: https://cli.github.com
# Usage: gh auth login   (once)   then   bash scripts/create_issues.sh
set -euo pipefail
gh label create leakage --color B60205 --description "Data-leakage guarantees" 2>/dev/null || true
gh label create ci --color 0E8A16 --description "CI/CD pipeline" 2>/dev/null || true

gh issue create --label enhancement --label leakage \
  --title "Temporal leakage: restrict all behavioural features to the day-30 cut-off" \
  --body "Assessment and VLE features must only use records dated <= cutoff_day. Add a test that mutating post-cutoff records leaves X unchanged."
gh issue create --label enhancement --label leakage \
  --title "Group leakage: student-disjoint split with automatic overlap audit" \
  --body "Use GroupShuffleSplit by id_student and fail the run if any student is shared between train and test."
gh issue create --label ci \
  --title "Set up GitHub Actions CI (lint, test matrix, smoke run, build)" \
  --body "Ruff, pytest on 3.10-3.12 + Windows/macOS, coverage >= 90 %, synthetic smoke run with ROC-AUC quality gate."
gh issue create --label ci \
  --title "Release workflow: publish GitHub Release on version tag" \
  --body "On tag v*.*.*: verify version, test, build wheel/sdist, create release."
gh issue create --label enhancement \
  --title "Add SHAP instance-level explanations" \
  --body "Per-student SHAP values for RF and XGBoost, as in the thesis. Keep shap as an optional extra."
gh issue create --label enhancement \
  --title "Add GroupKFold cross-validation with confidence intervals" \
  --body "Report mean +- std of ROC-AUC/F1 over 5 student-disjoint folds instead of a single split."
gh issue create --label enhancement \
  --title "Probability calibration (NFR7)" \
  --body "Add CalibratedClassifierCV and a reliability diagram so probabilities can be read as risk levels."
echo "Done. Close the first four issues after pushing (they are implemented in v0.1.0)."
