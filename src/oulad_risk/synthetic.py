"""Small synthetic OULAD-shaped dataset.

The real OULAD licence (CC-BY 4.0) allows redistribution, but the raw files are ~450 MB,
so the CI pipeline and the unit tests use this generator instead. The tables have exactly
the real schema and contain a learnable signal (engagement and scores drive the outcome),
which lets the tests check that the model learns *something* without downloading data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODULES = [("AAA", "2013J"), ("BBB", "2013J"), ("BBB", "2014B"), ("CCC", "2014J")]
REGIONS = ["Scotland", "London Region", "Wales", "North Region", "South East Region"]
EDUCATION = ["A Level or Equivalent", "Lower Than A Level", "HE Qualification"]
IMD = ["0-10%", "10-20", "20-30%", "50-60%", "90-100%", None]
AGES = ["0-35", "35-55", "55<="]


def generate(n_students: int = 600, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Return a dict of synthetic tables keyed like :func:`oulad_risk.data.load_tables`."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(10_000, 99_999), size=n_students, replace=False)

    # ~15% of students take two modules -> exercises the group-leakage logic.
    rows = []
    for sid in ids:
        n_mod = 2 if rng.random() < 0.15 else 1
        for idx in rng.choice(len(MODULES), size=n_mod, replace=False):
            rows.append((*MODULES[idx], int(sid)))
    info = pd.DataFrame(rows, columns=["code_module", "code_presentation", "id_student"])
    n = len(info)

    engagement = rng.gamma(2.0, 1.0, n)  # latent "motivation" driving all behaviour
    info["gender"] = rng.choice(["M", "F"], n)
    info["region"] = rng.choice(REGIONS, n)
    info["highest_education"] = rng.choice(EDUCATION, n)
    info["imd_band"] = rng.choice(np.array(IMD, dtype=object), n)
    info["age_band"] = rng.choice(AGES, n, p=[0.7, 0.28, 0.02])
    info["num_of_prev_attempts"] = rng.poisson(0.2, n)
    info["studied_credits"] = rng.choice([30, 60, 90, 120, 240], n, p=[0.1, 0.6, 0.1, 0.15, 0.05])
    info["disability"] = rng.choice(["N", "Y"], n, p=[0.9, 0.1])

    risk_logit = (
        1.6 - 1.5 * engagement + 0.5 * info["num_of_prev_attempts"]
        + 0.004 * (info["studied_credits"] - 60) + rng.normal(0, 0.4, n)
    )  # fmt: skip
    at_risk = rng.random(n) < 1 / (1 + np.exp(-risk_logit))
    withdrawn = at_risk & (rng.random(n) < 0.5)
    info["final_result"] = np.where(
        withdrawn, "Withdrawn",
        np.where(at_risk, "Fail", np.where(rng.random(n) < 0.2, "Distinction", "Pass")),
    )  # fmt: skip

    reg = info[["code_module", "code_presentation", "id_student"]].copy()
    reg["date_registration"] = -rng.integers(1, 200, n)
    unreg = np.where(withdrawn, rng.integers(-20, 200, n), np.nan)
    reg["date_unregistration"] = unreg

    assessments = []
    aid = 1000
    for mod, pres in MODULES:
        for date, weight in [(19, 10), (54, 20), (117, 20), (None, 100)]:
            atype = "Exam" if date is None else "TMA"
            assessments.append((mod, pres, aid, atype, date, weight))
            aid += 1
    assess = pd.DataFrame(
        assessments,
        columns=["code_module", "code_presentation", "id_assessment", "assessment_type",
                 "date", "weight"],
    )  # fmt: skip

    sa_rows, vle_rows = [], []
    for i, r in info.iterrows():
        mine = assess[(assess.code_module == r.code_module)
                      & (assess.code_presentation == r.code_presentation)]  # fmt: skip
        for a in mine.itertuples():
            if pd.isna(a.date) or rng.random() > min(0.95, 0.35 + 0.25 * engagement[i]):
                continue
            submitted = int(a.date + rng.integers(-5, 6))
            score = float(np.clip(rng.normal(45 + 12 * engagement[i], 12), 0, 100))
            sa_rows.append((a.id_assessment, r.id_student, submitted, 0, round(score)))
        n_events = rng.poisson(8 + 25 * engagement[i])
        days = rng.integers(-15, 240, n_events)
        for d in days:
            vle_rows.append((r.code_module, r.code_presentation, r.id_student,
                             int(rng.integers(500_000, 500_060)), int(d),
                             int(rng.integers(1, 10))))  # fmt: skip

    student_assessment = pd.DataFrame(
        sa_rows, columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]
    )
    student_vle = pd.DataFrame(
        vle_rows,
        columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
    )
    return {
        "studentInfo": info,
        "studentRegistration": reg,
        "assessments": assess,
        "studentAssessment": student_assessment,
        "studentVle": student_vle,
    }


def write(out_dir: str | Path, n_students: int = 600, seed: int = 0) -> Path:
    """Write the synthetic tables as CSV files with the original OULAD file names."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in generate(n_students, seed).items():
        df.to_csv(out / f"{name}.csv", index=False)
    return out
