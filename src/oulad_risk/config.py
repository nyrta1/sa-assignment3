"""Central configuration: constants, table schemas and the experiment settings object."""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Outcomes treated as "at risk" (positive class).
AT_RISK_OUTCOMES: frozenset[str] = frozenset({"Fail", "Withdrawn"})

#: Keys that identify one enrolment (one student taking one module presentation).
ENROLMENT_KEYS: list[str] = ["code_module", "code_presentation", "id_student"]

#: Required columns per OULAD table.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "studentInfo": [
        "code_module", "code_presentation", "id_student", "gender", "region",
        "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
        "studied_credits", "disability", "final_result",
    ],
    "studentRegistration": [
        "code_module", "code_presentation", "id_student",
        "date_registration", "date_unregistration",
    ],
    "assessments": [
        "code_module", "code_presentation", "id_assessment",
        "assessment_type", "date", "weight",
    ],
    "studentAssessment": ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
    "studentVle": [
        "code_module", "code_presentation", "id_student", "id_site", "date", "sum_click",
    ],
}  # fmt: skip

#: Tables without which the pipeline cannot run at all; every other table is optional.
MANDATORY_TABLES: tuple[str, ...] = ("studentInfo",)

CATEGORICAL_FEATURES: list[str] = [
    "code_module", "gender", "region", "highest_education", "imd_band", "age_band", "disability",
]  # fmt: skip

#: Columns that must NEVER reach the model (target or information from after the outcome).
FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {"final_result", "date_unregistration", "at_risk", "id_student", "code_presentation"}
)

MODEL_CHOICES: tuple[str, ...] = ("rf", "logreg", "xgb")


@dataclass(frozen=True)
class ExperimentConfig:
    """All settings that influence a run. Saved next to the results for reproducibility."""

    cutoff_day: int = 30
    model: str = "rf"
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    exclude_withdrawn_before_cutoff: bool = True

    def __post_init__(self) -> None:
        if self.cutoff_day < 0:
            raise ValueError("cutoff_day must be >= 0")
        if self.model not in MODEL_CHOICES:
            raise ValueError(f"model must be one of {MODEL_CHOICES}, got {self.model!r}")
        for name in ("test_size", "val_size"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")

    def to_dict(self) -> dict:
        return asdict(self)
