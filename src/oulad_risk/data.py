"""Loading and schema validation of the raw OULAD CSV tables."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from oulad_risk.config import MANDATORY_TABLES, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

#: Narrow dtypes keep the ~10.6M-row studentVle table small in memory.
_DTYPES: dict[str, dict[str, str]] = {
    "studentVle": {
        "code_module": "category",
        "code_presentation": "category",
        "id_student": "int32",
        "id_site": "int32",
        "date": "int16",
        "sum_click": "int32",
    },
}


class SchemaError(ValueError):
    """Raised when a mandatory table is missing or a table lacks required columns."""


def find_table(data_dir: str | Path, name: str) -> Path | None:
    """Return the path of ``<name>.csv`` or ``<name>.csv.gz`` inside *data_dir*.

    The search is recursive and case-insensitive, so both the original OULAD archive
    layout and the Kaggle mirror layout are accepted without renaming files.
    """
    wanted = {f"{name.lower()}.csv", f"{name.lower()}.csv.gz"}
    for path in sorted(Path(data_dir).rglob("*")):
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


def validate_columns(df: pd.DataFrame, table: str) -> None:
    """Raise :class:`SchemaError` if *df* lacks any required column of *table*."""
    missing = [c for c in REQUIRED_COLUMNS[table] if c not in df.columns]
    if missing:
        raise SchemaError(f"Table '{table}' is missing required columns: {missing}")


def load_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every known OULAD table found in *data_dir*.

    Missing optional tables are skipped with a warning (their features are simply not
    built); a missing mandatory table raises :class:`SchemaError`.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    tables: dict[str, pd.DataFrame] = {}
    for name, columns in REQUIRED_COLUMNS.items():
        path = find_table(data_dir, name)
        if path is None:
            if name in MANDATORY_TABLES:
                raise SchemaError(f"Mandatory table '{name}.csv' not found in {data_dir}")
            logger.warning("Optional table '%s' not found - its features are skipped", name)
            continue
        header = pd.read_csv(path, nrows=0).columns
        validate_columns(pd.DataFrame(columns=header), name)
        df = pd.read_csv(
            path,
            usecols=columns,
            dtype=_DTYPES.get(name),
            na_values=["", "?"],
        )
        logger.info("Loaded %-19s %10d rows from %s", name, len(df), path.name)
        tables[name] = df
    return tables
