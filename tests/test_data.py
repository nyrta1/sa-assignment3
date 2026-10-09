import pandas as pd
import pytest

from oulad_risk.data import SchemaError, find_table, load_tables, validate_columns


def test_load_tables_reads_all_five_tables(data_dir):
    tables = load_tables(data_dir)
    assert set(tables) == {
        "studentInfo",
        "studentRegistration",
        "assessments",
        "studentAssessment",
        "studentVle",
    }
    assert tables["studentVle"]["date"].dtype == "int16"


def test_find_table_is_case_insensitive(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "STUDENTINFO.csv").write_text("a\n1\n")
    assert find_table(tmp_path, "studentInfo").name == "STUDENTINFO.csv"
    assert find_table(tmp_path, "studentVle") is None


def test_missing_mandatory_table_raises(tmp_path):
    with pytest.raises(SchemaError, match="Mandatory"):
        load_tables(tmp_path)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tables(tmp_path / "nope")


def test_optional_tables_may_be_absent(data_dir, tmp_path):
    (tmp_path / "studentInfo.csv").write_text((data_dir / "studentInfo.csv").read_text())
    assert set(load_tables(tmp_path)) == {"studentInfo"}


def test_validate_columns_reports_missing():
    with pytest.raises(SchemaError, match="final_result"):
        validate_columns(pd.DataFrame(columns=["id_student"]), "studentInfo")
