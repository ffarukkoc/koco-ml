from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _samples import BinaryTable, RegressionTable
from koco._helpers import (
    _Alert, _CheckArtifact, _CheckColumns, _CheckUniqueColumns, _Done, _Info, _OutputPath, _Rate, _Step,
    _ValidatePlan, _ValidateTargetAndFeatures,
)

def test_check_columns_lists_every_missing_column():
    Table = pd.DataFrame({"a": [1]})
    _CheckColumns(Table, ["a"])
    with pytest.raises(ValueError, match="'x', 'y'"):
        _CheckColumns(Table, ["a", "x", "y", "x"])

def test_check_unique_columns():
    Table = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicated"):
        _CheckUniqueColumns(Table)
    _CheckUniqueColumns(pd.DataFrame({"a": [1], "b": [2]}))

def test_rate_sums_to_one_hundred():
    Rate = _Rate(pd.Series([1, 1, 2]))
    assert Rate.tolist() == [25.0, 25.0, 50.0]

def test_validate_plan_checks_structure():
    _ValidatePlan({"Columns": {}})
    _ValidatePlan({"PlanType": "Ignored", "Columns": {"a": {}}})
    for Plan in ({"PlanType": "Null"}, ["not", "a", "dict"], {"Columns": [1]}):
        with pytest.raises(ValueError, match="Columns"):
            _ValidatePlan(Plan)

def test_check_artifact():
    Valid = {"Model": object(), "ModelType": "Regression", "Features": ["a"]}
    _CheckArtifact(Valid)
    with pytest.raises(ValueError, match="old 'Task' key"):
        _CheckArtifact({"Model": 1, "Task": "Regression", "Features": ["a"]})
    with pytest.raises(ValueError, match="ModelType"):
        _CheckArtifact({"Model": 1, "Features": ["a"]})
    with pytest.raises(ValueError, match="'Regression' or 'Binary'"):
        _CheckArtifact({**Valid, "ModelType": "Other"})
    with pytest.raises(ValueError):
        _CheckArtifact("not a dict")

def test_output_path_defaults_and_folders(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Stamp = date.today().strftime("%y%m%d")
    assert _OutputPath(None, "Fn", ".txt") == Path(f"{Stamp}_Fn.txt")
    assert _OutputPath("Sub/Folder", "Fn", ".txt") == Path("Sub/Folder") / f"{Stamp}_Fn.txt"
    assert (tmp_path / "Sub" / "Folder").is_dir()
    assert _OutputPath(tmp_path, "Fn", ".txt") == tmp_path / f"{Stamp}_Fn.txt"
    assert _OutputPath("Other/Named.xlsx", "Fn", ".txt") == Path("Other/Named.xlsx")
    assert (tmp_path / "Other").is_dir()
    assert capsys.readouterr().out == ""
    with pytest.raises(TypeError):
        _OutputPath(5, "Fn", ".txt")

def test_output_path_reports_existing_files(tmp_path, capsys):
    Existing = tmp_path / "a.txt"
    Existing.write_text("x")
    assert _OutputPath(Existing, "Fn", ".txt") == Existing
    assert "same name" in capsys.readouterr().out
    _OutputPath(Existing, "Fn", ".txt", Notify=False)
    assert capsys.readouterr().out == ""

def test_message_symbols(capsys):
    _Alert("a"); _Info("b"); _Done("c"); _Step("d")
    assert capsys.readouterr().out.splitlines() == ["❗ a", "ℹ b", "✓ c", "• d"]

def test_validate_target_and_features_defaults_to_all_other_columns():
    Table = RegressionTable()
    assert _ValidateTargetAndFeatures(Table, "y", None, "Regression") == ["a", "b", "c"]

def test_validate_target_and_features_rejects_bad_inputs():
    Table = RegressionTable()
    with pytest.raises(ValueError, match="Target column 'y' must not"):
        _ValidateTargetAndFeatures(Table, "y", ["a", "y"], "Regression")
    with pytest.raises(ValueError, match="Duplicate"):
        _ValidateTargetAndFeatures(Table, "y", ["a", "a"], "Regression")
    with pytest.raises(ValueError, match="Binary Target"):
        _ValidateTargetAndFeatures(Table, "y", ["a"], "Binary")
    with pytest.raises(ValueError, match="not found"):
        _ValidateTargetAndFeatures(Table, "zz", ["a"], "Regression")
    Table.loc[0, "a"] = np.inf
    with pytest.raises(ValueError, match="inf"):
        _ValidateTargetAndFeatures(Table, "y", ["a"], "Regression")
    _ValidateTargetAndFeatures(BinaryTable(), "y", ["a"], "Binary")
