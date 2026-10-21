import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import koco
from _samples import WOETable

def NullTable() -> pd.DataFrame:
    return pd.DataFrame({
        "Income": [1.0, 2.0, None, 4.0],
        "Age": [20.0, None, 40.0, 50.0],
        "Seg": ["a", "a", None, "b"],
        "Flag": [1.0, 2.0, 3.0, 4.0],
    })

def test_fit_null_plan_values_and_json_safety(tmp_path):
    Plan, Check = koco.FitNullPlan(NullTable(), [("Income", "Median"), ("Age", "Mean"), ("Seg", "Mode"), ("Flag", "FixValue", np.int64(7))])
    assert Plan["Columns"]["Income"]["Value"] == 2.0
    assert Plan["Columns"]["Seg"]["Value"] == "a"
    assert Plan["Columns"]["Flag"]["Value"] == 7 and type(Plan["Columns"]["Flag"]["Value"]) is int
    koco.ExportJson(Plan, tmp_path / "plan.json")
    assert koco.ImportJson(tmp_path / "plan.json") == Plan

def test_fit_null_plan_validates_before_printing_anything(capsys):
    Table = NullTable()
    Cases = [
        ([("Income", "Median"), ("Income", "Mean")], "more than one rule"),
        ([("Income", "Nonsense")], "Unknown Strategy"),
        ([("Income", "FixValue")], "needs a value"),
        ([("Seg", "Median")], "numeric"),
        ([("Income",)], "Each rule"),
    ]
    for Plan, Message in Cases:
        with pytest.raises(ValueError, match=Message):
            koco.FitNullPlan(Table, Plan)
    with pytest.raises(ValueError, match="only null"):
        koco.FitNullPlan(pd.DataFrame({"a": [np.nan, np.nan]}), [("a", "Mode")])
    assert capsys.readouterr().out == ""

def test_apply_null_plan_fills_drops_and_keeps_index(capsys):
    Table = NullTable()
    Plan, _ = koco.FitNullPlan(Table, [("Income", "DropRows"), ("Age", "Mean"), ("Seg", "Drop"), ("Flag", "KeepNaN")])
    Result = koco.ApplyNullPlan(Table, Plan)
    assert Result.index.tolist() == [0, 1, 3]
    assert "Seg" not in Result.columns and Result["Age"].isna().sum() == 0
    assert "Dropped row ids: [2]" in capsys.readouterr().out
    assert "Flag" not in koco.ApplyNullPlan(Table, Plan, NaNHandling="Safe", Verbose=False).columns
    assert capsys.readouterr().out == ""
    assert Table["Age"].isna().sum() == 1 and len(Table) == 4

def test_apply_null_plan_does_not_touch_the_source_table():
    Table = NullTable()
    Before = Table.copy()
    Plan, _ = koco.FitNullPlan(Table, [("Income", "Median"), ("Age", "Mean")])
    koco.ApplyNullPlan(Table, Plan, Verbose=False)
    pd.testing.assert_frame_equal(Table, Before)

def test_apply_null_plan_rejects_bad_inputs():
    Table = NullTable()
    Plan, _ = koco.FitNullPlan(Table, [("Income", "Median")])
    with pytest.raises(ValueError, match="NaNHandling"):
        koco.ApplyNullPlan(Table, Plan, NaNHandling="Other")
    with pytest.raises(ValueError, match="valid null Strategy"):
        koco.ApplyNullPlan(Table, {"Columns": {"Income": {"Strategy": "Robust", "Offset": 0, "Scale": 1}}})
    with pytest.raises(ValueError, match="valid null Strategy"):
        koco.ApplyNullPlan(Table, {"Columns": {"Income": {"Value": 1}}})
    with pytest.raises(ValueError, match="Columns"):
        koco.ApplyNullPlan(Table, {"PlanType": "Null"})

def test_apply_outlier_plan_caps_and_reports(capsys):
    Table = pd.DataFrame({"Income": [1.0, 500.0], "Age": [10.0, 50.0]})
    Plan = {"Columns": {"Income": ["U", 100], "Age": ["L", 18], "Gone": ["U", 1]}}
    Result = koco.ApplyOutlierPlan(Table, Plan)
    assert Result["Income"].tolist() == [1.0, 100.0] and Result["Age"].tolist() == [18.0, 50.0]
    assert "Gone exists in the outlier plan" in capsys.readouterr().out
    assert Table["Income"].tolist() == [1.0, 500.0]

    BadMethod = {"Columns": {"Income": ["X", 100]}}
    assert koco.ApplyOutlierPlan(Table, BadMethod)["Income"].tolist() == [1.0, 500.0]
    assert "Method must be 'U' or 'L'" in capsys.readouterr().out

def test_apply_outlier_plan_validation():
    Plan = {"Columns": {"a": ["U", 1]}}
    with pytest.raises(ValueError, match="duplicated"):
        koco.ApplyOutlierPlan(pd.DataFrame([[1, 2]], columns=["a", "a"]), Plan)
    with pytest.raises(ValueError, match="Method, Value"):
        koco.ApplyOutlierPlan(pd.DataFrame({"a": [1]}), {"Columns": {"a": ["U"]}})

def test_fit_scaling_plan_round_trip_matches_sklearn():
    Table = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 10.0, 20.0, 40.0], "c": [-4.0, 2.0, 0.0, 4.0]})
    Plan = koco.FitScalingPlan(Table, {"Standardization": ["a"], "Normalization": ["b"], "MaxAbs": ["c"]})
    Result = koco.ApplyScalingPlan(Table, Plan)
    assert Result["a"].mean() == pytest.approx(0) and Result["a"].std(ddof=0) == pytest.approx(1)
    assert Result["b"].min() == 0 and Result["b"].max() == 1
    assert Result["c"].abs().max() == 1
    assert Table["a"].tolist() == [1.0, 2.0, 3.0, 4.0]

def test_fit_scaling_plan_validation():
    Table = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Unknown scaling"):
        koco.FitScalingPlan(Table, {"Other": ["a"]})
    with pytest.raises(ValueError, match="multiple"):
        koco.FitScalingPlan(Table, {"Robust": ["a"], "MaxAbs": ["a"]})
    with pytest.raises(ValueError, match="not found"):
        koco.FitScalingPlan(Table, {"Robust": ["zz"]})
    with pytest.raises(TypeError):
        koco.FitScalingPlan(Table, {"Robust": "a"})
    with pytest.raises(ValueError, match="valid scaling Strategy"):
        koco.ApplyScalingPlan(Table, {"Columns": {"a": {"Strategy": "Median", "Value": 1}}})

def test_apply_scaling_validates_the_plan_and_columns():
    Table = pd.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
    Good = {"Strategy": "Robust", "Offset": 1.0, "Scale": 2.0}
    assert koco.ApplyScalingPlan(Table, {"Columns": {"a": Good}})["a"].tolist() == [0.0, 0.5]
    for Spec in ({**Good, "Scale": 0}, {**Good, "Scale": float("nan")}, {**Good, "Offset": "1"}, {"Strategy": "Robust"}, "text"):
        with pytest.raises(ValueError, match="non-zero Scale"):
            koco.ApplyScalingPlan(Table, {"Columns": {"a": Spec}})
    with pytest.raises(ValueError, match="numeric"):
        koco.ApplyScalingPlan(Table, {"Columns": {"s": Good}})
    with pytest.raises(ValueError, match="not found"):
        koco.ApplyScalingPlan(Table, {"Columns": {"zz": Good}})
    with pytest.raises(ValueError, match="Columns"):
        koco.ApplyScalingPlan(Table, {"PlanType": "Scaling"})
    assert koco.ApplyScalingPlan(Table, {"PlanType": "Scaling", "Columns": {"a": Good}})["a"].tolist() == [0.0, 0.5]

def test_profile_null_columns_exports_only_when_a_path_is_given(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Stamp = date.today().strftime("%y%m%d")
    Profile = koco.ProfileNullColumns(NullTable())
    assert set(Profile["ColumnName"]) == {"Income", "Age", "Seg"}
    assert list(tmp_path.iterdir()) == []

    koco.ProfileNullColumns(NullTable(), FilePath="Data/NullAnalysis/Profile.xlsx")
    assert (tmp_path / "Data" / "NullAnalysis" / "Profile.xlsx").exists()
    koco.ProfileNullColumns(NullTable(), FilePath="Data/NullAnalysis")
    assert (tmp_path / "Data" / "NullAnalysis" / f"{Stamp}_NullProfile.xlsx").exists()
    capsys.readouterr()
    koco.ProfileNullColumns(NullTable(), FilePath=tmp_path / "Data" / "NullAnalysis")
    assert capsys.readouterr().out.count("same name") == 1

    assert koco.ProfileNullColumns(pd.DataFrame({"a": [1]}), FilePath="Nothing").empty
    assert not (tmp_path / "Nothing").exists()
    for Invalid in (True, False, 1.5):
        with pytest.raises(TypeError):
            koco.ProfileNullColumns(NullTable(), FilePath=Invalid)

def test_no_plan_function_checks_a_plan_type():
    Plan, _ = koco.FitNullPlan(NullTable(), [("Income", "Median")])
    assert "PlanType" not in Plan and "PlanType" not in koco.FitScalingPlan(NullTable(), {"Robust": ["Flag"]})

# ---------- WOE ----------

def test_fit_woe_plan_reproduces_the_original_algorithm():
    # Reference values come from the original WOETransformation function on the same table.
    Table = WOETable()
    Plan, Binning, Selection = koco.FitWOEPlan(Table, "y", CoverBound=0.05, MeanDiffBound=0.02)

    Expected = {"score": 0.530185, "sparse": 0.185839, "age": 0.167398, "income": 0.053832, "noise": 0.02186, "tiny_null": 0.01439, "const": 0.0}
    assert Selection["ColumnName"].tolist() == list(Expected)
    assert Selection["IV"].round(6).tolist() == pytest.approx(list(Expected.values()), abs=2e-6)
    assert Selection.set_index("ColumnName")["Selection"].to_dict() == {"score": 1, "sparse": 1, "age": 1, "income": 1, "noise": 1, "tiny_null": 0, "const": 0}
    assert set(Plan["Columns"]) == {"score", "sparse", "age", "income", "noise"}

    Score = Plan["Columns"]["score"]
    assert Score["Cuts"] == pytest.approx([-1.115825, -0.65657, 0.286758, 0.667316, 1.15991], abs=1e-6)
    assert Score["WOE"] == pytest.approx([1.54651, 0.709541, 0.289997, -0.009693, -0.505104, -1.174289], abs=1e-6)
    assert Score["Bins"] == [1, 2, 3, 4, 5, 6] and Score["NullBin"] is None and Score["NullWOE"] is None
    assert Plan["Columns"]["age"]["Cuts"] == [26.0, 33.0, 45.0, 48.0, 65.0]
    Sparse = Plan["Columns"]["sparse"]
    assert Sparse["NullBin"] == 6 and Sparse["NullWOE"] == pytest.approx(0.137996, abs=1e-6)
    assert Binning.query("ColumnName == 'sparse'")["Freq"].tolist() == [518, 260, 518, 519, 260, 925]

def test_fit_woe_plan_binning_table_is_consistent():
    Table = WOETable()
    _, Binning, _ = koco.FitWOEPlan(Table, "y", CoverBound=0.05, MeanDiffBound=0.02)
    Columns = ["ColumnName", "Bin", "LowerBound", "UpperBound", "Coverage", "Rate", "Freq", "Target", "NonTarget", "BinFreq",
               "BinTarget", "BinNonTarget", "BinPercTarget", "BinPercNonTarget", "WOE", "IV", "BinClass"]
    assert list(Binning.columns) == Columns
    for Feature, Group in Binning.groupby("ColumnName"):
        assert Group["Freq"].sum() == Table[Feature].notna().sum() + Table[Feature].isna().sum()
        assert Group["Target"].sum() == Table["y"].sum()
        assert Group["Coverage"].sum() == pytest.approx(1)
    Score = Binning.query("ColumnName == 'score'")
    assert Score["LowerBound"].iloc[0] == -np.inf and Score["UpperBound"].iloc[-1] == np.inf
    assert sorted(Score["BinClass"].unique()) == [-1, 0, 1]

def test_fit_woe_plan_default_bounds_work_on_small_tables():
    # The original function raised ZeroDivisionError here (fewer than 5,000 rows with the default bounds).
    Plan, Binning, Selection = koco.FitWOEPlan(WOETable(Rows=3000), "y")
    assert len(Selection) == 7 and not Binning.empty

def test_woe_plan_round_trip_and_apply(tmp_path):
    Table = WOETable()
    Plan, _, _ = koco.FitWOEPlan(Table, "y", CoverBound=0.05, MeanDiffBound=0.02)
    koco.ExportJson(Plan, tmp_path / "woe.json")
    Loaded = koco.ImportJson(tmp_path / "woe.json")
    assert Loaded == Plan
    Before = Table.copy()
    Result = koco.ApplyWOEPlan(Table, Loaded)
    pd.testing.assert_frame_equal(Table, Before)
    for Column in Plan["Columns"]:
        assert Result[f"{Column}_WOE"].notna().all() and Result[f"{Column}_Bin"].dtype == np.int64
    Score = Plan["Columns"]["score"]
    Inside = Result.loc[Table["score"].between(-0.6, 0.2), "score_WOE"]
    assert (Inside == Score["WOE"][2]).all()                       # the bin (-0.65657, 0.286758]
    NullRows = Table["sparse"].isna()
    assert (Result.loc[NullRows, "sparse_Bin"] == 6).all() and (Result.loc[NullRows, "sparse_WOE"] == Plan["Columns"]["sparse"]["NullWOE"]).all()

def test_apply_woe_plan_edges_and_unseen_nulls(capsys):
    Plan = {"Columns": {"x": {"Cuts": [1.0, 2.0], "Bins": [1, 2, 3], "WOE": [-1.0, 0.0, 1.0], "NullBin": None, "NullWOE": None}}}
    Table = pd.DataFrame({"x": [-5.0, 1.0, 1.5, 2.0, 2.5, np.inf, np.nan]})
    Result = koco.ApplyWOEPlan(Table, Plan)
    assert Result["x_Bin"].tolist() == [1, 1, 2, 2, 3, 3, 0]
    assert Result["x_WOE"].tolist() == [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert "x: 1 null values were not seen" in capsys.readouterr().out
    WithNull = {"Columns": {"x": {**Plan["Columns"]["x"], "NullBin": 4, "NullWOE": 0.7}}}
    assert koco.ApplyWOEPlan(Table, WithNull)["x_WOE"].iloc[-1] == 0.7

def test_woe_zero_adjustment_keeps_woe_finite():
    Rng = np.random.RandomState(1)
    Table = pd.DataFrame({"x": Rng.normal(size=2000)})
    Table["y"] = (Table["x"] > 1.0).astype(int)                    # the top bin has no non-events
    Plan, Binning, Selection = koco.FitWOEPlan(Table, "y", CoverBound=0.05, MeanDiffBound=0.02)
    assert np.isfinite(Binning["WOE"]).all() and np.isfinite(Binning["IV"]).all()
    assert Selection["Selection"].tolist() == [1] and json.dumps(Plan)
    with pytest.raises(ValueError, match="ZeroAdjustment"):
        koco.FitWOEPlan(Table, "y", ZeroAdjustment=0)

def test_fit_woe_plan_all_null_feature_and_no_selection(capsys):
    Table = WOETable(Rows=500)
    Table["empty"] = np.nan
    Plan, Binning, Selection = koco.FitWOEPlan(Table, "y", Features=["empty", "const"])
    assert Plan == {"Columns": {}} and Selection["IV"].tolist() == [0.0, 0.0]
    assert "empty has only null values" in capsys.readouterr().out
    assert koco.ApplyWOEPlan(Table, Plan).shape == Table.shape

def test_fit_woe_plan_validation():
    Table = WOETable(Rows=500)
    with pytest.raises(ValueError, match="Binary Target"):
        koco.FitWOEPlan(Table, "income")
    with pytest.raises(ValueError, match="CoverBound"):
        koco.FitWOEPlan(Table, "y", CoverBound=0)
    with pytest.raises(ValueError, match="MeanDiffBound"):
        koco.FitWOEPlan(Table, "y", IVBound=-1)
    with pytest.raises(ValueError, match="numeric"):
        koco.FitWOEPlan(Table.assign(text="a"), "y", Features=["text"])
    with pytest.raises(ValueError, match="not found"):
        koco.FitWOEPlan(Table, "y", Features=["missing"])
    with pytest.raises(ValueError, match="Target"):
        koco.FitWOEPlan(Table.assign(y=Table["y"].astype("float").where(Table.index > 0)), "y")

def test_apply_woe_plan_validation():
    Good = {"Cuts": [1.0], "Bins": [1, 2], "WOE": [0.1, -0.1], "NullBin": None, "NullWOE": None}
    Table = pd.DataFrame({"x": [0.0, 2.0], "s": ["a", "b"]})
    assert koco.ApplyWOEPlan(Table, {"Columns": {"x": Good}}).shape == (2, 4)
    for Bad in ({**Good, "Bins": [1]}, {**Good, "Cuts": [2.0, 1.0], "Bins": [1, 2, 3], "WOE": [0, 0, 0]},
                {**Good, "WOE": [0.1, float("nan")]}, {**Good, "NullBin": 3}, {"Cuts": [1.0]}, "text"):
        with pytest.raises(ValueError, match="valid WOE entry"):
            koco.ApplyWOEPlan(Table, {"Columns": {"x": Bad}})
    with pytest.raises(ValueError, match="not found"):
        koco.ApplyWOEPlan(Table, {"Columns": {"zz": Good}})
    with pytest.raises(ValueError, match="numeric"):
        koco.ApplyWOEPlan(Table, {"Columns": {"s": Good}})
    with pytest.raises(ValueError, match="Columns"):
        koco.ApplyWOEPlan(Table, {"x": Good})
