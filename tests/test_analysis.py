import matplotlib

matplotlib.use("Agg")

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import koco
from _samples import WOETable

def test_aggregation_report_rejects_non_tuple_sort_items():
    Table = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 1, 5]})
    with pytest.raises(TypeError):
        koco.AggregationReport(Table, ["g"], Sort=["Rate"])
    with pytest.raises(TypeError):
        koco.AggregationReport(Table, ["g"], Sort=[["Rate", "Desc"]])
    Report = koco.AggregationReport(Table, ["g"], Agg={"v": ["sum"]}, Sort=[("sumv", "Desc")])
    assert Report["g"].tolist() == ["b", "a"]
    assert Report["Rate"].sum() == pytest.approx(100)

def test_column_binning_and_sample_binning():
    Table = pd.DataFrame({"x": range(100)})
    Binned = koco.ColumnBinning(Table, "x", Bins=4)
    assert Binned["Observation"].sum() == 100 and Binned["Rate"].sum() == pytest.approx(100)
    assert koco.SampleBinning(Table, "x", Bins=10)["Observation"].tolist() == [10] * 10
    with pytest.raises(ValueError, match="not found"):
        koco.ColumnBinning(Table, "missing")

def test_data_summary_statistics():
    Table = pd.DataFrame({"n": [1.0, 2.0, 3.0, None], "s": ["a", "a", "b", None]})
    Summary = koco.DataSummary(Table).set_index("ColumnName")
    assert Summary.loc["n", "NullCount"] == 1
    assert Summary.loc["n", "MedValue"] == 2.0
    assert Summary.loc["n", "P99Value"] == pytest.approx(2.98)
    assert Summary.loc["s", "ModValue"] == "a"
    assert np.isnan(Summary.loc["s", "AvgValue"])

def test_cross_check_keeps_quoted_text_and_supports_backticks():
    Table = pd.DataFrame({"Name": ["A AND B", "x", "A and b"], "Net Income": [1.0, None, 3.0]})
    Result = koco.CrossCheck(Table, {
        "Literal": "Name = 'A AND B'",
        "Spaced": "`Net Income` IS NULL",
        "Combined": "Name = 'x' OR `Net Income` IS NOT NULL",
    }).set_index("Control")
    assert Result["Count"].to_dict() == {"Literal": 1, "Spaced": 1, "Combined": 3}
    with pytest.raises(ValueError):
        koco.CrossCheck(Table, {})
    with pytest.raises(ValueError, match="Failed to evaluate"):
        koco.CrossCheck(Table, {"Bad": "Unknown > 1"})

def test_outlier_report_methods_and_errors():
    Table = pd.DataFrame({"x": [1.0] * 20 + [100.0], "s": ["a"] * 21})
    Both = koco.OutlierReport(Table)
    assert Both.loc[0, "IQROutlierCount"] == 1
    One = koco.OutlierReport(Table, Method="IQR")
    assert list(One.columns) == ["Column", "Method", "LowerBound", "UpperBound", "OutlierCount", "OutlierRate"]
    with pytest.raises(ValueError, match="Method"):
        koco.OutlierReport(Table, Method="iqr")
    with pytest.raises(ValueError, match="numeric"):
        koco.OutlierReport(Table[["s"]])

def test_outlier_count_by_variable_prints_percentile_labels(capsys):
    Table = pd.DataFrame({"x": np.arange(1, 101, dtype=float)})
    koco.OutlierCountByVariable(Table, "x")
    Output = capsys.readouterr().out
    assert "P1 (" in Output and "P99 (" in Output and "Q1" not in Output
    koco.OutlierCountByVariable(Table, "x", Thresholds=[("Cap", 90)])
    assert "Cap (90.00)" in capsys.readouterr().out
    with pytest.raises(ValueError):
        koco.OutlierCountByVariable(Table, "missing")

def test_target_divergence_detects_different_groups():
    Rng = np.random.RandomState(0)
    Table = pd.DataFrame({"g": ["a"] * 200 + ["b"] * 200, "y": np.r_[Rng.normal(0, 1, 200), Rng.normal(3, 1, 200)]})
    Result = koco.TargetDivergenceByColumn(Table, "y", "g", MinObservationSize=100)
    assert Result["Divergent"] and Result["GroupCount"] == 2

def test_draw_chart_saves_with_default_or_given_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Table = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    Stamp = date.today().strftime("%y%m%d")
    Default = koco.DrawChart(Table, BarColumn="x", Title="a/b:c", SaveImage=True)
    assert Default == f"{Stamp}_DrawChart.png" and (tmp_path / Default).exists()
    Named = koco.DrawChart(Table, BarColumn="x", FilePath="Charts/Hist.png")
    assert Named == str(Path("Charts/Hist.png")) and (tmp_path / "Charts" / "Hist.png").exists()
    InFolder = koco.DrawChart(Table, BarColumn="x", FilePath="Charts/More")
    assert InFolder == str(Path("Charts/More") / f"{Stamp}_DrawChart.png")
    assert koco.DrawChart(Table, BarColumn="x") is None
    with pytest.raises(ValueError):
        koco.DrawChart(Table)
    with pytest.raises(ValueError, match="not found"):
        koco.DrawChart(Table, BarColumn="missing")
    with pytest.raises(TypeError):
        koco.DrawChart(Table, BarColumn="x", FilePath=5)

# ---------- RepresentationReport ----------

def _Population():
    Population = WOETable(Rows=4000, Seed=1)
    Population["zeros"] = 0.0
    return Population

def test_representation_report_reproduces_the_original_rep_function():
    # Reference values come from the original Rep function on the same tables.
    Population = _Population()
    Random = koco.RepresentationReport(Population, Population.sample(800, random_state=0)).set_index("ColumnName")["Rep"]
    assert Random.round(6).to_dict() == pytest.approx({
        "const": 1.0, "zeros": 1.0, "y": 0.986433, "income": 0.95888, "age": 0.915273,
        "tiny_null": 0.634063, "noise": -2.724389, "sparse": -6.800598, "score": -12.8782,
    }, abs=2e-6)
    Shifted = koco.RepresentationReport(Population, Population.assign(score=Population["score"] + 2).sample(800, random_state=2))
    assert Shifted["Rep"].is_monotonic_decreasing and Shifted.iloc[-1]["ColumnName"] == "score"
    assert Shifted.set_index("ColumnName").loc["score", "Rep"] == pytest.approx(-365.207964, abs=1e-4)

def test_representation_report_identical_tables_and_detail(capsys):
    Population = _Population()
    Report = koco.RepresentationReport(Population, Population, Detail=True, Statistics=["AvgValue", "NullRate"])
    assert (Report["Rep"] == 1.0).all() and list(Report.columns) == ["ColumnName", "Rep", "AvgValue", "NullRate"]
    assert (Report[["AvgValue", "NullRate"]] == 0).all().all()
    assert "Representation power of the sample: 100.00" in capsys.readouterr().out

def test_representation_report_zero_and_missing_rules():
    Population = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0], "b": [1.0, 2.0, 3.0, 4.0], "c": [np.nan, np.nan, np.nan, np.nan]})
    Sample = pd.DataFrame({"a": [0.0, 0.0], "b": [1.0, 2.0], "c": [1.0, 2.0]})
    Report = koco.RepresentationReport(Population, Sample, Statistics=["AvgValue", "ZeroRate"], Detail=True).set_index("ColumnName")
    assert Report.loc["a", "Rep"] == 1.0                                   # 0 vs 0 is no difference
    assert Report.loc["c", "AvgValue"] == 1.0                              # missing in one table only
    assert Report.loc["c", "ZeroRate"] == 0.0 and Report.loc["b", "AvgValue"] == pytest.approx(-0.4)
    assert koco.RepresentationReport(Population[["c"]], Population[["c"]], Statistics=["AvgValue"]).loc[0, "Rep"] == 1.0
    PopulationZero = pd.DataFrame({"a": [0.0, 0.0]})
    assert koco.RepresentationReport(PopulationZero, pd.DataFrame({"a": [1.0, 2.0]}), Statistics=["AvgValue"]).loc[0, "Rep"] == 0.0

def test_representation_report_validation():
    Table = pd.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"], "b": [True, False]})
    with pytest.raises(TypeError):
        koco.RepresentationReport(Table, [1, 2])
    with pytest.raises(ValueError, match="Statistics"):
        koco.RepresentationReport(Table, Table, Statistics=["Nope"])
    with pytest.raises(ValueError, match="Statistics"):
        koco.RepresentationReport(Table, Table, Statistics=[])
    with pytest.raises(ValueError, match="no numeric column"):
        koco.RepresentationReport(Table[["s", "b"]], Table[["s", "b"]])
    with pytest.raises(ValueError, match="no numeric column"):
        koco.RepresentationReport(Table[["a"]], Table[["s"]])
    assert koco.RepresentationReport(Table, Table)["ColumnName"].tolist() == ["a"]
