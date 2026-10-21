import numpy as np
import pandas as pd
import pytest

import koco
from _samples import BinaryTable, NoiseTable, RegressionTable

def test_correlation_removes_the_weaker_duplicate():
    Rng = np.random.RandomState(0)
    Base = Rng.normal(size=200)
    Table = pd.DataFrame({"a": Base, "a_copy": Base + Rng.normal(size=200) * 0.01, "b": Rng.normal(size=200)})
    Table["y"] = Table["a"] + Rng.normal(size=200) * 0.1
    _, Remaining, Eliminated = koco.Correlation(Table, "y", Limit=0.9)
    assert len(Eliminated) == 1 and set(Eliminated) <= {"a", "a_copy"}
    assert "b" in Remaining
    with pytest.raises(ValueError, match="not found"):
        koco.Correlation(Table, "zz")

def test_vif_treats_bool_as_numeric_and_uses_threshold():
    Rng = np.random.RandomState(0)
    X1 = Rng.normal(size=200)
    Table = pd.DataFrame({"x1": X1, "x2": X1 + Rng.normal(size=200) * 0.05, "flag": Rng.rand(200) > 0.5, "y": Rng.normal(size=200)})
    Report, High = koco.VIFReport(Table, Target="y")
    assert "flag" in Report["Feature"].tolist()
    assert set(High) == {"x1", "x2"}
    _, HighStrict = koco.VIFReport(Table, Target="y", Threshold=1000)
    assert HighStrict == []

def test_vif_pre_check_errors_and_auto_clean():
    Table = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 1.0, 4.0, 3.0, 7.0], "s": list("abcde"), "k": [1] * 5})
    with pytest.raises(ValueError, match="pre-check failed"):
        koco.VIFReport(Table)
    Report, _ = koco.VIFReport(Table, AutoClean=True)
    assert set(Report["Feature"]) == {"a", "b"}

def test_model_features_select_finds_the_signal_in_both_methods():
    Table = RegressionTable()
    for Method in ("Combined", "OnlyNoise"):
        Report = koco.SelectModelFeatures(Table, ["a", "b", "c"], "y", Models="LassoR", CVFolds=3, NRepeats=3, EliminationMethod=Method)
        Selected = Report.loc[Report["Decision"] == "Select", "Feature"].tolist()
        assert {"a", "b"} <= set(Selected) and "c" not in Selected
        assert Report["IsNoiseProbe"].sum() == 10
        assert {"PermutationNoiseThreshold", "MutualInfoNoiseThreshold"} <= set(Report.columns)

def test_model_features_select_binary():
    Report = koco.SelectModelFeatures(
        BinaryTable(), ["a", "b", "c"], "y", Models="LogisticC", CVFolds=3, NRepeats=2, ModelType="Binary", EliminationMethod="OnlyNoise",
    )
    assert "a" in Report.loc[Report["Decision"] == "Select", "Feature"].tolist()

def test_only_noise_method_is_not_looser_than_combined_on_pure_noise():
    Selected = {}
    for Method in ("Combined", "OnlyNoise"):
        Total = 0
        for Seed in range(3):
            Table = NoiseTable(Features=10, Seed=Seed)
            Report = koco.SelectModelFeatures(
                Table, [Column for Column in Table.columns if Column != "y"], "y", Models="LassoR",
                CVFolds=3, NRepeats=2, RandomState=Seed, MinSelect=0, EliminationMethod=Method,
            )
            Total += int((Report["Decision"] == "Select").sum())
        Selected[Method] = Total
    assert Selected["OnlyNoise"] <= Selected["Combined"]

def test_model_features_select_input_validation():
    Table = RegressionTable()
    Cases = [
        (dict(EliminationMethod="Other"), "EliminationMethod"),
        (dict(ModelType="Other"), "ModelType"),
        (dict(MinSelect=9), "MinSelect"),
        (dict(Models="LogisticC"), "Unknown Regression model"),
        (dict(NNoiseFeatures=0), "NNoiseFeatures"),
        (dict(CVFolds=1), "CVFolds"),
    ]
    for Arguments, Message in Cases:
        with pytest.raises(ValueError, match=Message):
            koco.SelectModelFeatures(Table, ["a", "b", "c"], "y", **Arguments)
    with pytest.raises(ValueError, match="Duplicate"):
        koco.SelectModelFeatures(Table, ["a", "a"], "y")
    Table.loc[0, "a"] = np.nan
    with pytest.raises(ValueError, match="null"):
        koco.SelectModelFeatures(Table, ["a", "b"], "y")

# ---------- BackwardSelection ----------

def _SelectionTable(Seed=0, Rows=500):
    Rng = np.random.RandomState(Seed)
    Table = pd.DataFrame(Rng.normal(size=(Rows, 6)), columns=list("abcdef"))
    Table["y"] = 2 * Table["a"] + Table["b"] - 0.8 * Table["c"] + Rng.normal(size=Rows)
    Table["yb"] = (Rng.rand(Rows) < 1 / (1 + np.exp(-(1.5 * Table["a"] - Table["b"])))).astype(int)
    return Table

def test_backward_selection_regression_removes_noise_and_reports_steps():
    Table = _SelectionTable().drop(columns="yb")
    Remaining, Steps = koco.BackwardSelection(Table, "y")
    assert Remaining == ["a", "b", "c"]
    assert list(Steps.columns) == ["Step", "RemovedFeature", "PValue", "Remaining"]
    assert Steps["Step"].tolist() == [1, 2, 3] and Steps["Remaining"].tolist() == [5, 4, 3]
    assert set(Steps["RemovedFeature"]) == {"d", "e", "f"} and (Steps["PValue"] > 0.05).all()

def test_backward_selection_binary_uses_logit():
    Table = _SelectionTable().drop(columns="y")
    Remaining, _ = koco.BackwardSelection(Table, "yb", ModelType="Binary")
    assert Remaining == ["a", "b"]

def test_backward_selection_honours_max_p_value_and_features():
    Table = _SelectionTable().drop(columns="yb")
    Loose, _ = koco.BackwardSelection(Table, "y", MaxPValue=0.99)
    assert set(Loose) >= {"a", "b", "c"}
    Subset, _ = koco.BackwardSelection(Table, "y", Features=["a", "d"])
    assert Subset == ["a"]

def test_backward_selection_returns_nothing_when_no_feature_passes(capsys):
    Rng = np.random.RandomState(3)
    Table = pd.DataFrame(Rng.normal(size=(200, 3)), columns=list("abc"))
    Table["y"] = Rng.normal(size=200)
    Remaining, Steps = koco.BackwardSelection(Table, "y", MaxPValue=1e-6)
    assert Remaining == [] and len(Steps) == 3
    assert "No feature has a p-value" in capsys.readouterr().out

def test_backward_selection_validation():
    Table = _SelectionTable()
    with pytest.raises(ValueError, match="ModelType"):
        koco.BackwardSelection(Table.drop(columns="yb"), "y", ModelType="Other")
    with pytest.raises(ValueError, match="MaxPValue"):
        koco.BackwardSelection(Table.drop(columns="yb"), "y", MaxPValue=1)
    with pytest.raises(ValueError, match="Binary Target"):
        koco.BackwardSelection(Table.drop(columns="yb"), "y", ModelType="Binary")
    with pytest.raises(ValueError, match="null"):
        koco.BackwardSelection(Table.drop(columns="yb").assign(a=lambda D: D["a"].where(D.index > 0)), "y")
    with pytest.raises(ValueError, match="not found"):
        koco.BackwardSelection(Table, "y", Features=["zz"])

def test_backward_selection_separation_gives_a_clear_error():
    Table = pd.DataFrame({"x": np.arange(40.0), "z": np.random.RandomState(0).normal(size=40)})
    Table["y"] = (Table["x"] >= 20).astype(int)
    with pytest.raises(ValueError, match="could not be fitted|p-values could not"):
        koco.BackwardSelection(Table, "y", ModelType="Binary")
