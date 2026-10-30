import numpy as np
import pandas as pd
import pytest

import koco
from _samples import BinaryTable, RegressionTable

MEAN_ABS_Y = 2.0

def RegressionPerformance(Models: dict, Label: str = "L1") -> pd.DataFrame:
    """Models: name -> {sample: (WMAPE, R2, Bias)}; MAE and RMSE follow from WMAPE."""
    Rows = [
        {"Label": Label, "AlgorithmName": Name, "SampleName": Sample, "RMSE": Wmape * MEAN_ABS_Y * 1.2,
         "MAE": Wmape * MEAN_ABS_Y, "WMAPE": Wmape, "R2": R2, "Bias": Bias}
        for Name, Samples in Models.items() for Sample, (Wmape, R2, Bias) in Samples.items()
    ]
    return pd.DataFrame(Rows)

def BinaryPerformance(Models: dict, Label: str = "B1") -> pd.DataFrame:
    """Models: name -> {sample: (AUC, LogLoss)}; Gini is 2 * AUC - 1."""
    Rows = [
        {"Label": Label, "AlgorithmName": Name, "SampleName": Sample, "AUC": Auc, "Gini": 2 * Auc - 1, "KS": Auc - 0.1,
         "LogLoss": LogLoss, "Brier": LogLoss / 3}
        for Name, Samples in Models.items() for Sample, (Auc, LogLoss) in Samples.items()
    ]
    return pd.DataFrame(Rows)

def Regression(Models: dict) -> pd.DataFrame:
    return RegressionPerformance({"DummyR": {"Train": (1.0, 0.0, 0.0), "Test": (1.0, 0.0, 0.0)}, **Models})

GOOD = {"Train": (0.20, 0.90, 0.0), "Test": (0.21, 0.88, 0.01)}

def Row(Result: pd.DataFrame, Name: str) -> pd.Series:
    return Result.loc[Result["AlgorithmName"] == Name].iloc[0]

def test_select_best_model_regression_ranks_and_explains():
    Performance = Regression({
        "Good": GOOD,
        "Weaker": {"Train": (0.30, 0.80, 0.0), "Test": (0.32, 0.78, 0.01)},
        "Overfit": {"Train": (0.10, 0.99, 0.0), "Test": (0.40, 0.60, 0.01)},
        "NoGain": {"Train": (0.95, 0.10, 0.0), "Test": (0.95, 0.10, 0.0)},
        "Biased": {"Train": (0.20, 0.90, 0.0), "Test": (0.21, 0.88, 0.50)},
    })
    Before = Performance.copy()
    Result = koco.SelectBestModel(Performance)
    pd.testing.assert_frame_equal(Performance, Before)
    assert list(Result["AlgorithmName"][:2]) == ["Good", "Weaker"]
    assert Result["AlgorithmName"].iloc[-1] == "DummyR" and Row(Result, "DummyR")["Result"] == "Baseline"
    assert Result["Recommended"].sum() == 1 and Row(Result, "Good")["Recommended"]
    assert Row(Result, "NoGain")["Baseline"] == "Fail" and Row(Result, "NoGain")["Result"] == "Eliminated"
    assert Row(Result, "Biased")["Bias"] == "Fail" and Row(Result, "Biased")["Result"] == "Eliminated"
    assert Row(Result, "Overfit")["Overfit"] == "Fail" and Row(Result, "Overfit")["Result"] == "Eliminated"
    assert Row(Result, "Good")["FitGap"] == pytest.approx(0.02)
    assert Row(Result, "Good")["RelativeBias"] == pytest.approx(0.01 / MEAN_ABS_Y)

def test_select_best_model_ignores_other_samples():
    Performance = Regression({"Good": GOOD, "Weaker": {"Train": (0.30, 0.80, 0.0), "Test": (0.32, 0.78, 0.01)}})
    Extended = pd.concat([Performance, Performance.assign(SampleName="OOT", WMAPE=9.0, R2=-5.0)], ignore_index=True)
    pd.testing.assert_frame_equal(koco.SelectBestModel(Performance), koco.SelectBestModel(Extended))

def test_select_best_model_tie_prefers_smaller_fit_gap():
    Performance = Regression({
        "Wide": {"Train": (0.10, 0.95, 0.0), "Test": (0.200, 0.88, 0.0)},
        "Narrow": {"Train": (0.19, 0.90, 0.0), "Test": (0.201, 0.88, 0.0)},
        "Far": {"Train": (0.30, 0.80, 0.0), "Test": (0.300, 0.78, 0.0)},
    })
    Result = koco.SelectBestModel(Performance)
    assert list(Result["TieBand"][:3]) == ["Yes", "Yes", "No"]
    assert list(Result["Result"][:3]) == [1, 1, 3]
    assert Result["AlgorithmName"].iloc[0] == "Narrow" and Result["Recommended"].iloc[0]

def test_select_best_model_undefined_relative_bias_stays_eligible():
    Performance = Regression({"Good": GOOD, "Zero": {"Train": (0.0, 0.5, 0.0), "Test": (0.0, 0.5, 0.0)}})
    Result = koco.SelectBestModel(Performance)
    assert Row(Result, "Zero")["Bias"] == "NA" and Row(Result, "Zero")["Result"] != "Eliminated"

def test_select_best_model_falls_back_when_every_model_memorises(capsys):
    Performance = Regression({
        "A": {"Train": (0.01, 0.9995, 0.0), "Test": (0.02, 0.9990, 0.0)},
        "B": {"Train": (0.01, 0.9996, 0.0), "Test": (0.03, 0.9980, 0.0)},
    })
    Result = koco.SelectBestModel(Performance)
    assert list(Result["Memorize"][:2]) == ["Fail", "Fail"]
    assert Result["Recommended"].sum() == 1 and Result["AlgorithmName"].iloc[0] == "A"
    assert "ranking the models that pass Overfit" in capsys.readouterr().out
    assert not koco.SelectBestModel(Performance, MemorizationLimit=0.99999)["Memorize"].eq("Fail").any()

def test_select_best_model_recommends_nothing_when_all_are_eliminated(capsys):
    Performance = Regression({"NoGain": {"Train": (0.95, 0.10, 0.0), "Test": (0.95, 0.10, 0.0)}})
    Result = koco.SelectBestModel(Performance)
    assert not Result["Recommended"].any()
    assert "nothing is recommended" in capsys.readouterr().out

def test_select_best_model_reports_duplicates_and_missing_metrics(capsys):
    Performance = Regression({"Good": GOOD, "Other": {"Train": (0.30, 0.80, 0.0), "Test": (0.32, 0.78, 0.01)}})
    Duplicated = pd.concat([Performance, Performance.loc[Performance["AlgorithmName"] == "Good"].iloc[[0]]], ignore_index=True)
    Result = koco.SelectBestModel(Duplicated)
    assert Row(Result, "Good")["Data"] == "Fail" and Row(Result, "Good")["Result"] == "Eliminated"
    assert "Duplicate rows" in capsys.readouterr().out
    Missing = Performance.copy()
    Missing.loc[(Missing["AlgorithmName"] == "Other") & (Missing["SampleName"] == "Test"), "WMAPE"] = np.nan
    assert Row(koco.SelectBestModel(Missing), "Other")["Data"] == "Fail"
    assert "Missing or non-finite" in capsys.readouterr().out

def test_select_best_model_without_baseline_skips_the_step(capsys):
    Result = koco.SelectBestModel(Regression({"Good": GOOD}).query("AlgorithmName != 'DummyR'"))
    assert Result["Baseline"].iloc[0] == "NA" and Result["Recommended"].iloc[0]
    assert "No usable DummyR" in capsys.readouterr().out

def test_select_best_model_compares_each_label_with_its_own_baseline():
    Performance = pd.concat([Regression({"Good": GOOD}), Regression({"Good": GOOD}).assign(Label="L2")], ignore_index=True)
    Result = koco.SelectBestModel(Performance)
    assert set(Result.loc[Result["AlgorithmName"] == "Good", "Baseline"]) == {"Pass"}
    assert Result["Recommended"].sum() == 1

def test_select_best_model_binary():
    Performance = BinaryPerformance({
        "DummyC": {"Train": (0.5, 0.69), "Test": (0.5, 0.69)},
        "Logistic": {"Train": (0.90, 0.40), "Test": (0.88, 0.42)},
        "Forest": {"Train": (1.00, 0.05), "Test": (0.89, 0.45)},
    })
    Result = koco.SelectBestModel(Performance)
    assert "Bias" not in Result.columns and "RelativeBias" not in Result.columns
    assert Result["AlgorithmName"].iloc[0] == "Logistic" and Result["Recommended"].iloc[0]
    assert Row(Result, "Forest")["Memorize"] == "Fail" and Row(Result, "Forest")["Result"] == "Eliminated"
    assert "Gini_Test" in Result.columns

def test_select_best_model_on_real_modelling_output():
    _, Performance, _, _ = koco.RegressionModelling(RegressionTable(), ["DummyR", "RidgeR", "RandFR"], "y")
    Result = koco.SelectBestModel(Performance)
    assert Result["Recommended"].sum() == 1 and Result["AlgorithmName"].iloc[-1] == "DummyR"
    _, Performance, _, _ = koco.ClassificationModelling(BinaryTable(), ["DummyC", "LogisticC"], "y")
    assert koco.SelectBestModel(Performance)["Recommended"].sum() == 1

def test_select_best_model_validation():
    Performance = Regression({"Good": GOOD})
    with pytest.raises(TypeError):
        koco.SelectBestModel([1, 2])
    with pytest.raises(ValueError, match="empty"):
        koco.SelectBestModel(Performance.iloc[:0])
    with pytest.raises(ValueError, match="SampleName"):
        koco.SelectBestModel(Performance.drop(columns="SampleName"))
    with pytest.raises(ValueError, match="Binary metrics"):
        koco.SelectBestModel(Performance.drop(columns="RMSE"))
    with pytest.raises(ValueError, match="different sample"):
        koco.SelectBestModel(Performance, SelectionSample="Train")
    with pytest.raises(ValueError, match="not found"):
        koco.SelectBestModel(Performance, SelectionSample="Validation")
    with pytest.raises(ValueError, match="MinImprovementOverBaseline"):
        koco.SelectBestModel(Performance, MinImprovementOverBaseline=1)
    with pytest.raises(ValueError, match="MaxFitGap"):
        koco.SelectBestModel(Performance, MaxFitGap=-0.1)
    with pytest.raises(ValueError, match="TieTolerance"):
        koco.SelectBestModel(Performance, TieTolerance=True)
    with pytest.raises(ValueError, match="Columns not found"):
        koco.SelectBestModel(Performance.drop(columns="Bias"))

def Holdout(Test=(0.20, 0.9, 0.0), OOT=(0.22, 0.9, 0.01)) -> pd.DataFrame:
    return RegressionPerformance({"M": {"Test": Test, "OOT": OOT}})

def test_evaluate_on_holdout_regression():
    Performance = Holdout()
    Before = Performance.copy()
    Result = koco.EvaluateOnHoldout(Performance, "L1", "M")
    pd.testing.assert_frame_equal(Performance, Before)
    assert Result["Decay"] == pytest.approx(0.1) and Result["RelativeBias"] == pytest.approx(0.005)
    assert Result["WMAPE_Test"] == 0.2 and Result["WMAPE_OOT"] == 0.22 and Result["Pass"] is True
    assert not koco.EvaluateOnHoldout(Holdout(OOT=(0.40, 0.9, 0.0)), "L1", "M")["Pass"]
    assert not koco.EvaluateOnHoldout(Holdout(OOT=(0.22, 0.9, 0.5)), "L1", "M")["Pass"]
    assert koco.EvaluateOnHoldout(Holdout(OOT=(0.15, 0.9, 0.0)), "L1", "M")["Decay"] < 0

def test_evaluate_on_holdout_binary():
    Performance = BinaryPerformance({"M": {"Test": (0.90, 0.4), "OOT": (0.80, 0.5)}})
    Result = koco.EvaluateOnHoldout(Performance, "B1", "M")
    assert "RelativeBias" not in Result.index and Result["Decay"] == pytest.approx(0.25) and Result["Pass"]
    assert not koco.EvaluateOnHoldout(Performance, "B1", "M", MaxDecay=0.1)["Pass"]

def test_evaluate_on_holdout_skips_zero_denominators_and_fails_missing_metrics(capsys):
    Result = koco.EvaluateOnHoldout(Holdout(OOT=(0.0, 0.9, 0.0)), "L1", "M")
    assert np.isnan(Result["RelativeBias"]) and Result["Pass"]
    assert "skipped" in capsys.readouterr().out
    Skipped = koco.EvaluateOnHoldout(Holdout(Test=(0.0, 0.9, 0.0), OOT=(0.0, 0.9, 0.0)), "L1", "M")
    assert not Skipped["Pass"]
    Missing = Holdout(OOT=(np.nan, 0.9, 0.0))
    assert not koco.EvaluateOnHoldout(Missing, "L1", "M")["Pass"]
    assert "missing" in capsys.readouterr().out

def test_evaluate_on_holdout_validation():
    Performance = Holdout()
    with pytest.raises(ValueError, match="exactly one"):
        koco.EvaluateOnHoldout(Performance, "L1", "Unknown")
    with pytest.raises(ValueError, match="exactly one"):
        koco.EvaluateOnHoldout(pd.concat([Performance, Performance]), "L1", "M")
    with pytest.raises(ValueError, match="different sample"):
        koco.EvaluateOnHoldout(Performance, "L1", "M", HoldoutSample="Test")
    with pytest.raises(ValueError, match="MaxDecay"):
        koco.EvaluateOnHoldout(Performance, "L1", "M", MaxDecay=-1)
    with pytest.raises(TypeError):
        koco.EvaluateOnHoldout("table", "L1", "M")

def RegressionPredictions(Rows: int = 400) -> pd.DataFrame:
    Rng = np.random.RandomState(0)
    Y = Rng.uniform(1, 10, Rows)
    return pd.DataFrame({"y": Y, "Good": Y + Rng.normal(0, 0.5, Rows), "Same": Y + Rng.normal(0, 0.5, Rows), "Bad": Y + Rng.normal(0, 3.0, Rows)})

def BinaryPredictions(Rows: int = 600) -> pd.DataFrame:
    Rng = np.random.RandomState(0)
    Y = Rng.randint(0, 2, Rows)
    Strong = np.clip(0.5 + (Y - 0.5) * 0.8 + Rng.normal(0, 0.15, Rows), 0, 1)
    Noisy = np.clip(0.5 + (Y - 0.5) * 0.1 + Rng.normal(0, 0.3, Rows), 0, 1)
    return pd.DataFrame({"y": Y, "Strong": Strong, "Same": np.clip(Strong + Rng.normal(0, 0.01, Rows), 0, 1), "Noisy": Noisy})

CANDIDATES = {"Good": ("L1", "RidgeR"), "Same": ("L1", "RandFR"), "Bad": ("L1", "KNNR")}

def test_bootstrap_compare_regression_verdicts():
    Predictions = RegressionPredictions()
    Before = Predictions.copy()
    Result = koco.BootstrapCompare(Predictions, "y", CANDIDATES, "Regression", Resamples=200)
    pd.testing.assert_frame_equal(Predictions, Before)
    assert list(Result.columns) == ["Candidate", "Label", "AlgorithmName", "WMAPE", "Delta", "CILow", "CIHigh", "ProbBetter", "Verdict"]
    Verdicts = dict(zip(Result["Candidate"], Result["Verdict"]))
    assert Result["WMAPE"].is_monotonic_increasing and Result["Verdict"].eq("Reference").sum() == 1
    assert Verdicts["Bad"] == "Worse" and Verdicts["Same"] in {"Tied", "Worse", "Reference"}
    Reference = koco.BootstrapCompare(Predictions, "y", CANDIDATES, "Regression", Reference="Bad", Resamples=200)
    assert dict(zip(Reference["Candidate"], Reference["Verdict"]))["Good"] == "Better"
    assert (Result["CILow"] <= Result["CIHigh"]).all()

def test_bootstrap_compare_is_reproducible_and_reports_progress(capsys):
    Predictions = RegressionPredictions()
    First = koco.BootstrapCompare(Predictions, "y", CANDIDATES, "Regression", Resamples=100, RandomState=7)
    assert "Bootstrap" in capsys.readouterr().out
    pd.testing.assert_frame_equal(First, koco.BootstrapCompare(Predictions, "y", CANDIDATES, "Regression", Resamples=100, RandomState=7))

def test_bootstrap_compare_binary_metrics():
    Predictions = BinaryPredictions()
    for Metric in ("Gini", "AUC", "LogLoss"):
        Result = koco.BootstrapCompare(Predictions, "y", {"Strong": ("B1", "A"), "Noisy": ("B1", "B")}, "Binary", Metric=Metric, Resamples=100)
        assert Result["Candidate"].iloc[0] == "Strong" and dict(zip(Result["Candidate"], Result["Verdict"]))["Noisy"] == "Worse"
    Tied = koco.BootstrapCompare(Predictions, "y", {"Strong": ("B1", "A"), "Same": ("B1", "B")}, "Binary", Resamples=100)
    assert set(Tied["Verdict"]) == {"Reference", "Tied"}

def test_bootstrap_compare_validation():
    Predictions = RegressionPredictions()
    Binary = BinaryPredictions()
    Pair = {"Good": ("L1", "A"), "Bad": ("L1", "B")}
    with pytest.raises(TypeError):
        koco.BootstrapCompare([1], "y", Pair, "Regression")
    with pytest.raises(ValueError, match="ModelType"):
        koco.BootstrapCompare(Predictions, "y", Pair, "Other")
    with pytest.raises(ValueError, match="Metric"):
        koco.BootstrapCompare(Predictions, "y", Pair, "Regression", Metric="Gini")
    with pytest.raises(ValueError, match="at least two"):
        koco.BootstrapCompare(Predictions, "y", {"Good": ("L1", "A")}, "Regression")
    with pytest.raises(ValueError, match="twice"):
        koco.BootstrapCompare(Predictions, "y", {"Good": ("L1", "A"), "Bad": ("L1", "A")}, "Regression")
    with pytest.raises(ValueError, match="Resamples"):
        koco.BootstrapCompare(Predictions, "y", Pair, "Regression", Resamples=10)
    with pytest.raises(ValueError, match="ConfidenceLevel"):
        koco.BootstrapCompare(Predictions, "y", Pair, "Regression", ConfidenceLevel=1)
    with pytest.raises(ValueError, match="Columns not found"):
        koco.BootstrapCompare(Predictions.drop(columns="Bad"), "y", Pair, "Regression")
    with pytest.raises(ValueError, match="Reference"):
        koco.BootstrapCompare(Predictions, "y", Pair, "Regression", Reference="Other")
    with pytest.raises(ValueError, match="Target must not"):
        koco.BootstrapCompare(Predictions, "y", {"y": ("L1", "A"), "Bad": ("L1", "B")}, "Regression")
    with pytest.raises(ValueError, match="nulls"):
        koco.BootstrapCompare(Predictions.assign(Bad=np.nan), "y", Pair, "Regression")
    with pytest.raises(ValueError, match="every Target value is 0"):
        koco.BootstrapCompare(Predictions.assign(y=0.0), "y", Pair, "Regression")
    with pytest.raises(ValueError, match="0 and 1"):
        koco.BootstrapCompare(Binary.assign(y=Binary["y"] * 2), "y", {"Strong": ("B1", "A"), "Noisy": ("B1", "B")}, "Binary")
    with pytest.raises(ValueError, match="probabilities"):
        koco.BootstrapCompare(Binary.assign(Noisy=Binary["Noisy"] + 1), "y", {"Strong": ("B1", "A"), "Noisy": ("B1", "B")}, "Binary")
