from datetime import date

import numpy as np
import pandas as pd
import pytest

import koco
from _samples import BinaryTable, RegressionTable

def test_regression_metrics_known_values():
    Metrics = koco.RegressionMetrics([1, 2, 3, 4], [1, 2, 3, 6])
    assert Metrics["N"] == 4 and Metrics["MAE"] == 0.5 and Metrics["Bias"] == 0.5
    assert Metrics["RMSE"] == pytest.approx(1.0)
    Perfect = koco.RegressionMetrics([1, 2, 3], [1, 2, 3])
    assert Perfect["R2"] == 1 and Perfect["WMAPE"] == 0
    with pytest.raises(ValueError):
        koco.RegressionMetrics([], [])
    with pytest.raises(ValueError):
        koco.RegressionMetrics([1, 2], [1])

def test_classification_metrics_known_values():
    Metrics = koco.ClassificationMetrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], Threshold=0.5)
    assert Metrics["AUC"] == 0.75 and Metrics["Gini"] == 0.5
    assert (Metrics["TP"], Metrics["TN"], Metrics["FP"], Metrics["FN"]) == (1, 2, 0, 1)
    assert Metrics["Precision"] == 1 and Metrics["Recall"] == 0.5
    Single = koco.ClassificationMetrics([1, 1], [0.6, 0.7])
    assert np.isnan(Single["AUC"])
    with pytest.raises(ValueError):
        koco.ClassificationMetrics([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        koco.ClassificationMetrics([0, 1], [0.1, 1.2])

def test_regression_modelling_end_to_end():
    Artifacts, Performance, Log, TrainData = koco.RegressionModelling(RegressionTable(), ["RidgeR", "RandFR"], "y")
    assert set(Artifacts) == {"RidgeR", "RandFR"}
    assert set(Performance["SampleName"]) == {"Train", "Test"}
    assert list(Performance.columns) == koco.metrics.PERFORMANCE_COLUMNS["Regression"]
    assert Artifacts["RidgeR"]["ModelType"] == "Regression"
    Test = RegressionTable().drop(index=TrainData.index)
    assert Performance.query("AlgorithmName == 'RidgeR' and SampleName == 'Test'")["R2"].iloc[0] > 0.8
    assert len(koco.ApplyModel(Test, Artifacts["RidgeR"])) == len(Test)

def test_classification_modelling_end_to_end():
    Artifacts, Performance, Log, _ = koco.ClassificationModelling(BinaryTable(), ["LogisticC", "RandFC"], "y", ClassWeight="balanced")
    Scores = koco.ApplyModel(BinaryTable(), Artifacts["LogisticC"])
    assert list(Scores.columns) == ["Probability", "Prediction"]
    assert 0 < Artifacts["LogisticC"]["Threshold"] < 1
    assert Performance.query("AlgorithmName == 'LogisticC' and SampleName == 'Test'")["AUC"].iloc[0] > 0.9
    Fixed = koco.ApplyModel(BinaryTable(), Artifacts["LogisticC"], Threshold=0.99)
    assert Fixed["Prediction"].sum() <= Scores["Prediction"].sum()

def test_modelling_logs_instead_of_failing_for_single_algorithms():
    Table = RegressionTable()
    Table.loc[0, "a"] = np.nan
    Runnable, Log = koco.CheckModellingInputs(Table, ["RidgeR", "RandFR", "RandFC", "Unknown"], "y")
    assert Runnable == ["RandFR"]
    Errors = Log[Log["Severity"] == "Error"].set_index("AlgorithmName")["Message"]
    assert "NaN" in Errors["RidgeR"] and "Binary algorithm" in Errors["RandFC"] and "Unknown" in Errors["Unknown"]
    _, Log = koco.CheckModellingInputs(RegressionTable(), ["RidgeR"], "y", ModelParams={"RidgeR": {"alhpa": 1}})
    assert "Unknown parameters" in Log["Message"].iloc[-1]

def test_kwargs_estimators_warn_about_unrecognised_parameters():
    pytest.importorskip("xgboost")
    Runnable, Log = koco.CheckModellingInputs(RegressionTable(), ["XGBR"], "y", ModelParams={"XGBR": {"n_estmators": 5, "n_estimators": 5}})
    assert Runnable == ["XGBR"]
    Warnings = Log[Log["Severity"] == "Warning"]["Message"].tolist()
    assert any("n_estmators" in Message and "n_estimators'" not in Message for Message in Warnings)

def test_modelling_input_validation():
    Table = RegressionTable()
    with pytest.raises(ValueError, match="ModelType"):
        koco.CheckModellingInputs(Table, ["RidgeR"], "y", ModelType="Other")
    with pytest.raises(ValueError, match="Binary Target"):
        koco.ClassificationModelling(Table, ["LogisticC"], "y")
    with pytest.raises(ValueError, match="must not also appear"):
        koco.RegressionModelling(Table, ["RidgeR"], "y", Features=["a", "y"])
    with pytest.raises(ValueError, match="Unknown Split"):
        koco.RegressionModelling(Table, ["RidgeR"], "y", Split={"StratifyClasses": True})
    with pytest.raises(ValueError, match="Threshold"):
        koco.ClassificationModelling(BinaryTable(), ["LogisticC"], "y", Threshold=2)

def test_apply_model_requires_model_type_and_numeric_features():
    Table = RegressionTable()
    Artifacts, _, _, _ = koco.RegressionModelling(Table, ["RidgeR"], "y", Split=False)
    Artifact = Artifacts["RidgeR"]
    NoModelType = {Key: Value for Key, Value in Artifact.items() if Key != "ModelType"}
    with pytest.raises(ValueError, match="ModelType"):
        koco.ApplyModel(Table, NoModelType)
    with pytest.raises(ValueError, match="old 'Task' key"):
        koco.ApplyModel(Table, {**NoModelType, "Task": "Regression"})
    with pytest.raises(ValueError, match="ModelType must be"):
        koco.ApplyModel(Table, {**Artifact, "ModelType": "Other"})
    with pytest.raises(ValueError, match="not found"):
        koco.ApplyModel(Table.drop(columns="a"), Artifact)
    with pytest.raises(ValueError, match="numeric"):
        koco.ApplyModel(Table.assign(a="text"), Artifact)
    with pytest.raises(ValueError, match="Threshold"):
        koco.ApplyModel(Table, Artifact, Threshold=0.5)

def test_calculate_model_predictions_names_columns():
    Table = BinaryTable()
    Classifiers, _, _, _ = koco.ClassificationModelling(Table, ["LogisticC"], "y", Split=False)
    Regressors, _, _, _ = koco.RegressionModelling(RegressionTable(), ["RidgeR"], "y", Split=False)
    Result = koco.CalculateModelPredictions(Table.assign(Id=range(len(Table))), ["Id", "y"], {**Classifiers, **Regressors}, Rename={"LogisticC": "LR"})
    assert list(Result.columns) == ["Id", "y", "LRProbability", "LRPrediction", "RidgeR"]
    with pytest.raises(ValueError, match="overwrite"):
        koco.CalculateModelPredictions(Table.assign(RidgeR=1), ["RidgeR"], Regressors)

def test_train_test_split_stratification():
    Table = BinaryTable(Rows=400)
    Train, Test = koco.TrainTestSplit(Table, "y", TestSize=0.25, StratifyClasses=True)
    assert len(Test) == 100
    assert abs(Train["y"].mean() - Test["y"].mean()) < 0.02
    with pytest.raises(ValueError, match="Shuffle"):
        koco.TrainTestSplit(Table, "y", Shuffle=False, StratifyClasses=True)
    with pytest.raises(ValueError, match="float"):
        koco.TrainTestSplit(Table, "y", TestSize=1)
    Ordered_Train, Ordered_Test = koco.TrainTestSplit(Table, "y", Shuffle=False)
    assert Ordered_Train.index.max() < Ordered_Test.index.min()

def test_default_label_is_the_date_and_run():
    _, Performance, _, _ = koco.RegressionModelling(RegressionTable(), ["RidgeR"], "y")
    assert set(Performance["Label"]) == {f"{date.today().strftime('%y%m%d')}_Run"}
