import importlib

import pytest

import koco
from koco.guides import _METRIC_GUIDE, _NULL_GUIDE, _SCALING_GUIDE
from koco.metrics import PERFORMANCE_COLUMNS
from koco.preprocessing import NULL_STRATEGIES, SCALING_METHODS, _ScalerClasses
from koco.registry import MODEL_SPECS, REGISTRIES, BuildModelKwargs, LoadEstimatorClass

def test_every_key_matches_its_model_type_suffix():
    for Key, Spec in MODEL_SPECS.items():
        assert Key.endswith("R" if Spec.ModelType == "Regression" else "C"), Key
    assert sum(len(Registry) for Registry in REGISTRIES.values()) == len(MODEL_SPECS)

def test_every_spec_points_to_an_existing_estimator_class():
    for Key, Spec in MODEL_SPECS.items():
        if Spec.Module.split(".")[0] not in ("xgboost", "lightgbm", "catboost"):
            assert hasattr(importlib.import_module(Spec.Module), Spec.ClassName), Key

def test_nan_and_scaling_flags_of_known_models():
    assert not MODEL_SPECS["RidgeR"].NaNTolerant and MODEL_SPECS["RidgeR"].ScalingSensitive
    assert MODEL_SPECS["RandFR"].NaNTolerant and not MODEL_SPECS["RandFR"].ScalingSensitive
    assert not MODEL_SPECS["GaussNBC"].NaNTolerant and not MODEL_SPECS["GaussNBC"].ScalingSensitive

def test_build_model_kwargs_priority():
    EstimatorClass = LoadEstimatorClass("RandFR")
    assert BuildModelKwargs("RandFR", EstimatorClass, 7, None)["random_state"] == 7
    assert BuildModelKwargs("RandFR", EstimatorClass, 7, {"random_state": 9, "n_estimators": 5}) == {
        "n_estimators": 5, "random_state": 9, "n_jobs": -1,
    }
    assert BuildModelKwargs("DummyR", LoadEstimatorClass("DummyR"), 7, None) == {"strategy": "median"}
    with pytest.raises(KeyError):
        LoadEstimatorClass("Unknown")

def test_every_null_strategy_scaling_method_and_metric_has_a_guide_entry():
    assert set(NULL_STRATEGIES) == set(_NULL_GUIDE)
    assert set(SCALING_METHODS) == set(_SCALING_GUIDE) == set(_ScalerClasses())
    Plain = {"Label", "AlgorithmName", "SampleName"}
    for Columns in PERFORMANCE_COLUMNS.values():
        assert set(Columns) - Plain <= set(_METRIC_GUIDE)

def test_guides_in_both_languages():
    for Language in ("ENG", "TR", "eng"):
        assert len(koco.NullStrategyGuide(Language)) == len(NULL_STRATEGIES)
        assert len(koco.ScalingMethodGuide(Language)) == len(SCALING_METHODS)
        assert not koco.MetricGuide("All", Language).isna().any().any()
    assert "Strateji" in koco.NullStrategyGuide("TR").columns
    assert set(koco.MetricGuide("Regression")["Metric"]).isdisjoint({"AUC", "Gini", "KS"})
    for Guide in (koco.NullStrategyGuide, koco.ScalingMethodGuide, koco.MetricGuide):
        with pytest.raises(ValueError, match="Language"):
            Guide(Language="DE")
    with pytest.raises(ValueError):
        koco.MetricGuide("Other")

def test_guides_print(capsys):
    koco.ScalingMethodGuide(Print=True)
    koco.NullStrategyGuide(Print=True)
    koco.MetricGuide("Binary", Print=True)
    Output = capsys.readouterr().out
    assert "RobustScaler" in Output and "KeepNaN" in Output and "Gini" in Output

def test_model_registry_guide():
    Guide = koco.ModelRegistryGuide()
    assert len(Guide) == len(MODEL_SPECS)
    assert set(koco.ModelRegistryGuide("Binary")["ModelType"]) == {"Binary"}
    assert Guide.set_index("Key").loc["RidgeR", "AcceptsNaN"] == "No"
    with pytest.raises(ValueError, match="ModelType"):
        koco.ModelRegistryGuide("Other")
