import koco

def test_version_and_public_names_are_unique():
    assert koco.__version__ == "0.1.0"
    assert len(koco.__all__) == len(set(koco.__all__))
    for Name in koco.__all__:
        assert callable(getattr(koco, Name)), Name

def test_renamed_functions():
    for Name in ("ApplyOutlierPlan", "ApplyScalingPlan", "FitWOEPlan", "ApplyWOEPlan", "BackwardSelection", "ExportJson", "ImportJson", "ExportModelArtifacts", "ExportModelBundle", "ImportModelBundle", "ApplyModelBundle", "FitScalingPlan", "ScalingMethodGuide", "MetricGuide", "DrawChart"):
        assert Name in koco.__all__, Name
    for OldName in ("ExportPlan", "ImportPlan", "ScalingPlan", "ApplyOutlierManipulation", "ApplyScaling"):
        assert not hasattr(koco, OldName)

def test_every_public_function_has_a_docstring_with_the_standard_sections():
    for Name in koco.__all__:
        Doc = getattr(koco, Name).__doc__
        assert Doc and "Returns:" in Doc and "Raises:" in Doc, Name
        assert "Output:" not in Doc and "Args:" not in Doc, Name

def test_feature_selection_module_name():
    assert hasattr(koco, "feature_selection") and not hasattr(koco, "featureSelection")
