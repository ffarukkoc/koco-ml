from __future__ import annotations

import inspect
import time
import warnings
from collections.abc import Callable
from datetime import date, datetime

import numpy as np
import pandas as pd

from ._helpers import DEFAULT_RANDOM_STATE, _Alert, _Done, _Step, _ValidateTargetAndFeatures
from .metrics import PERFORMANCE_COLUMNS, ClassificationMetrics, RegressionMetrics
from .registry import MODEL_SPECS, REGISTRIES, BuildModelKwargs, LoadEstimatorClass
from .scoring import ApplyModel, PositiveProbability

__all__ = ["CheckModellingInputs", "TrainTestSplit", "RegressionModelling", "ClassificationModelling"]

#********* Modelling Functions *********#

_SPLIT_KEYS = {
    "Regression": ["TestSize", "RandomState", "Shuffle", "StratifyBins"],
    "Binary"    : ["TestSize", "RandomState", "Shuffle", "StratifyClasses"],
}
_LOG_COLUMNS = ["Label", "AlgorithmName", "SampleName", "Stage", "Severity", "Message", "SystemDate"]
_SEPARATED_TRAIN_AUC = 0.99

def _LogRow(
    Label: str,
    AlgorithmName: str,
    SampleName: str,
    Stage: str,
    Severity: str,
    Message: str,
) -> dict:
    """
    Build one row of the modelling Log.

    Arg:
       Label: Run label.
       AlgorithmName: Algorithm key, or "-" when the row is not about one algorithm.
       SampleName: Train / Test, or "-" when the row is not about one sample.
       Stage: Validation, Fit, Threshold, Predict or Metric.
       Severity: Error, Warning or Info.
       Message: Description of the event.

    Returns:
       Dict keyed by the Log columns, stamped with the current time as SystemDate.

    Raises:
       None.
    """
    return {
        "Label": Label, "AlgorithmName": AlgorithmName, "SampleName": SampleName,
        "Stage": Stage, "Severity": Severity, "Message": Message, "SystemDate": datetime.now(),
    }

def _RunCaptured(
    Func: Callable,
    *Args,
) -> tuple:
    """
    Call Func and collect the warnings it raises instead of printing them.

    Arg:
       Func: Function to call.
       Args: Positional arguments passed to Func.

    Returns:
       Tuple of (Result, Messages); Messages are unique "Category: message" strings.

    Raises:
       Any exception raised by Func.
    """
    with warnings.catch_warnings(record=True) as Caught:
        warnings.simplefilter("always")
        Result = Func(*Args)
    Messages = list(dict.fromkeys(f"{Item.category.__name__}: {Item.message}" for Item in Caught))
    return Result, Messages

def _YoudenThreshold(
    YTrue: pd.Series,
    YProb: np.ndarray,
) -> float:
    """
    Find the probability cut-off maximising TPR - FPR (Youden's J).

    Arg:
       YTrue: Actual 0/1 values.
       YProb: Predicted probability of class 1.

    Returns:
       Cut-off between 0 and 1; roc_curve's first threshold is inf, hence the cap at 1.

    Raises:
       None.
    """
    from sklearn.metrics import roc_curve

    FPR, TPR, Thresholds = roc_curve(YTrue, YProb)
    return float(min(Thresholds[np.argmax(TPR - FPR)], 1.0))

def _ClassWeightParams(
    EstimatorClass: type,
    ClassWeight: str | dict,
    YTrain: pd.Series,
) -> dict | None:
    """
    Translate ClassWeight into the parameter of each library.

    Arg:
       EstimatorClass: Classifier class.
       ClassWeight: "balanced" or a dict such as {0: 1, 1: 5}.
       YTrain: Train target used to compute balanced weights.

    Returns:
       Dict with scale_pos_weight (XGBoost), class_weights (CatBoost) or class_weight; None if the
       algorithm has no equivalent.

    Raises:
       None.
    """
    if ClassWeight == "balanced":
        Counts = YTrain.value_counts()
        Weights = {Cls: len(YTrain) / (2 * Counts[Cls]) for Cls in (0, 1)}
    else:
        Weights = ClassWeight

    if EstimatorClass.__name__ == "XGBClassifier":
        return {"scale_pos_weight": Weights[1] / Weights[0]}
    if EstimatorClass.__name__ == "CatBoostClassifier":
        return {"class_weights": [Weights[0], Weights[1]]}
    if "class_weight" in inspect.signature(EstimatorClass).parameters:
        return {"class_weight": ClassWeight}
    return None

def _ValidateModellingInputs(
    Table: pd.DataFrame,
    Models: str | list[str],
    Target: str,
    Features: list[str] | None,
    Label: str | None,
    Split: dict | bool | None,
    ModelParams: dict[str, dict] | None,
    RandomState: int,
    ModelType: str,
    ClassWeight: str | dict | None,
    Threshold: str | float,
) -> tuple[list[str], pd.DataFrame, dict]:
    """
    Validate and normalise the modelling inputs; the single place where input defaults are applied.

    Arg:
       Same as CheckModellingInputs.

    Returns:
       Tuple of (RunnableModels, Log, Inputs).
         Inputs: Normalised values with the keys Models (list), Features (list), SplitFlag (bool),
            SplitParams (dict) and ModelParams (dict).

    Raises:
       Same as CheckModellingInputs.
    """
    LogLabel = Label if Label is not None else "-"
    Log = []

    # ---- Fatal checks ----
    if ModelType not in REGISTRIES:
        raise ValueError(f"ModelType must be one of {list(REGISTRIES)}.")
    Features = _ValidateTargetAndFeatures(Table, Target, Features, ModelType)

    if isinstance(Models, str):
        Models = [Models]
    if not isinstance(Models, list) or not Models or not all(isinstance(Key, str) for Key in Models):
        raise TypeError("Models must be a model key or a non-empty list of model keys.")

    if Label is not None and not isinstance(Label, str):
        raise TypeError("Label must be a string.")
    if not isinstance(RandomState, int) or isinstance(RandomState, bool):
        raise TypeError("RandomState must be an integer.")

    if (Split is None) or (Split is True):
        SplitFlag, SplitParams = True, {}
    elif Split is False:
        SplitFlag, SplitParams = False, {}
    elif isinstance(Split, dict):
        SplitFlag, SplitParams = True, Split
    else:
        raise TypeError("Split must be False, True, None or a dict of split parameters.")
    UnknownSplitKeys = [Key for Key in SplitParams if Key not in _SPLIT_KEYS[ModelType]]
    if UnknownSplitKeys:
        raise ValueError(f"Unknown Split parameters: {UnknownSplitKeys}. Must be one of {_SPLIT_KEYS[ModelType]}.")

    ModelParams = {} if ModelParams is None else ModelParams
    if not isinstance(ModelParams, dict) or not all(isinstance(Params, dict) for Params in ModelParams.values()):
        raise TypeError("ModelParams must be a dict such as {'RandFR': {'n_estimators': 500}}.")

    if ModelType == "Binary":
        IsWeightDict = isinstance(ClassWeight, dict) and set(ClassWeight) == {0, 1} and all(
            isinstance(Weight, (int, float)) and not isinstance(Weight, bool) and Weight > 0 for Weight in ClassWeight.values()
        )
        if ClassWeight is not None and ClassWeight != "balanced" and not IsWeightDict:
            raise ValueError("ClassWeight must be None, 'balanced' or a dict such as {0: 1, 1: 5}.")
        IsThresholdNumber = isinstance(Threshold, (int, float)) and not isinstance(Threshold, bool)
        if Threshold != "auto" and not (IsThresholdNumber and 0 < Threshold < 1):
            raise ValueError("Threshold must be 'auto' or a number between 0 and 1.")

    # ---- Non-fatal checks, written to the log ----
    TargetNullCount = int(Table[Target].isna().sum())
    if TargetNullCount:
        Log.append(_LogRow(LogLabel, "-", "-", "Validation", "Warning",
                           f"{TargetNullCount} rows with null Target are excluded from modelling."))

    ModellingRows = Table.loc[Table[Target].notna(), Features]
    NullFeatures = ModellingRows.columns[ModellingRows.isna().any()].tolist()
    ConstantFeatures = ModellingRows.columns[ModellingRows.nunique(dropna=True) <= 1].tolist()
    if ConstantFeatures:
        Log.append(_LogRow(LogLabel, "-", "-", "Validation", "Warning",
                           f"Constant features carry no information: {ConstantFeatures}"))

    UnusedParams = [Key for Key in ModelParams if Key not in Models]
    if UnusedParams:
        Log.append(_LogRow(LogLabel, "-", "-", "Validation", "Warning",
                           f"ModelParams given for algorithms not in Models are ignored: {UnusedParams}"))

    RunnableModels = []
    for Key in dict.fromkeys(Models):
        if Models.count(Key) > 1:
            Log.append(_LogRow(LogLabel, Key, "-", "Validation", "Warning",
                               "Algorithm is listed more than once in Models; it is run only once."))
        if Key not in REGISTRIES[ModelType]:
            if Key in MODEL_SPECS:
                Message = f"{MODEL_SPECS[Key].ModelType} algorithm cannot be used for a {ModelType} model type; skipped."
            else:
                Message = f"Unknown algorithm key; skipped. Must be one of {list(REGISTRIES[ModelType])}."
            Log.append(_LogRow(LogLabel, Key, "-", "Validation", "Error", Message))
            continue
        try:
            EstimatorClass = LoadEstimatorClass(Key)
        except (ImportError, AttributeError) as Err:
            Log.append(_LogRow(LogLabel, Key, "-", "Validation", "Error",
                               f"Algorithm could not be loaded; skipped. {type(Err).__name__}: {Err}"))
            continue
        Spec = MODEL_SPECS[Key]
        if not Spec.NaNTolerant and NullFeatures:
            Log.append(_LogRow(LogLabel, Key, "-", "Validation", "Error",
                               f"Algorithm does not accept NaN values; skipped. Null values found in: {NullFeatures}"))
            continue

        Parameters = inspect.signature(EstimatorClass).parameters
        AcceptsAnyKwargs = any(Param.kind is inspect.Parameter.VAR_KEYWORD for Param in Parameters.values())
        UnlistedParams = [Name for Name in ModelParams.get(Key, {}) if Name not in Parameters]
        if UnlistedParams and not AcceptsAnyKwargs:
            Log.append(_LogRow(LogLabel, Key, "-", "Validation", "Error",
                               f"Unknown parameters for {EstimatorClass.__name__}; skipped: {UnlistedParams}"))
            continue
        if UnlistedParams:
            # **kwargs estimators accept any name silently, so a typo would otherwise go unnoticed.
            try:
                KnownParams = set(EstimatorClass().get_params())
            except Exception:
                KnownParams = set(Parameters)
            UnknownParams = [Name for Name in UnlistedParams if Name not in KnownParams]
            if UnknownParams:
                Log.append(_LogRow(LogLabel, Key, "-", "Validation", "Warning",
                                   f"Parameters not recognised by {EstimatorClass.__name__}.get_params() and passed as **kwargs; check for typos: {UnknownParams}"))

        if Spec.ScalingSensitive:
            Log.append(_LogRow(LogLabel, Key, "-", "Validation", "Warning",
                               "Scaling-sensitive algorithm; features are assumed to be already scaled."))
        RunnableModels.append(Key)

    Inputs = {
        "Models"     : Models,
        "Features"   : Features,
        "SplitFlag"  : SplitFlag,
        "SplitParams": SplitParams,
        "ModelParams": ModelParams,
    }
    return RunnableModels, pd.DataFrame(Log, columns=_LOG_COLUMNS), Inputs

def CheckModellingInputs(
    Table: pd.DataFrame,
    Models: str | list[str],
    Target: str,
    Features: list[str] | None = None,
    Label: str | None = None,
    Split: dict | bool | None = None,
    ModelParams: dict[str, dict] | None = None,
    RandomState: int = DEFAULT_RANDOM_STATE,
    ModelType: str = "Regression",
    ClassWeight: str | dict | None = None,
    Threshold: str | float = "auto",
) -> tuple[list[str], pd.DataFrame]:
    """
    Validate the inputs of RegressionModelling / ClassificationModelling and decide which
    algorithms can be run.

    Structural problems stop the process with an exception. Problems that only affect a single
    algorithm are written to the log and that algorithm is skipped, so the rest can still run.

    Arg:
       Table: Modelling table containing the Target and Features columns.
       Models: One or more keys of the registry that belongs to ModelType.
       Target: Numeric target column name; for Binary it must contain only 0 and 1.
       Features: Feature column names. If None, every column except Target is used.
       Label: Run label written to the log.
       Split: False uses the full Table as Train; a dict provides split parameters. None enables
          splitting with defaults. Allowed keys: TestSize, RandomState, Shuffle and StratifyBins
          (Regression) or StratifyClasses (Binary).
       ModelParams: Algorithm specific parameters, e.g. {"RandFR": {"n_estimators": 500}}.
       RandomState: Seed applied to every algorithm that supports one.
       ModelType: "Regression" or "Binary".
       ClassWeight: Binary only. None, "balanced" or a dict such as {0: 1, 1: 5}.
       Threshold: Binary only. "auto" or a number between 0 and 1.

    Returns:
       Tuple of (RunnableModels, Log).
         RunnableModels: Model keys that passed validation, in the requested order.
         Log: DataFrame with columns Label, AlgorithmName, SampleName, Stage, Severity,
            Message, SystemDate. Severity is Error for skipped algorithms and Warning otherwise.

    Raises:
       TypeError: If an input has the wrong type.
       ValueError: If ModelType is unknown, Table is empty, a column is missing, Target or Features are
          not numeric, contain inf values, Target is also a feature, a Binary Target is not 0/1,
          or Split, ClassWeight or Threshold are invalid.
    """
    RunnableModels, Log, _ = _ValidateModellingInputs(
        Table, Models, Target, Features, Label, Split, ModelParams, RandomState, ModelType, ClassWeight, Threshold,
    )
    return RunnableModels, Log

def TrainTestSplit(
    Table: pd.DataFrame,
    Target: str,
    TestSize: float = 0.3,
    RandomState: int = DEFAULT_RANDOM_STATE,
    Shuffle: bool = True,
    StratifyBins: int | None = None,
    StratifyClasses: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a table into Train and Test samples.

    Arg:
       Table: Source table to split.
       Target: Target column; only used when StratifyBins or StratifyClasses is given.
       TestSize: Share of rows assigned to Test, a float between 0 and 1.
       RandomState: Seed controlling the shuffle.
       Shuffle: If False, the last TestSize share of rows (in current order) becomes Test,
          which suits time-ordered data.
       StratifyBins: Continuous Target only. Number of Target quantile bins used to stratify the
          split so both samples share a similar Target distribution. Requires Shuffle=True.
       StratifyClasses: Categorical Target only. If True, both samples keep the same class
          ratio. Requires Shuffle=True.

    Returns:
       Tuple of (Train, Test) DataFrames keeping the original index.

    Raises:
       ValueError: If Target is missing, TestSize is out of range, or the stratify options are
          invalid.
    """
    from sklearn.model_selection import train_test_split

    if Target not in Table.columns:
        raise ValueError(f"Target column not found in Table: {Target}")
    if not isinstance(TestSize, float) or not 0 < TestSize < 1:
        raise ValueError("TestSize must be a float between 0 and 1.")
    if StratifyBins is not None and StratifyClasses:
        raise ValueError("Use either StratifyBins or StratifyClasses, not both.")
    if StratifyBins is not None:
        if not Shuffle:
            raise ValueError("StratifyBins requires Shuffle=True.")
        if not isinstance(StratifyBins, int) or isinstance(StratifyBins, bool) or StratifyBins < 2:
            raise ValueError("StratifyBins must be an integer >= 2.")
    if StratifyClasses and not Shuffle:
        raise ValueError("StratifyClasses requires Shuffle=True.")

    Stratify = None
    if StratifyBins is not None:
        Stratify = pd.qcut(Table[Target], q=StratifyBins, labels=False, duplicates="drop")
    elif StratifyClasses:
        Stratify = Table[Target]

    Train, Test = train_test_split(
        Table, test_size=TestSize, random_state=RandomState, shuffle=Shuffle, stratify=Stratify,
    )
    return Train, Test

def _RunModelling(
    ModelType: str,
    Table: pd.DataFrame,
    Models: str | list[str],
    Target: str,
    Features: list[str] | None,
    Label: str | None,
    Split: dict | bool | None,
    ModelParams: dict[str, dict] | None,
    RandomState: int,
    ClassWeight: str | dict | None = None,
    Threshold: str | float = "auto",
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Shared engine of RegressionModelling and ClassificationModelling.

    Arg:
       ModelType: "Regression" or "Binary"; the remaining arguments are documented in the wrappers.

    Returns:
       Tuple of (ModelArtifacts, Performance, Log, TrainData).

    Raises:
       TypeError / ValueError: If the inputs fail the fatal checks of CheckModellingInputs or a
          Binary Train sample contains a single class.
    """
    from sklearn.metrics import roc_auc_score

    Label = Label if Label is not None else f"{date.today().strftime('%y%m%d')}_Run"
    RunnableModels, ValidationLog, Inputs = _ValidateModellingInputs(
        Table, Models, Target, Features, Label, Split, ModelParams, RandomState, ModelType, ClassWeight, Threshold,
    )
    Log = ValidationLog.to_dict("records")

    Features = Inputs["Features"]
    SplitFlag, SplitParams = Inputs["SplitFlag"], Inputs["SplitParams"]
    ModelParams = Inputs["ModelParams"]

    ModelTable = Table.loc[Table[Target].notna(), Features + [Target]].copy()
    if SplitFlag:
        SplitDefaults = {"RandomState": RandomState}
        if ModelType == "Binary":
            SplitDefaults["StratifyClasses"] = SplitParams.get("Shuffle", True)
        Train, Test = TrainTestSplit(ModelTable, Target, **{**SplitDefaults, **SplitParams})
        Samples = {"Train": Train, "Test": Test}
    else:
        Samples = {"Train": ModelTable}
    TrainX, TrainY = Samples["Train"][Features], Samples["Train"][Target]

    if ModelType == "Regression":
        for SampleName, Sample in Samples.items():
            ZeroCount = int((Sample[Target] == 0).sum())
            if ZeroCount:
                Log.append(_LogRow(Label, "-", SampleName, "Metric", "Warning",
                                   f"{ZeroCount} rows with zero Target are excluded from MAPE."))
    else:
        if TrainY.nunique() < 2:
            raise ValueError("Train sample contains a single class; use a stratified split or more data.")
        for SampleName, Sample in Samples.items():
            if Sample[Target].nunique() < 2:
                Log.append(_LogRow(Label, "-", SampleName, "Metric", "Warning",
                                   "Sample contains a single class; AUC, Gini and KS are not available."))
        if ClassWeight is not None:
            Log.append(_LogRow(Label, "-", "Train", "Fit", "Info",
                               "Class weighting shifts predicted probabilities; MeanProb will not match TargetRate."))

    ModelArtifacts, PerformanceRows = {}, []
    for Key in RunnableModels:
        EstimatorClass = LoadEstimatorClass(Key)
        UserParams = dict(ModelParams.get(Key, {}))
        if ClassWeight is not None:
            WeightParams = _ClassWeightParams(EstimatorClass, ClassWeight, TrainY)
            if WeightParams is None:
                Log.append(_LogRow(Label, Key, "Train", "Fit", "Warning",
                                   "Algorithm does not support class weights; trained without weighting."))
            else:
                UserParams = {**WeightParams, **UserParams}
        Kwargs = BuildModelKwargs(Key, EstimatorClass, RandomState, UserParams)

        try:
            StartTime = time.perf_counter()
            Estimator, Messages = _RunCaptured(EstimatorClass(**Kwargs).fit, TrainX, TrainY)
            FitTime = round(time.perf_counter() - StartTime, 2)
        except Exception as Err:
            Log.append(_LogRow(Label, Key, "Train", "Fit", "Error", f"Fit failed; skipped. {type(Err).__name__}: {Err}"))
            _Alert(f"{Key:<12}: fit failed")
            continue
        Log.extend(_LogRow(Label, Key, "Train", "Fit", "Warning", Message) for Message in Messages)

        ModelThreshold = None
        if ModelType == "Binary" and Threshold == "auto":
            try:
                TrainProb, _ = _RunCaptured(PositiveProbability, Estimator, TrainX)
            except Exception as Err:
                Log.append(_LogRow(Label, Key, "Train", "Threshold", "Error",
                                   f"Threshold could not be calculated; skipped. {type(Err).__name__}: {Err}"))
                continue
            ModelThreshold = _YoudenThreshold(TrainY, TrainProb)
            if roc_auc_score(TrainY, TrainProb) >= _SEPARATED_TRAIN_AUC:
                Log.append(_LogRow(Label, Key, "Train", "Threshold", "Warning",
                                   f"Train probabilities are almost perfectly separated (AUC >= {_SEPARATED_TRAIN_AUC}); "
                                   "the auto threshold may not transfer to other samples."))
        elif ModelType == "Binary":
            ModelThreshold = float(Threshold)

        ModelArtifacts[Key] = {
            "Model"    : Estimator,
            "ModelType": ModelType,
            "Algorithm": EstimatorClass.__name__,
            "Features" : list(Features),
            "Target"   : Target,
            "Params"   : Kwargs,
            "Threshold": ModelThreshold,
            "Label"    : Label,
            "TrainDate": datetime.now(),
            "TrainSize": len(Samples["Train"]),
        }

        for SampleName, Sample in Samples.items():
            Stage = "Predict"
            try:
                Predictions, Messages = _RunCaptured(ApplyModel, Sample, ModelArtifacts[Key])
                Stage = "Metric"
                if ModelType == "Regression":
                    Metrics = RegressionMetrics(Sample[Target], Predictions)
                else:
                    Metrics = ClassificationMetrics(Sample[Target], Predictions["Probability"], ModelThreshold)
            except Exception as Err:
                Log.append(_LogRow(Label, Key, SampleName, Stage, "Error", f"{type(Err).__name__}: {Err}"))
                continue
            Log.extend(_LogRow(Label, Key, SampleName, "Predict", "Warning", Message) for Message in Messages)
            PerformanceRows.append({"Label": Label, "AlgorithmName": Key, "SampleName": SampleName, **Metrics, "FitTimeSec": FitTime})

        _Done(f"{Key:<12}: fitted in {FitTime:.2f}s")

    Performance = pd.DataFrame(PerformanceRows, columns=PERFORMANCE_COLUMNS[ModelType])
    Log = pd.DataFrame(Log, columns=_LOG_COLUMNS)

    RequestedCount = len(dict.fromkeys(Inputs["Models"]))
    _Step(
        f"Label             : {Label}\n"
        f"  Requested models  : {RequestedCount}\n"
        f"  Fitted models     : {len(ModelArtifacts)}\n"
        f"  Samples           : {', '.join(f'{Name}={len(Sample)}' for Name, Sample in Samples.items())}\n"
        f"  Errors / Warnings : {int((Log['Severity'] == 'Error').sum())} / {int((Log['Severity'] == 'Warning').sum())}"
    )

    return ModelArtifacts, Performance, Log, Samples["Train"]

def RegressionModelling(
    Table: pd.DataFrame,
    Models: str | list[str],
    Target: str,
    Features: list[str] | None = None,
    Label: str | None = None,
    Split: dict | bool | None = None,
    ModelParams: dict[str, dict] | None = None,
    RandomState: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train one or more regression algorithms and report their performance on each sample.

    Non-fatal problems (unknown algorithm, missing package, NaN intolerance, invalid parameters,
    fit / predict failures, library warnings) are written to the log and the run continues with
    the remaining algorithms. Use ModelRegistryGuide("Regression") to see the available algorithms.

    Arg:
       Table: Cleaned modelling table; it is not modified.
       Models: One or more Regression registry keys (ending with "R"), e.g. ["RidgeR", "RandFR"].
       Target: Continuous target column name. Rows with a null Target are excluded.
       Features: Feature column names. If None, every column except Target is used.
       Label: Tag used to compare different runs. If None, "<yymmdd>_Run" is used.
       Split: False uses the full Table as Train; a dict splits Table into Train / Test with
          TrainTestSplit. None enables splitting with defaults. Dict keys: TestSize, RandomState,
          Shuffle, StratifyBins. e.g. {"TestSize": 0.25, "StratifyBins": 10}
       ModelParams: Algorithm specific parameters overriding registry defaults,
          e.g. {"RandFR": {"n_estimators": 500}, "RidgeR": {"alpha": 10}}.
       RandomState: Seed applied to every algorithm that supports one and to the split,
          unless overridden in ModelParams or Split.

    Returns:
       Tuple of (ModelArtifacts, Performance, Log, TrainData).
         ModelArtifacts: Dict keyed by algorithm; each entry holds Model, ModelType, Algorithm,
            Features, Target, Params, Threshold (None), Label, TrainDate and TrainSize.
            Pass an entry to ApplyModel.
         Performance: One row per algorithm and sample with columns Label, AlgorithmName,
            SampleName, N, RMSE, MAE, MedAE, MAPE, WMAPE, R2, Bias, FitTimeSec.
         Log: Errors and warnings with columns Label, AlgorithmName, SampleName, Stage,
            Severity, Message, SystemDate.
         TrainData: Exact rows (original index, Features + Target) the models were fitted on,
            so the run can be reproduced regardless of the incoming row order.

    Raises:
       TypeError / ValueError: If the inputs fail the fatal checks of CheckModellingInputs.
    """
    return _RunModelling("Regression", Table, Models, Target, Features, Label, Split, ModelParams, RandomState)

def ClassificationModelling(
    Table: pd.DataFrame,
    Models: str | list[str],
    Target: str,
    Features: list[str] | None = None,
    Label: str | None = None,
    Split: dict | bool | None = None,
    ModelParams: dict[str, dict] | None = None,
    RandomState: int = DEFAULT_RANDOM_STATE,
    ClassWeight: str | dict | None = None,
    Threshold: str | float = "auto",
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train one or more binary classification algorithms and report their performance on each sample.

    Models predict the probability of class 1. The 0/1 prediction uses a Threshold fitted on
    Train only and reused for every other sample, so no information leaks from Test.
    Non-fatal problems are written to the log and the run continues with the remaining
    algorithms. Use ModelRegistryGuide("Binary") to see the available algorithms.

    Arg:
       Table: Cleaned modelling table; it is not modified.
       Models: One or more Binary registry keys (ending with "C"), e.g. ["LogisticC", "RandFC"].
       Target: Binary target column containing only 0 and 1. Rows with a null Target are excluded.
       Features: Feature column names. If None, every column except Target is used.
       Label: Tag used to compare different runs. If None, "<yymmdd>_Run" is used.
       Split: False uses the full Table as Train; a dict splits Table into Train / Test with
          TrainTestSplit, stratified on Target by default. None enables splitting with defaults.
          Dict keys: TestSize, RandomState, Shuffle, StratifyClasses. e.g. {"TestSize": 0.25}
       ModelParams: Algorithm specific parameters overriding registry defaults and ClassWeight,
          e.g. {"RandFC": {"n_estimators": 500}, "LogisticC": {"C": 0.1}}.
       RandomState: Seed applied to every algorithm that supports one and to the split,
          unless overridden in ModelParams or Split.
       ClassWeight: Imbalance handling applied to every algorithm that supports it. None (no
          weighting), "balanced" (weights inversely proportional to class frequencies) or a dict
          such as {0: 1, 1: 5}. It is translated to class_weight, scale_pos_weight (XGBoost) or
          class_weights (CatBoost); other algorithms are trained unweighted with a warning.
       Threshold: "auto" picks, per algorithm, the cut-off maximising TPR - FPR (Youden's J) on
          Train; a number between 0 and 1 is used as a fixed cut-off for all algorithms.

    Returns:
       Tuple of (ModelArtifacts, Performance, Log, TrainData).
         ModelArtifacts: Dict keyed by algorithm; each entry holds Model, ModelType, Algorithm,
            Features, Target, Params, Threshold, Label, TrainDate and TrainSize. Pass an entry to
            ApplyModel, optionally with a different Threshold.
         Performance: One row per algorithm and sample with Label, AlgorithmName, SampleName,
            the ClassificationMetrics outputs and FitTimeSec.
         Log: Errors, warnings and info with columns Label, AlgorithmName, SampleName, Stage,
            Severity, Message, SystemDate.
         TrainData: Exact rows (original index, Features + Target) the models were fitted on,
            so the run can be reproduced regardless of the incoming row order.

    Raises:
       TypeError / ValueError: If the inputs fail the fatal checks of CheckModellingInputs or the
          Train sample contains a single class.
    """
    return _RunModelling(
        "Binary", Table, Models, Target, Features, Label, Split, ModelParams, RandomState, ClassWeight, Threshold,
    )
