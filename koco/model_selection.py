from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import DEFAULT_RANDOM_STATE, _Alert, _CheckColumns, _Done, _Step
from .modelling import _SEPARATED_TRAIN_AUC

__all__ = ["SelectBestModel", "EvaluateOnHoldout", "BootstrapCompare"]

#********* Model Selection Functions *********#

_MODEL_KEY = ["Label", "AlgorithmName"]
_HIGHER_IS_BETTER = {"R2", "Gini", "AUC", "KS"}
# Per model type: Gain (lower is better) and Beat (higher is better) are compared with the baseline; Primary decides the
# tie band; Gap is the Train - Selection difference checked for overfitting; Memorize flags near-perfect Train fits.
_SELECTION = {
    "Regression": {"Baseline": "DummyR", "Gain": "WMAPE", "Beat": "R2", "Primary": "WMAPE", "Gap": "R2",
                   "Memorize": "R2", "MemorizeLimit": 0.999, "Ranks": ["WMAPE", "RMSE", "MAE", "R2"]},
    "Binary"    : {"Baseline": "DummyC", "Gain": "LogLoss", "Beat": "Gini", "Primary": "Gini", "Gap": "Gini",
                   "Memorize": "AUC", "MemorizeLimit": _SEPARATED_TRAIN_AUC, "Ranks": ["Gini", "KS", "LogLoss", "Brier"]},
}
_DECIMALS = 4  # same as the Performance table
_PROGRESS_WORK = 50_000_000  # rows x candidates x resamples from which BootstrapCompare reports progress

def _DetectModelType(
    Performance: pd.DataFrame,
) -> str:
    """
    Validate a Performance table and detect its model type from the metric columns.

    Arg:
       Performance: Table returned by the modelling functions, possibly extended with other samples (e.g. OOT).

    Returns:
       "Regression" or "Binary".

    Raises:
       TypeError: If Performance is not a DataFrame.
       ValueError: If Performance is empty, lacks Label, AlgorithmName or SampleName, or holds
          Regression and Binary metrics together (or neither).
    """
    if not isinstance(Performance, pd.DataFrame):
        raise TypeError("Performance must be a DataFrame.")
    if Performance.empty:
        raise ValueError("Performance must not be empty.")
    _CheckColumns(Performance, _MODEL_KEY + ["SampleName"])
    IsBinary, IsRegression = "Gini" in Performance.columns, "RMSE" in Performance.columns
    if IsBinary == IsRegression:
        raise ValueError("Performance must hold either Binary metrics (Gini, ...) or Regression metrics (RMSE, ...).")
    return "Binary" if IsBinary else "Regression"

def _CheckNumber(
    Value,
    Name: str,
    Upper: float | None = None,
) -> None:
    """Raise ValueError unless Value is a finite number >= 0 (and < Upper when given)."""
    if isinstance(Value, bool) or not isinstance(Value, (int, float)) or not np.isfinite(Value) or Value < 0 or (Upper is not None and Value >= Upper):
        raise ValueError(f"{Name} must be a number >= 0" + (f" and < {Upper}." if Upper is not None else "."))

def _StepResult(
    Passed: pd.Series,
    Evaluated: pd.Series,
) -> pd.Series:
    """Convert boolean checks to "Pass" / "Fail"; rows that were not evaluated become "NA"."""
    return pd.Series(np.where(Evaluated, np.where(Passed, "Pass", "Fail"), "NA"), index=Passed.index)

def SelectBestModel(
    Performance: pd.DataFrame,
    SelectionSample: str = "Test",
    TrainSample: str = "Train",
    BaselineAlgorithm: str | None = None,
    MinImprovementOverBaseline: float = 0.10,
    MaxRelativeBias: float = 0.05,
    MaxFitGap: float = 0.15,
    MemorizationLimit: float | None = None,
    TieTolerance: float = 0.01,
) -> pd.DataFrame:
    """
    Choose the best model from a Performance table and show how every model was eliminated or ranked.

    A model is one Label and AlgorithmName pair. Only the TrainSample and SelectionSample rows are
    read, so other samples (e.g. OOT) stay an honest final check (see EvaluateOnHoldout).
    Steps: Data, Baseline, Bias (Regression only), Memorize and Overfit; see GUIDE.md for the rules.

    Arg:
       Performance: Table from RegressionModelling / ClassificationModelling, optionally with extra
          sample rows; it is not modified.
       SelectionSample: Sample used for the choice. It must differ from TrainSample.
       TrainSample: Sample the models were fitted on.
       BaselineAlgorithm: Registry key of the baseline model. None uses DummyR (Regression) or DummyC (Binary).
       MinImprovementOverBaseline: Required relative gain over the baseline, in [0, 1).
       MaxRelativeBias: Largest allowed |Bias| / mean|y| (Regression only).
       MaxFitGap: Largest allowed Train - SelectionSample difference of R2 (Regression) or Gini (Binary).
       MemorizationLimit: Train R2 (Regression) or AUC (Binary) from which a model counts as
          memorising. None uses the default of the model type.
       TieTolerance: Relative distance from the best primary metric (WMAPE or Gini) inside which models are tied.

    Returns:
       DataFrame with ranked models first, then eliminated models, then baselines. Columns: Label,
       AlgorithmName, one column per step ("Pass", "Fail" or "NA" when not evaluated), FitGap,
       RelativeBias (Regression only), the primary metric on both samples, AvgRank, TieBand,
       Result ("Eliminated", "Baseline" or the rank) and Recommended (True for one model).

    Raises:
       TypeError: If Performance is not a DataFrame.
       ValueError: If Performance is invalid, a sample is missing, a metric column is missing, or an
          argument is out of range.
    """
    ModelType = _DetectModelType(Performance)
    Config = _SELECTION[ModelType]
    if not isinstance(SelectionSample, str) or not isinstance(TrainSample, str) or SelectionSample == TrainSample:
        raise ValueError("SelectionSample and TrainSample must be different sample names.")
    _CheckNumber(MinImprovementOverBaseline, "MinImprovementOverBaseline", Upper=1)
    for Name, Value in [("MaxRelativeBias", MaxRelativeBias), ("MaxFitGap", MaxFitGap), ("TieTolerance", TieTolerance)]:
        _CheckNumber(Value, Name)
    if MemorizationLimit is not None:
        _CheckNumber(MemorizationLimit, "MemorizationLimit")
    BaselineAlgorithm = BaselineAlgorithm or Config["Baseline"]
    MemorizationLimit = Config["MemorizeLimit"] if MemorizationLimit is None else MemorizationLimit

    Samples = [TrainSample, SelectionSample]
    Needed = list(dict.fromkeys([Config[Key] for Key in ("Gain", "Beat", "Primary", "Gap", "Memorize")] + Config["Ranks"]
                                + (["MAE", "WMAPE", "Bias"] if ModelType == "Regression" else [])))
    _CheckColumns(Performance, Needed)
    Used = Performance.loc[Performance["SampleName"].isin(Samples), _MODEL_KEY + ["SampleName"] + Needed].copy()
    MissingSamples = [Sample for Sample in Samples if Sample not in set(Used["SampleName"])]
    if MissingSamples:
        raise ValueError(f"Samples not found in Performance: {MissingSamples}")
    Used[Needed] = Used[Needed].apply(pd.to_numeric, errors="coerce")

    Duplicated = Used.duplicated(_MODEL_KEY + ["SampleName"], keep=False)
    DuplicateModels = set(Used.loc[Duplicated, _MODEL_KEY].itertuples(index=False, name=None))
    Wide = Used.loc[~Duplicated].pivot(index=_MODEL_KEY, columns="SampleName", values=Needed)
    Wide = Wide.reindex(columns=pd.MultiIndex.from_product([Needed, Samples]))
    Wide.columns = [f"{Metric}_{Sample}" for Metric, Sample in Wide.columns]
    AllModels = pd.MultiIndex.from_frame(Used[_MODEL_KEY].drop_duplicates())
    Table = Wide.reindex(AllModels).reset_index()

    Sel = SelectionSample
    Columns = [f"{Metric}_{Sample}" for Metric in Needed for Sample in Samples]
    IsDuplicate = pd.Series([Key in DuplicateModels for Key in zip(Table["Label"], Table["AlgorithmName"])], index=Table.index)
    DataOK = ~IsDuplicate & np.isfinite(Table[Columns]).all(axis=1)
    if IsDuplicate.any():
        _Alert(f"Duplicate rows for the same Label, AlgorithmName and sample; eliminated: {sorted(DuplicateModels)}")
    if (~IsDuplicate & ~DataOK).any():
        _Alert(f"Missing or non-finite metrics; eliminated: {Table.loc[~IsDuplicate & ~DataOK, _MODEL_KEY].values.tolist()}")

    IsBaseline = Table["AlgorithmName"].eq(BaselineAlgorithm)
    Candidate = DataOK & ~IsBaseline

    Base = Table.loc[IsBaseline & DataOK].set_index("Label")
    BaseGain = Table["Label"].map(Base[f"{Config['Gain']}_{Sel}"])
    BaseBeat = Table["Label"].map(Base[f"{Config['Beat']}_{Sel}"])
    NoBaseline = sorted(set(Table.loc[Candidate, "Label"]) - set(Base.index))
    if NoBaseline:
        _Alert(f"No usable {BaselineAlgorithm} for Label(s) {NoBaseline}; Baseline step skipped (NA).")
    BaselinePass = (Table[f"{Config['Gain']}_{Sel}"] <= BaseGain * (1 - MinImprovementOverBaseline)) & (Table[f"{Config['Beat']}_{Sel}"] > BaseBeat)

    Steps = {"Data": _StepResult(DataOK, pd.Series(True, index=Table.index))}
    Steps["Baseline"] = _StepResult(BaselinePass, Candidate & BaseGain.notna())
    if ModelType == "Regression":
        MeanAbsY = Table[f"MAE_{Sel}"] / Table[f"WMAPE_{Sel}"].replace(0, np.nan)
        Table["RelativeBias"] = (Table[f"Bias_{Sel}"].abs() / MeanAbsY).round(_DECIMALS)
        Steps["Bias"] = _StepResult(Table["RelativeBias"] <= MaxRelativeBias, Candidate & Table["RelativeBias"].notna())
    Table["FitGap"] = (Table[f"{Config['Gap']}_{Samples[0]}"] - Table[f"{Config['Gap']}_{Sel}"]).round(_DECIMALS)
    Steps["Memorize"] = _StepResult(Table[f"{Config['Memorize']}_{Samples[0]}"] < MemorizationLimit, Candidate)
    Steps["Overfit"] = _StepResult(Table["FitGap"] <= MaxFitGap, Candidate)

    Eligible = Candidate & ~Steps["Baseline"].eq("Fail")
    if ModelType == "Regression":
        Eligible &= ~Steps["Bias"].eq("Fail")
    Stable = Steps["Memorize"].eq("Pass") & Steps["Overfit"].eq("Pass")
    if (Eligible & Stable).any():
        Chosen = Eligible & Stable
    elif (Eligible & Steps["Overfit"].eq("Pass")).any():
        Chosen = Eligible & Steps["Overfit"].eq("Pass")
        _Alert("No model passed both Memorize and Overfit; ranking the models that pass Overfit (check MemorizationLimit).")
    elif (Eligible & Steps["Memorize"].eq("Pass")).any():
        Chosen = Eligible & Steps["Memorize"].eq("Pass")
        _Alert("No stable model survived (every model overfits); ranking the models that do not memorise Train.")
    else:
        Chosen = pd.Series(False, index=Table.index)
        _Alert("No model survived the elimination steps; nothing is recommended.")

    Primary = f"{Config['Primary']}_{Sel}"
    Table["AvgRank"] = np.nan
    Table["TieBand"] = "NA"
    Table["Result"] = pd.Series(np.where(IsBaseline, "Baseline", "Eliminated"), index=Table.index, dtype=object)
    Table["Recommended"] = False
    Order = []
    if Chosen.any():
        Ranks = pd.DataFrame({
            Metric: Table.loc[Chosen, f"{Metric}_{Sel}"].rank(ascending=Metric not in _HIGHER_IS_BETTER) for Metric in Config["Ranks"]
        })
        Table.loc[Chosen, "AvgRank"] = Ranks.mean(axis=1).round(_DECIMALS)
        Values = Table.loc[Chosen, Primary]
        if Config["Primary"] in _HIGHER_IS_BETTER:
            InBand = Values >= Values.max() - TieTolerance * abs(Values.max())
        else:
            InBand = Values <= Values.min() + TieTolerance * abs(Values.min())
        Table.loc[Chosen, "TieBand"] = np.where(InBand, "Yes", "No")
        Tied = Table.loc[Chosen & Table["TieBand"].eq("Yes")].sort_values(["FitGap", "AvgRank"])
        Rest = Table.loc[Chosen & Table["TieBand"].eq("No")].sort_values(["AvgRank", Primary], ascending=[True, Config["Primary"] not in _HIGHER_IS_BETTER])
        Order = list(Tied.index) + list(Rest.index)
        Table.loc[Tied.index, "Result"] = 1
        for Position, Index in enumerate(Rest.index, start=len(Tied) + 1):
            Table.loc[Index, "Result"] = Position
        Table.loc[Tied.index[0], "Recommended"] = True

    for Name, Values in Steps.items():
        Table[Name] = Values.where(~IsBaseline, "NA")
    Remaining = [Index for Index in Table.index if Index not in Order]
    Remaining.sort(key=lambda Index: bool(IsBaseline[Index]))
    Columns = _MODEL_KEY + list(Steps) + ["FitGap"] + (["RelativeBias"] if ModelType == "Regression" else []) \
        + [f"{Config['Primary']}_{Samples[0]}", Primary, "AvgRank", "TieBand", "Result", "Recommended"]
    Ranked = Table.loc[Order + Remaining, Columns].reset_index(drop=True)
    Ranked["Result"] = Ranked["Result"].astype(object)

    _Step(
        f"Models      : {int((~IsBaseline).sum())} (+{int(IsBaseline.sum())} baseline)\n"
        f"  Eliminated  : {int((Ranked['Result'] == 'Eliminated').sum())}\n"
        f"  Ranked      : {len(Order)} (tied at rank 1: {int((Ranked['TieBand'] == 'Yes').sum())})"
    )
    if Chosen.any():
        Best = Ranked.iloc[0]
        _Done(f"Recommended: {Best['Label']} / {Best['AlgorithmName']} ({Primary} = {Best[Primary]:g})")
    return Ranked

def EvaluateOnHoldout(
    Performance: pd.DataFrame,
    Label: str,
    AlgorithmName: str,
    SelectionSample: str = "Test",
    HoldoutSample: str = "OOT",
    MaxDecay: float = 0.30,
    MaxRelativeBias: float = 0.05,
) -> pd.Series:
    """
    Check one chosen model on a holdout sample (e.g. OOT) after the selection is finished.

    Run it once for the model SelectBestModel recommended; do not use the result to pick another
    model, otherwise the holdout is no longer independent.

    Arg:
       Performance: Table with the SelectionSample and HoldoutSample rows; it is not modified.
       Label: Label of the model.
       AlgorithmName: Algorithm key of the model.
       SelectionSample: Sample the model was chosen on.
       HoldoutSample: Sample that was kept out of the selection.
       MaxDecay: Largest allowed relative worsening of the primary metric (WMAPE increase or Gini
          decrease) from SelectionSample to HoldoutSample.
       MaxRelativeBias: Largest allowed |Bias| / mean|y| on the HoldoutSample (Regression only).

    Returns:
       Series with Label, AlgorithmName, the primary metric on both samples, Decay, RelativeBias
       (Regression only) and Pass. A check with a zero denominator is NaN and skipped with an alert;
       Pass is False when metrics are missing or no check could be calculated.

    Raises:
       TypeError: If Performance is not a DataFrame.
       ValueError: If Performance is invalid, the samples are equal, an argument is out of range, or
          the model does not have exactly one row in each sample.
    """
    ModelType = _DetectModelType(Performance)
    Config = _SELECTION[ModelType]
    if SelectionSample == HoldoutSample:
        raise ValueError("SelectionSample and HoldoutSample must be different sample names.")
    _CheckNumber(MaxDecay, "MaxDecay")
    _CheckNumber(MaxRelativeBias, "MaxRelativeBias")

    Primary = Config["Primary"]
    Needed = list(dict.fromkeys([Primary] + (["MAE", "WMAPE", "Bias"] if ModelType == "Regression" else [])))
    _CheckColumns(Performance, Needed)
    Rows = Performance.loc[
        Performance["Label"].eq(Label) & Performance["AlgorithmName"].eq(AlgorithmName)
        & Performance["SampleName"].isin([SelectionSample, HoldoutSample])
    ].set_index("SampleName")
    if len(Rows) != 2 or not Rows.index.is_unique:
        raise ValueError(f"Model needs exactly one {SelectionSample} and one {HoldoutSample} row: {Label} / {AlgorithmName}")
    Rows = Rows[Needed].apply(pd.to_numeric, errors="coerce")

    Before, After = Rows.loc[SelectionSample, Primary], Rows.loc[HoldoutSample, Primary]
    Direction = 1 if Primary in _HIGHER_IS_BETTER else -1
    Decay = Direction * (Before - After) / abs(Before) if Before != 0 else np.nan
    Result = {"Label": Label, "AlgorithmName": AlgorithmName,
              f"{Primary}_{SelectionSample}": Before, f"{Primary}_{HoldoutSample}": After, "Decay": Decay}
    Missing = not np.isfinite([Before, After]).all()
    Limits = {"Decay": MaxDecay}
    if ModelType == "Regression":
        HoldoutRow = Rows.loc[HoldoutSample]
        MeanAbsY = HoldoutRow["MAE"] / HoldoutRow["WMAPE"] if HoldoutRow["WMAPE"] != 0 else np.nan
        Result["RelativeBias"] = abs(HoldoutRow["Bias"]) / MeanAbsY
        Missing = Missing or not np.isfinite(HoldoutRow[["MAE", "WMAPE", "Bias"]].to_numpy(dtype=float)).all()
        Limits["RelativeBias"] = MaxRelativeBias

    Skipped = [Key for Key in Limits if not np.isfinite(Result[Key])]
    Checked = [Result[Key] <= Limit for Key, Limit in Limits.items() if Key not in Skipped]
    if Missing:
        _Alert("Metrics are missing or not finite; Pass is False.")
    elif Skipped:
        _Alert(f"{Skipped} could not be calculated (zero denominator) and were skipped.")
    Passed = bool(Checked) and all(Checked) and not Missing
    Result = {Key: (round(float(Value), _DECIMALS) + 0.0 if isinstance(Value, (float, np.floating)) else Value) for Key, Value in Result.items()}
    Result["Pass"] = Passed
    return pd.Series(Result)

def _BootstrapMetric(
    Metric: str,
    Y: np.ndarray,
    P: np.ndarray,
) -> float:
    """Calculate one BootstrapCompare metric from raw values; NaN when the metric is undefined for the sample."""
    if Metric == "WMAPE":
        return np.abs(P - Y).sum() / np.abs(Y).sum()
    if Metric == "RMSE":
        return float(np.sqrt(np.mean((P - Y) ** 2)))
    if Metric == "MAE":
        return float(np.mean(np.abs(P - Y)))
    if Metric == "LogLoss":
        Clipped = np.clip(P, 1e-15, 1 - 1e-15)
        return float(-np.mean(Y * np.log(Clipped) + (1 - Y) * np.log(1 - Clipped)))
    if Y.min() == Y.max():
        return np.nan
    from sklearn.metrics import roc_auc_score

    AUC = roc_auc_score(Y, P)
    return float(AUC if Metric == "AUC" else 2 * AUC - 1)

def BootstrapCompare(
    Predictions: pd.DataFrame,
    Target: str,
    Candidates: dict[str, tuple[str, str]],
    ModelType: str,
    Metric: str | None = None,
    Reference: str | None = None,
    Resamples: int = 1000,
    ConfidenceLevel: float = 0.90,
    RandomState: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Test with a paired bootstrap whether candidate models really differ on the same rows.

    Rows are resampled with replacement and all models are scored on the same resampled rows. If
    the interval of the metric difference to the Reference contains 0, the candidate is tied with
    it. Use it on the SelectionSample rows only; see GUIDE.md for the limits of the test.

    Arg:
       Predictions: One row per sample row with the Target column and one prediction column per
          candidate (Regression: predicted value; Binary: probability of class 1); it is not modified.
       Target: Name of the Target column in Predictions.
       Candidates: Candidate column name -> (Label, AlgorithmName) with at least two entries, e.g.
          {"Fix9_RandFR": ("Fix 9 Feature", "RandFR"), "Fix12_ExtraTreesR": ("Fix 12 Feature", "ExtraTreesR")}.
       ModelType: "Regression" or "Binary".
       Metric: Regression: WMAPE (default), RMSE, MAE. Binary: Gini (default), AUC, LogLoss.
       Reference: Candidate key to compare against; None uses the best candidate on all rows.
       Resamples: Number of bootstrap samples, at least 100.
       ConfidenceLevel: Width of the interval, between 0 and 1 (exclusive).
       RandomState: Seed of the resampling.

    Returns:
       DataFrame, best metric first, with Candidate, Label, AlgorithmName, the metric on all rows,
       Delta (mean improvement over the Reference; positive is better), CILow, CIHigh,
       ProbBetter (share of resamples where the candidate beats the Reference) and Verdict
       ("Reference", "Tied", "Better" or "Worse").

    Raises:
       TypeError: If Predictions is not a DataFrame.
       ValueError: If an argument is invalid, a column is missing, the data contain nulls or
          invalid values (Binary Target not 0/1 with both classes, probabilities outside [0, 1],
          an all-zero Regression Target), or Reference is not a candidate.
    """
    if not isinstance(Predictions, pd.DataFrame):
        raise TypeError("Predictions must be a DataFrame.")
    if ModelType not in _SELECTION:
        raise ValueError('ModelType must be "Regression" or "Binary".')
    Allowed = {"Regression": ["WMAPE", "RMSE", "MAE"], "Binary": ["Gini", "AUC", "LogLoss"]}[ModelType]
    Metric = Metric or Allowed[0]
    if Metric not in Allowed:
        raise ValueError(f"Metric for {ModelType} must be one of {Allowed}.")
    if not isinstance(Candidates, dict) or len(Candidates) < 2 or not all(
        isinstance(Value, tuple) and len(Value) == 2 for Value in Candidates.values()
    ):
        raise ValueError("Candidates must map at least two column names to (Label, AlgorithmName) tuples.")
    if len(set(Candidates.values())) != len(Candidates):
        raise ValueError("Candidates must not contain the same (Label, AlgorithmName) twice.")
    if isinstance(Resamples, bool) or not isinstance(Resamples, int) or Resamples < 100:
        raise ValueError("Resamples must be an integer >= 100.")
    if not isinstance(ConfidenceLevel, (int, float)) or not 0 < ConfidenceLevel < 1:
        raise ValueError("ConfidenceLevel must be between 0 and 1.")
    Names = list(Candidates)
    _CheckColumns(Predictions, [Target] + Names)
    if Target in Names:
        raise ValueError("Target must not be a candidate column.")
    if Reference is not None and Reference not in Candidates:
        raise ValueError(f"Reference must be one of the candidates: {Names}")

    Data = Predictions[[Target] + Names]
    if Data.isna().any().any() or not np.isfinite(Data.to_numpy(dtype=float)).all():
        raise ValueError("Target and candidate columns must not contain nulls or infinite values.")
    Y = Data[Target].to_numpy(dtype=float)
    P = {Name: Data[Name].to_numpy(dtype=float) for Name in Names}
    if ModelType == "Binary":
        if not np.isin(Y, [0, 1]).all() or Y.min() == Y.max():
            raise ValueError("Binary Target must contain only 0 and 1, with both classes present.")
        if any(((Values < 0) | (Values > 1)).any() for Values in P.values()):
            raise ValueError("Binary candidate columns must contain probabilities between 0 and 1.")
    elif Metric == "WMAPE" and np.abs(Y).sum() == 0:
        raise ValueError("WMAPE is undefined when every Target value is 0.")

    Sign = 1.0 if Metric in _HIGHER_IS_BETTER else -1.0
    Full = {Name: _BootstrapMetric(Metric, Y, P[Name]) for Name in Names}
    if Reference is None:
        Reference = max(Names, key=lambda Name: Sign * Full[Name])

    Generator = np.random.default_rng(RandomState)
    Draws = np.empty((Resamples, len(Names)))
    _Step(f"Bootstrap   : {Resamples} resamples, {len(Names)} candidates, {len(Y)} rows ({Metric})")
    ReportEvery = max(Resamples // 4, 1) if len(Y) * len(Names) * Resamples >= _PROGRESS_WORK else None
    for Draw in range(Resamples):
        Index = Generator.integers(0, len(Y), len(Y))
        Draws[Draw] = [_BootstrapMetric(Metric, Y[Index], P[Name][Index]) for Name in Names]
        if ReportEvery and (Draw + 1) % ReportEvery == 0 and Draw + 1 < Resamples:
            _Info(f"Resampled {Draw + 1} / {Resamples}")
    Draws = Draws[np.isfinite(Draws).all(axis=1)]
    if len(Draws) < Resamples * 0.5:
        raise ValueError("Too many resamples had a single class; use more rows.")

    Alpha = (1 - ConfidenceLevel) / 2
    Rows = []
    for Position, Name in enumerate(Names):
        Delta = Sign * (Draws[:, Position] - Draws[:, Names.index(Reference)])
        Low, High = np.quantile(Delta, [Alpha, 1 - Alpha])
        Verdict = "Reference" if Name == Reference else "Tied" if Low <= 0 <= High else "Better" if Low > 0 else "Worse"
        Rows.append({
            "Candidate": Name, "Label": Candidates[Name][0], "AlgorithmName": Candidates[Name][1],
            Metric: round(Full[Name], _DECIMALS), "Delta": round(float(Delta.mean()), _DECIMALS) + 0.0,
            "CILow": round(float(Low), _DECIMALS) + 0.0, "CIHigh": round(float(High), _DECIMALS) + 0.0,
            "ProbBetter": round(float((Delta > 0).mean()), _DECIMALS), "Verdict": Verdict,
        })
    return pd.DataFrame(Rows).sort_values(Metric, ascending=Sign < 0, kind="mergesort").reset_index(drop=True)
