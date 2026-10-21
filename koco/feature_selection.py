from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import DEFAULT_RANDOM_STATE, _Alert, _CheckColumns, _Done, _Info, _Step, _ValidateTargetAndFeatures
from .registry import REGISTRIES, BuildModelKwargs, LoadEstimatorClass

__all__ = ["Correlation", "VIFReport", "SelectModelFeatures", "BackwardSelection"]

#********* Feature Selection Functions *********#

_ELIMINATION_METHODS = ("Combined", "OnlyNoise")

def Correlation(
    Table: pd.DataFrame,
    Target: str,
    Limit: float = 0.6,
    Method: str = "pearson",
    MethodT: str = "pearson",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Evaluate pairwise correlations and mark redundant features.

    Pairs are processed from the highest absolute correlation down to Limit; in each pair the
    feature with the weaker absolute correlation to Target is eliminated.

    Arg:
       Table: Source table containing only numeric feature columns and Target.
       Target: Target column used to compare correlated features.
       Limit: Absolute correlation threshold for elimination.
       Method: Correlation method for feature-to-feature comparison ("pearson", "spearman", "kendall").
       MethodT: Correlation method for feature-to-target comparison.

    Returns:
       Tuple of (CorrelationTable, RemainingColumns, EliminatedColumns).

    Raises:
       ValueError: If Target is missing, there are fewer than 2 feature columns, or a column is not numeric.
    """
    _CheckColumns(Table, [Target])
    VarList = [c for c in Table.columns if c != Target]

    if len(VarList) < 2:
        raise ValueError("Data must contain at least 2 feature columns (excluding Target).")

    CorrMatrix = Table[VarList].corr(method=Method).to_numpy()

    # Upper triangle of the matrix gives every feature pair once, without Python loops.
    Idx1, Idx2 = np.triu_indices(len(VarList), k=1)
    Names                  = np.array(VarList, dtype=object)
    CorrTable              = pd.DataFrame({"Var1": Names[Idx1], "Var2": Names[Idx2], "Corr": CorrMatrix[Idx1, Idx2]})
    CorrTable["AbsCorr"]   = CorrTable["Corr"].abs()
    CorrTable["Idx1"]      = Idx1
    CorrTable["Idx2"]      = Idx2
    CorrTable              = CorrTable.sort_values("AbsCorr", ascending=False).reset_index(drop=True)

    # Target correlations
    TargetCorr             = np.array([Table[Col].corr(Table[Target], method=MethodT) for Col in VarList])
    CorrTable["Var1TCorr"] = TargetCorr[CorrTable["Idx1"].to_numpy()]
    CorrTable["Var2TCorr"] = TargetCorr[CorrTable["Idx2"].to_numpy()]

    # Elimination pass — table sorted descending, break when below limit. DeleteIdx holds the
    # index of the feature that was deleted for each pair (-1 = none yet).
    Var1Idx, Var2Idx = CorrTable["Idx1"].to_numpy(), CorrTable["Idx2"].to_numpy()
    AbsCorr          = CorrTable["AbsCorr"].to_numpy()
    Var1AbsTCorr     = np.abs(CorrTable["Var1TCorr"].to_numpy())
    Var2AbsTCorr     = np.abs(CorrTable["Var2TCorr"].to_numpy())
    DeleteIdx        = np.full(len(CorrTable), -1)

    for Position in range(len(CorrTable)):
        if AbsCorr[Position] <= Limit:
            break
        if DeleteIdx[Position] != -1:
            continue  # one of this pair already marked for deletion

        DeleteVar = Var1Idx[Position] if Var1AbsTCorr[Position] < Var2AbsTCorr[Position] else Var2Idx[Position]

        # Propagate deletion to all rows involving DeleteVar
        Mask = (DeleteIdx == -1) & ((Var1Idx == DeleteVar) | (Var2Idx == DeleteVar))
        DeleteIdx[Mask] = DeleteVar

    CorrTable["Delete"] = np.where(DeleteIdx == -1, "-", Names[DeleteIdx])
    EliminatedColumns   = [VarList[Idx] for Idx in pd.unique(DeleteIdx[DeleteIdx != -1])]
    RemainingColumns    = [c for c in VarList if c not in EliminatedColumns]

    CorrTable = CorrTable.drop(columns=["AbsCorr", "Idx1", "Idx2"])[["Var1", "Var2", "Corr", "Var1TCorr", "Var2TCorr", "Delete"]]
    CorrTable = CorrTable.sort_values(["Var1", "Var2"]).reset_index(drop=True)

    _Step(
        f"Total columns      : {len(Table.columns)}\n"
        f"  Columns for corr   : {len(VarList)}\n"
        f"  Eliminated columns : {len(EliminatedColumns)}\n"
        f"  Remaining columns  : {len(RemainingColumns)}"
    )

    return CorrTable, RemainingColumns, EliminatedColumns

def VIFReport(
    Table: pd.DataFrame,
    Target: str | None = None,
    AutoClean: bool = False,
    Threshold: float = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Report the Variance Inflation Factor of each feature to detect multicollinearity.

    Arg:
       Table: Source table containing the feature columns; bool columns are treated as 0/1.
       Target: Optional target column excluded from the report.
       AutoClean: When True, drop non-numeric and constant columns and rows with null or inf
          values before the calculation; when False, raise an error if any are found.
       Threshold: Features with a VIF above this value are listed as high VIF features.

    Returns:
       Tuple of (Report, HighVIFFeatures); HighVIFFeatures lists features with VIF above Threshold.

    Raises:
       ImportError: If statsmodels is not installed (pip install "koco-ml[stats]").
       ValueError: If AutoClean is False and problem columns are found.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Columns = Table.columns.tolist()
    if Target is not None and Target in Columns:
        Columns.remove(Target)

    SubData = Table[Columns].copy()
    BoolColumns = SubData.select_dtypes(include="bool").columns
    if len(BoolColumns):
        SubData[BoolColumns] = SubData[BoolColumns].astype(int)

    # Data quality checks before VIF calculation
    Problems: dict[str, list[str]] = {}
    NonNumericCols = SubData.select_dtypes(exclude=[np.number]).columns.tolist()
    NumericCols = SubData.select_dtypes(include=[np.number]).columns.tolist()

    # 1) NonNumeric columns
    if NonNumericCols:
        Problems["NonNumericCols"] = NonNumericCols

    # 2) Columns with inf / -inf values
    if NumericCols:
        InfMask = np.isinf(SubData[NumericCols].to_numpy())
        if InfMask.any():
            InfCounts = pd.DataFrame(InfMask, columns=NumericCols).sum(axis=0)
            InfCols = InfCounts[InfCounts > 0].index.tolist()
            Problems["InfCols"] = InfCols

    # 3) Columns with null values
    NullCounts = SubData[NumericCols].isna().sum()
    NullCols = NullCounts[NullCounts > 0].index.tolist()
    if NullCols:
        Problems["NullCols"] = NullCols

    # 4) Constant columns; dropna=False so an all-null column also counts as constant
    ConstantCols = [Col for Col in NumericCols if SubData[Col].nunique(dropna=False) <= 1]
    if ConstantCols:
        Problems["ConstantCols"] = ConstantCols

    if AutoClean:
        if Problems:
            _Alert("VIF pre-check warnings:")
            _Info("All observations with problems are dropped before the VIF calculation. Details:")
            for ProblemName, ProblemColumns in Problems.items():
                _Step(f"{ProblemName}: {ProblemColumns}")

            StartShape = SubData.shape

            if NonNumericCols:
                SubData = SubData.drop(columns=NonNumericCols)

            SubData = SubData.replace([np.inf, -np.inf], np.nan).dropna()

            ConstantCols = [Col for Col in SubData.columns if SubData[Col].nunique(dropna=False) <= 1]
            if ConstantCols:
                SubData = SubData.drop(columns=ConstantCols)

            _Info(f"Before cleaning: {StartShape}, after cleaning: {SubData.shape}.")
        else:
            _Done("VIF pre-check passed: all observations are clean.")
    else:
        if Problems:
            ErrorMessage = "VIF pre-check failed:\n- " + "\n- ".join(f"{ProblemName}: {ProblemColumns}" for ProblemName, ProblemColumns in Problems.items())
            raise ValueError(ErrorMessage)
        else:
            _Done("VIF pre-check passed: no non-numeric, inf/-inf, null, or constant columns detected.")

    # variance_inflation_factor assumes the design matrix contains an intercept, so one is
    # added for the calculation only and skipped when the report is built.
    SubData.insert(0, "_Intercept", 1.0)

    Report = pd.DataFrame({
        "Feature": SubData.columns.tolist()[1:],
        "VIF"    : [round(variance_inflation_factor(SubData.values, i), 2) for i in range(1, len(SubData.columns))],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    HighVIFFeatures = Report[Report["VIF"] > Threshold]["Feature"].tolist()

    if HighVIFFeatures:
        _Alert(f"{len(HighVIFFeatures)} features are indicating potential multicollinearity: {HighVIFFeatures}")
    else:
        _Done(f"No features have high (> {Threshold}) VIF detected.")

    return Report, HighVIFFeatures

def SelectModelFeatures(
    Table: pd.DataFrame,
    Features: list[str],
    Target: str,
    Models: str | list[str] | None = None,
    Scoring: str | None = None,
    NNoiseFeatures: int = 10,
    CVFolds: int = 5,
    NRepeats: int = 10,
    RandomState: int = DEFAULT_RANDOM_STATE,
    MinSelect: int = 1,
    ModelType: str = "Regression",
    EliminationMethod: str = "Combined",
    NJobs: int | None = -1,
) -> pd.DataFrame:
    """
    Rank candidate features by predictive relevance and split them into Select / Eliminate.

    Two independent, model-agnostic relevance signals are measured per feature: cross-validated
    permutation importance (fit on each training fold, scored on the held-out fold, averaged
    across Models/folds/repeats) and mutual information with the Target. Random Gaussian
    "noise probe" columns are injected alongside the real Features; a real feature that cannot
    clearly outscore the noise probes carries no more signal than chance and is eliminated.
    This yields a data-driven cutoff instead of an arbitrary Top-K/threshold, and is fully
    reproducible via RandomState.

    Arg:
       Table: Source table containing the candidate feature and Target columns, already
          null-free and scaled by the caller; this function does not transform the data.
       Features: Candidate numeric feature names to evaluate.
       Target: Target column name; numeric for Regression, 0/1 for Binary.
       Models: One or more registry keys of ModelType used to score importance; using more than one
          model reduces the risk of a single algorithm's bias driving the decision. None uses
          ("LassoR", "RandFR") for Regression and ("LogisticC", "RandFC") for Binary.
       Scoring: sklearn scoring string used by permutation_importance. None uses "r2" for
          Regression and "roc_auc" for Binary.
       NNoiseFeatures: Number of random noise-probe columns used as the elimination baseline. More
          probes give a stricter and more stable baseline: a useless feature beats all probes
          with a probability of about 1 / (NNoiseFeatures + 1).
       CVFolds: Number of cross-validation folds (stratified for Binary).
       NRepeats: Number of shuffles per fold used by permutation_importance.
       RandomState: Seed controlling noise generation, CV splits, and model fitting.
       MinSelect: Minimum number of features to keep; the best-ranked eliminated features are
          reprieved if fewer than this many pass the noise baseline.
       ModelType: "Regression" or "Binary".
       EliminationMethod: "Combined" selects a feature when its CombinedScore (average percentile
          rank of both signals) is above the best noise probe. "OnlyNoise" is stricter: the
          permutation importance (above 0) and the mutual information must both be above the best
          noise probe in raw values.
       NJobs: Parallel jobs of the permutation importance; models that already run in parallel
          use NJobs for themselves and permute sequentially to avoid nested parallelism.

    Returns:
       Report DataFrame sorted by CombinedScore descending, covering both the real Features and
       the noise-probe rows (flagged via IsNoiseProbe) so the baselines stay auditable. Columns:
       Feature, IsNoiseProbe, PermutationImportance, MutualInfo, CombinedScore (0-100 scale),
       NoiseThreshold (best probe CombinedScore), PermutationNoiseThreshold, MutualInfoNoiseThreshold,
       Decision ("Select"/"Eliminate"/"Noise"), and Reason (populated for Eliminate/Noise rows
       only). Filter on Decision to get the two groups, e.g. Report.loc[Report.Decision == "Select", "Feature"].

    Raises:
       ValueError: If ModelType or EliminationMethod is unknown, inputs are missing or invalid,
          columns are not numeric or contain nulls, a Binary Target is not 0/1, MinSelect is larger
          than the number of Features, or a model key does not belong to ModelType.
       ImportError: If the package of a requested model is not installed.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold, StratifiedKFold

    if ModelType not in REGISTRIES:
        raise ValueError(f"ModelType must be one of {list(REGISTRIES)}.")
    if EliminationMethod not in _ELIMINATION_METHODS:
        raise ValueError(f"EliminationMethod must be one of {list(_ELIMINATION_METHODS)}.")
    if not Features:
        raise ValueError("Features must contain at least one column name.")

    Features = _ValidateTargetAndFeatures(Table, Target, list(Features), ModelType)
    NullColumns = [Column for Column in Features + [Target] if Table[Column].isna().any()]
    if NullColumns:
        raise ValueError(f"Columns contain null values; apply a NullPlan first: {NullColumns}")
    if CVFolds < 2:
        raise ValueError("CVFolds must be >= 2.")
    if NNoiseFeatures < 1:
        raise ValueError("NNoiseFeatures must be >= 1 so a noise baseline can be established.")
    if not 0 <= MinSelect <= len(Features):
        raise ValueError(f"MinSelect must be between 0 and the number of Features ({len(Features)}).")

    if Models is None:
        Models = ["LassoR", "RandFR"] if ModelType == "Regression" else ["LogisticC", "RandFC"]
    ModelKeys = [Models] if isinstance(Models, str) else list(Models)
    UnknownModels = [Key for Key in ModelKeys if Key not in REGISTRIES[ModelType]]
    if UnknownModels:
        raise ValueError(f"Unknown {ModelType} model key(s): {UnknownModels}. Must be one of {list(REGISTRIES[ModelType])}.")

    Scoring = Scoring or ("r2" if ModelType == "Regression" else "roc_auc")
    MutualInfo = mutual_info_regression if ModelType == "Regression" else mutual_info_classif
    SplitterClass = KFold if ModelType == "Regression" else StratifiedKFold

    # ---- Prepare data — Table itself is never modified ----
    Rng  = np.random.RandomState(RandomState)
    Data = Table[Features + [Target]].copy()

    NoiseNames = [f"_NoiseProbe{i + 1}" for i in range(NNoiseFeatures)]
    for Name in NoiseNames:
        Data[Name] = Rng.normal(size=len(Data))

    WorkFeatures = Features + NoiseNames
    X = Data[WorkFeatures]
    Y = Data[Target]

    MISeries = pd.Series(MutualInfo(X, Y, random_state=RandomState), index=WorkFeatures)
    MINorm   = MISeries.rank(pct=True)

    # ---- Cross-validated permutation importance, averaged across Models / folds ----
    Splitter      = SplitterClass(n_splits=CVFolds, shuffle=True, random_state=RandomState)
    ImportanceSum = pd.Series(0.0, index=WorkFeatures)
    RunCount      = 0

    for ModelKey in ModelKeys:
        EstimatorClass = LoadEstimatorClass(ModelKey)
        ModelKwargs    = BuildModelKwargs(ModelKey, EstimatorClass, RandomState, None)
        # Models that already run in parallel are permuted sequentially to avoid nested parallelism.
        if "n_jobs" in ModelKwargs:
            ModelKwargs["n_jobs"] = NJobs
            PermutationJobs = None
        else:
            PermutationJobs = NJobs

        for TrainIdx, ValIdx in Splitter.split(X, Y):
            XTrain, XVal = X.iloc[TrainIdx], X.iloc[ValIdx]
            YTrain, YVal = Y.iloc[TrainIdx], Y.iloc[ValIdx]

            Estimator = EstimatorClass(**ModelKwargs)
            Estimator.fit(XTrain, YTrain)

            PermResult = permutation_importance(
                Estimator, XVal, YVal,
                n_repeats=NRepeats, random_state=RandomState, scoring=Scoring, n_jobs=PermutationJobs,
            )
            ImportanceSum += PermResult.importances_mean
            RunCount      += 1

    ImportanceAvg  = ImportanceSum / RunCount
    ImportanceNorm = ImportanceAvg.rank(pct=True)

    # ---- Blend signals, rescale to an intuitive 0-100 score ----
    CombinedScore  = (ImportanceNorm + MINorm) / 2 * 100
    NoiseThreshold = round(CombinedScore.loc[NoiseNames].max(), 2)
    PermutationBaseline = max(float(ImportanceAvg.loc[NoiseNames].max()), 0.0)
    MutualInfoBaseline  = float(MISeries.loc[NoiseNames].max())

    # Noise-probe rows are kept in the Report so the baselines stay auditable.
    Report = pd.DataFrame({
        "Feature"                  : WorkFeatures,
        "IsNoiseProbe"             : [Feature in NoiseNames for Feature in WorkFeatures],
        "PermutationImportance"    : ImportanceAvg.loc[WorkFeatures].round(4).values,
        "MutualInfo"               : MISeries.loc[WorkFeatures].round(4).values,
        "CombinedScore"            : CombinedScore.loc[WorkFeatures].round(2).values,
        "NoiseThreshold"           : NoiseThreshold,
        "PermutationNoiseThreshold": round(PermutationBaseline, 4),
        "MutualInfoNoiseThreshold" : round(MutualInfoBaseline, 4),
    }).sort_values("CombinedScore", ascending=False).reset_index(drop=True)

    RealMask = ~Report["IsNoiseProbe"]
    if EliminationMethod == "Combined":
        Passed = Report["CombinedScore"] > NoiseThreshold
    else:
        PermutationPassed = Report["Feature"].map(ImportanceAvg) > PermutationBaseline
        MutualInfoPassed  = Report["Feature"].map(MISeries) > MutualInfoBaseline
        Passed = PermutationPassed & MutualInfoPassed

    Report["Decision"] = "Noise"
    Report.loc[RealMask, "Decision"] = np.where(Passed[RealMask], "Select", "Eliminate")

    # Only Eliminate / Noise rows carry a Reason — a Select just means the feature cleared the bar.
    Report["Reason"] = "-"
    Report.loc[~RealMask, "Reason"] = "Random baseline probe used to compute the noise thresholds; not a candidate feature."
    EliminateMask = RealMask & (Report["Decision"] == "Eliminate")
    if EliminationMethod == "Combined":
        Report.loc[EliminateMask, "Reason"] = Report.loc[EliminateMask, "CombinedScore"].map(
            lambda Score: f"Combined score {Score:.1f} does not beat the noise-probe baseline ({NoiseThreshold:.1f}); indistinguishable from random noise."
        )
    else:
        for Idx in Report.index[EliminateMask]:
            Failed = []
            if not PermutationPassed[Idx]:
                Failed.append(f"permutation importance {Report.at[Idx, 'PermutationImportance']:.4f} <= noise baseline {PermutationBaseline:.4f}")
            if not MutualInfoPassed[Idx]:
                Failed.append(f"mutual information {Report.at[Idx, 'MutualInfo']:.4f} <= noise baseline {MutualInfoBaseline:.4f}")
            Report.at[Idx, "Reason"] = "Does not beat the noise probes: " + "; ".join(Failed) + "."

    SelectedCount = int(((Report["Decision"] == "Select") & RealMask).sum())
    if SelectedCount < MinSelect:
        ReprieveIdx = Report.loc[EliminateMask].index[: MinSelect - SelectedCount]
        for Idx in ReprieveIdx:
            Score = Report.at[Idx, "CombinedScore"]
            Report.at[Idx, "Reason"] = (
                f"Reprieved to satisfy MinSelect={MinSelect}; it did not clear the noise-probe baseline "
                f"(combined score {Score:.1f})."
            )
            Report.at[Idx, "Decision"] = "Select"

    _Step(
        f"Elimination method : {EliminationMethod}\n"
        f"  Candidate features : {len(Features)}\n"
        f"  Selected features  : {int(((Report['Decision'] == 'Select') & RealMask).sum())}\n"
        f"  Eliminated features: {int(((Report['Decision'] == 'Eliminate') & RealMask).sum())}\n"
        f"  Noise threshold    : {NoiseThreshold:.1f} / 100 (combined score)"
    )

    return Report

def BackwardSelection(
    Table: pd.DataFrame,
    Target: str,
    Features: list[str] | None = None,
    MaxPValue: float = 0.05,
    ModelType: str = "Regression",
) -> tuple[list[str], pd.DataFrame]:
    """
    Backward elimination: refit a statistical model and drop the feature with the highest p-value
    until every remaining feature has a p-value of at most MaxPValue.

    An intercept is always added and is never removed. Regression uses OLS and Binary uses Logit
    (statsmodels).

    Arg:
       Table: Source table containing the Target and the Features, null-free; scale or encode the
          features before calling it when the model needs it.
       Target: Numeric target column; 0/1 for Binary.
       Features: Candidate numeric feature names. If None, every column except Target is used.
       MaxPValue: Largest p-value a feature may have to stay, between 0 and 1.
       ModelType: "Regression" (OLS) or "Binary" (Logit).

    Returns:
       Tuple of (RemainingFeatures, Steps).
         RemainingFeatures: Kept features sorted by p-value, smallest first; empty if none passes.
         Steps: One row per removal with Step, RemovedFeature, PValue and Remaining (number of
            features left after the removal).

    Raises:
       ImportError: If statsmodels is not installed (pip install "koco-ml[stats]").
       ValueError: If ModelType or MaxPValue is invalid, columns are missing, not numeric, contain
          nulls or inf values, a Binary Target is not 0/1, or the model cannot be fitted
          (perfect separation, a Logit fit that does not converge, or collinear features); use
          Correlation or VIFReport first.
    """
    import warnings

    import statsmodels.api as sm
    from statsmodels.tools import sm_exceptions

    if ModelType not in REGISTRIES:
        raise ValueError(f"ModelType must be one of {list(REGISTRIES)}.")
    if not 0 < MaxPValue < 1:
        raise ValueError("MaxPValue must be between 0 and 1.")
    Features = _ValidateTargetAndFeatures(Table, Target, Features, ModelType)
    NullColumns = [Column for Column in Features + [Target] if Table[Column].isna().any()]
    if NullColumns:
        raise ValueError(f"Columns contain null values; apply a NullPlan first: {NullColumns}")

    X = Table[Features].astype(float).copy()
    X.insert(0, "_Intercept", 1.0)
    Y = Table[Target].astype(float)
    ModelClass = sm.Logit if ModelType == "Binary" else sm.OLS

    Remaining, Steps = list(Features), []
    PValues = pd.Series(dtype=float)
    while Remaining:
        # Newer statsmodels versions only warn about perfect separation, so the warning is turned into an error.
        SeparationWarning = getattr(sm_exceptions, "PerfectSeparationWarning", None)
        SeparationErrors = tuple(Name for Name in (getattr(sm_exceptions, "PerfectSeparationError", None), SeparationWarning) if Name)
        try:
            with warnings.catch_warnings():
                if SeparationWarning:
                    warnings.simplefilter("error", SeparationWarning)
                Fitted = ModelClass(Y, X[["_Intercept"] + Remaining]).fit(disp=0) if ModelType == "Binary" else ModelClass(Y, X[["_Intercept"] + Remaining]).fit()
        except (np.linalg.LinAlgError, *SeparationErrors) as Error:
            raise ValueError(f"The model could not be fitted ({type(Error).__name__}); check the features with Correlation / VIFReport first.") from Error
        # statsmodels 0.15+ may stop without converging instead of flagging perfect separation; such p-values are unreliable.
        if ModelType == "Binary" and not Fitted.mle_retvals.get("converged", True):
            raise ValueError("The model could not be fitted (the Logit fit did not converge, often because of perfect separation); check the features with Correlation / VIFReport first.")

        PValues = Fitted.pvalues.drop("_Intercept")
        if PValues.isna().any():
            raise ValueError(f"p-values could not be calculated for {PValues.index[PValues.isna()].tolist()}; the features are collinear or constant.")

        Worst = PValues.idxmax()
        if PValues[Worst] <= MaxPValue:
            break
        Remaining.remove(Worst)
        Steps.append({"Step": len(Steps) + 1, "RemovedFeature": Worst, "PValue": round(float(PValues[Worst]), 4), "Remaining": len(Remaining)})

    RemainingFeatures = PValues.sort_values().index.tolist() if Remaining else []
    _Step(
        f"Total features      : {len(Features)}\n"
        f"  Eliminated features: {len(Features) - len(RemainingFeatures)}\n"
        f"  Remaining features : {len(RemainingFeatures)}"
    )
    if not RemainingFeatures:
        _Alert(f"No feature has a p-value of at most {MaxPValue}.")

    return RemainingFeatures, pd.DataFrame(Steps, columns=["Step", "RemovedFeature", "PValue", "Remaining"])
