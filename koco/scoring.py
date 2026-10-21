from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _CheckArtifact, _CheckColumns, _CheckUniqueColumns
from .preprocessing import ApplyNullPlan, ApplyOutlierPlan, ApplyScalingPlan, ApplyWOEPlan

__all__ = ["ApplyModel", "CalculateModelPredictions", "ApplyModelBundle"]

#********* Scoring Functions *********#

def PositiveProbability(
    Model,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Return the predicted probability of class 1.

    Arg:
       Model: Fitted classifier with predict_proba and classes_.
       X: Feature table.

    Returns:
       Array of class 1 probabilities.

    Raises:
       ValueError: If class 1 is not among the model classes.
    """
    return np.asarray(Model.predict_proba(X))[:, list(Model.classes_).index(1)]

def ApplyModel(
    Table: pd.DataFrame,
    Artifact: dict,
    Threshold: float | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Generate predictions for a table with a fitted model artifact.

    The model type is read from the artifact: regression models return predicted values, binary
    models return the probability of class 1 and the 0/1 prediction at the given threshold.

    Arg:
       Table: Table containing the Features the model was trained on, preprocessed with the same
          plans as Train; Target is not required.
       Artifact: One entry of the ModelArtifacts dict returned by RegressionModelling or
          ClassificationModelling, e.g. ModelArtifacts["RandFR"].
       Threshold: Binary models only. Probability cut-off for class 1; if None, the threshold
          stored in the artifact during modelling is used.

    Returns:
       Regression: Series named Prediction, aligned with the index of Table.
       Binary: DataFrame with Probability and Prediction columns, aligned with the index of Table.

    Raises:
       ValueError: If Artifact is not a model artifact (Model, ModelType and Features are required),
          ModelType is unknown, a feature column is missing, duplicated or not numeric in Table, or
          Threshold is invalid or given for a regression model.
    """
    _CheckArtifact(Artifact)

    Features = Artifact["Features"]
    _CheckColumns(Table, Features)
    _CheckUniqueColumns(Table[Features])
    NonNumericFeatures = [Column for Column in Features if not pd.api.types.is_numeric_dtype(Table[Column])]
    if NonNumericFeatures:
        raise ValueError(f"Features must be numeric: {NonNumericFeatures}")

    X = Table[Features]
    if Artifact["ModelType"] == "Regression":
        if Threshold is not None:
            raise ValueError("Threshold is only used for Binary models.")
        Predictions = Artifact["Model"].predict(X)
        return pd.Series(np.asarray(Predictions).ravel(), index=Table.index, name="Prediction")

    Threshold = Artifact.get("Threshold") if Threshold is None else Threshold
    if not isinstance(Threshold, (int, float)) or isinstance(Threshold, bool) or not 0 <= Threshold <= 1:
        raise ValueError("Threshold must be a number between 0 and 1.")

    Probability = PositiveProbability(Artifact["Model"], X)
    return pd.DataFrame(
        {"Probability": Probability, "Prediction": (Probability >= Threshold).astype(int)},
        index=Table.index,
    )

def CalculateModelPredictions(
    Table: pd.DataFrame,
    OutputColumns: list[str],
    Artifacts: dict,
    Rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Return selected input columns alongside predictions from fitted models.

    Arg:
       Table: Input DataFrame containing the model features.
       OutputColumns: Columns to include in the returned DataFrame.
       Artifacts: Mapping of algorithm names to fitted regression or classification artifacts.
       Rename: Optional mapping from algorithm names to prediction column prefixes.

    Returns:
       A DataFrame with the selected input columns. Regression predictions use the
       algorithm name (or its renamed value); classification outputs use columns
       suffixed with the output labels, such as '_Probability' and '_Prediction'.

    Raises:
       ValueError: If requested columns are missing, a model returns an unsupported
          output type or incompatible length/index, or prediction columns would
          overwrite existing columns.
    """
    Rename = Rename or {}

    MissingColumns = [Column for Column in OutputColumns if Column not in Table.columns]
    if MissingColumns:
        raise ValueError(f"OutputColumns don't exist in the table: {MissingColumns}")

    ModelPredictions = Table[OutputColumns].copy()

    for AlgorithmName, Details in Artifacts.items():
        OutputName = Rename.get(AlgorithmName, AlgorithmName)
        Predictions = ApplyModel(Table, Details)

        if not isinstance(Predictions, (pd.Series, pd.DataFrame)):
            raise ValueError(f"Model '{AlgorithmName}' returned an unsupported prediction type")
        if len(Predictions) != len(Table) or not Predictions.index.equals(Table.index):
            raise ValueError(f"Model '{AlgorithmName}' predictions are not aligned with the input rows")

        if isinstance(Predictions, pd.Series):
            PredictionColumns = {OutputName: Predictions}
        else:
            if Predictions.empty or not Predictions.columns.is_unique:
                raise ValueError(f"Model '{AlgorithmName}' returned no predictions or duplicate output labels")
            PredictionColumns = {
                f"{OutputName}{PredictionName}": Predictions[PredictionName]
                for PredictionName in Predictions.columns
            }

        ConflictingColumns = [Column for Column in PredictionColumns if Column in ModelPredictions.columns]
        if ConflictingColumns:
            raise ValueError(f"Prediction columns would overwrite existing columns: {ConflictingColumns}")

        for Column, Values in PredictionColumns.items():
            ModelPredictions[Column] = Values

    return ModelPredictions

def ApplyModelBundle(
    Table: pd.DataFrame,
    Bundle: dict,
    Threshold: float | None = None,
    Verbose: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Score a raw table with a model bundle: the plans are applied in the order Null, Outlier,
    WOE, Scaling (only those stored in the bundle) and then the model.

    Arg:
       Table: Raw table with the columns the plans and the model need; Target is not required.
       Bundle: Bundle loaded with ImportModelBundle.
       Threshold: Binary models only. Probability cut-off for class 1; if None, the threshold
          stored in the artifact is used.
       Verbose: If True, the null plan prints its checks.

    Returns:
       Same as ApplyModel: a Series named Prediction (Regression) or a DataFrame with Probability
       and Prediction columns (Binary), aligned with the index of the rows that were scored.

    Raises:
       ValueError: If Bundle is not a model bundle, or the plans or the model reject the table.

    Notes:
       A null plan with DropRows removes rows, so the result can have fewer rows than Table; the
          index of Table is kept.
    """
    if not isinstance(Bundle, dict) or not {"Artifact", "Plans"} <= Bundle.keys():
        raise ValueError("Bundle must be loaded with ImportModelBundle.")
    _CheckArtifact(Bundle["Artifact"])

    Plans = Bundle["Plans"]
    Data = Table
    if "Null" in Plans:
        Data = ApplyNullPlan(Data, Plans["Null"], Verbose=Verbose)
    if "Outlier" in Plans:
        Data = ApplyOutlierPlan(Data, Plans["Outlier"])
    if "WOE" in Plans:
        Data = ApplyWOEPlan(Data, Plans["WOE"])
    if "Scaling" in Plans:
        Data = ApplyScalingPlan(Data, Plans["Scaling"])

    return ApplyModel(Data, Bundle["Artifact"], Threshold)
