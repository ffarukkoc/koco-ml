"""Private helpers shared by the koco modules. Not part of the public API; imports only pandas and numpy."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RANDOM_STATE = 1234

# Plans a model bundle can carry, in the order ApplyModelBundle applies them.
BUNDLE_PLANS = ("Null", "Outlier", "WOE", "Scaling")
# Message symbols: ❗ needs attention, ℹ information, ✓ completed, • regular progress.
def _Alert(Message: str) -> None:
    """Print a message that needs attention (warning, skipped item, overwritten file)."""
    print(f"❗ {Message}")

def _Info(Message: str) -> None:
    """Print an informational message."""
    print(f"ℹ {Message}")

def _Done(Message: str) -> None:
    """Print a completion message."""
    print(f"✓ {Message}")

def _Step(Message: str) -> None:
    """Print a regular progress message."""
    print(f"• {Message}")

def _OutputPath(
    FilePath: str | Path | None,
    FunctionName: str,
    Suffix: str,
    Notify: bool = True,
) -> Path:
    """
    Resolve and prepare the output file of an export function.

    Arg:
       FilePath: None writes <yymmdd>_<FunctionName><Suffix> to the current folder. A path with a
          file extension is the output file. An existing folder, or a path without an extension,
          is treated as a folder that receives the default file name.
       FunctionName: Name used in the default file name.
       Suffix: File extension of the default file name, e.g. ".xlsx".
       Notify: If True, print a message when a file with the same name already exists.

    Returns:
       Output file path; missing folders are created.

    Raises:
       TypeError: If FilePath is not None, a string or a Path.
    """
    if FilePath is not None and not isinstance(FilePath, (str, Path)):
        raise TypeError("FilePath must be a string, a Path or None.")

    DefaultName = f"{date.today().strftime('%y%m%d')}_{FunctionName}{Suffix}"
    if FilePath is None:
        OutputPath = Path(DefaultName)
    else:
        OutputPath = Path(FilePath)
        if OutputPath.is_dir() or OutputPath.suffix == "":
            OutputPath = OutputPath / DefaultName

    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    if Notify and OutputPath.exists():
        _Alert(f"A file with the same name exists; it will be updated: {OutputPath}")
    return OutputPath

def _CheckColumns(
    Table: pd.DataFrame,
    Columns: list[str],
) -> None:
    """
    Check that every column exists in the table.

    Arg:
       Table: Table to inspect.
       Columns: Column names that must exist.

    Returns:
       None.

    Raises:
       ValueError: If one or more columns are missing.
    """
    MissingColumns = [Column for Column in dict.fromkeys(Columns) if Column not in Table.columns]
    if MissingColumns:
        raise ValueError(f"Columns not found in Table: {MissingColumns}")

def _CheckUniqueColumns(
    Table: pd.DataFrame,
) -> None:
    """
    Check that the table has no duplicated column names.

    Arg:
       Table: Table to inspect.

    Returns:
       None.

    Raises:
       ValueError: If a column name appears more than once.
    """
    DuplicateColumns = Table.columns[Table.columns.duplicated()].unique().tolist()
    if DuplicateColumns:
        raise ValueError(f"Table has duplicated column names: {DuplicateColumns}")

def _Rate(
    Count: pd.Series,
) -> pd.Series:
    """
    Convert counts to percentage shares.

    Arg:
       Count: Observation counts.

    Returns:
       Series of shares in percent, rounded to 2 decimals.

    Raises:
       None.
    """
    return round(Count * 100 / Count.sum(), 2)

def _ValidatePlan(
    Plan: dict,
) -> None:
    """
    Check the outer structure of a plan.

    Arg:
       Plan: Plan produced by a Fit function or loaded with ImportJson.

    Returns:
       None.

    Raises:
       ValueError: If Plan is not a dict that contains a Columns dict.
    """
    if not isinstance(Plan, dict) or not isinstance(Plan.get("Columns"), dict):
        raise ValueError('Plan must be a dict that contains a "Columns" dict.')

def _CheckArtifact(
    Artifact: dict,
) -> None:
    """
    Check that a dict is a model artifact.

    Arg:
       Artifact: One entry of the ModelArtifacts dict returned by the modelling functions.

    Returns:
       None.

    Raises:
       ValueError: If Artifact has no Model, ModelType or Features, or ModelType is unknown.
    """
    if isinstance(Artifact, dict) and "Task" in Artifact and "ModelType" not in Artifact:
        raise ValueError("Artifact uses the old 'Task' key; models trained with an older koco version must be retrained.")
    if not isinstance(Artifact, dict) or not {"Model", "ModelType", "Features"} <= Artifact.keys():
        raise ValueError("Artifact must be a ModelArtifacts entry containing 'Model', 'ModelType' and 'Features'.")
    if Artifact["ModelType"] not in ("Regression", "Binary"):
        raise ValueError(f"Artifact ModelType must be 'Regression' or 'Binary', got {Artifact['ModelType']!r}.")

def _ValidateTargetAndFeatures(
    Table: pd.DataFrame,
    Target: str,
    Features: list[str] | None,
    ModelType: str,
) -> list[str]:
    """
    Validate the Target and Features columns shared by the modelling and feature selection functions.

    Arg:
       Table: Table containing the Target and Features columns.
       Target: Numeric target column; 0/1 for Binary.
       Features: Numeric feature names; None uses every column except Target.
       ModelType: "Regression" or "Binary".

    Returns:
       The validated Features list.

    Raises:
       TypeError: If Table, Target or Features have the wrong type.
       ValueError: If Table is empty, a column is missing, not numeric or contains inf values,
          Target is also a feature, features are duplicated, or Target is not usable for ModelType.
    """
    if not isinstance(Table, pd.DataFrame):
        raise TypeError("Table must be a DataFrame.")
    if Table.empty:
        raise ValueError("Table must not be empty.")

    if not isinstance(Target, str):
        raise TypeError("Target must be a string.")
    _CheckColumns(Table, [Target])
    if not pd.api.types.is_numeric_dtype(Table[Target]) or pd.api.types.is_bool_dtype(Table[Target]):
        raise ValueError(f"Target must be numeric: {Target}")
    TargetValues = Table[Target].dropna().to_numpy(dtype=float)
    if not np.isfinite(TargetValues).all():
        raise ValueError(f"Target contains inf or -inf values: {Target}")
    if len(np.unique(TargetValues)) < 2:
        raise ValueError(f"Target must contain at least 2 distinct non-null values: {Target}")
    if ModelType == "Binary" and set(np.unique(TargetValues)) != {0.0, 1.0}:
        raise ValueError(f"Binary Target must contain only 0 and 1 (multiclass is not supported yet): {Target}")

    if Features is None:
        Features = [Column for Column in Table.columns if Column != Target]
    if not isinstance(Features, list) or not Features or not all(isinstance(Column, str) for Column in Features):
        raise TypeError("Features must be a non-empty list of column names.")
    _CheckColumns(Table, Features)
    if Target in Features:
        raise ValueError(f"Target column '{Target}' must not also appear in Features.")
    DuplicateFeatures = list(dict.fromkeys(Column for Column in Features if Features.count(Column) > 1))
    if DuplicateFeatures:
        raise ValueError(f"Duplicate columns in Features: {DuplicateFeatures}")
    NonNumericFeatures = [Column for Column in Features if not pd.api.types.is_numeric_dtype(Table[Column])]
    if NonNumericFeatures:
        raise ValueError(f"Features must be numeric: {NonNumericFeatures}")
    InfMask = np.isinf(Table[Features].to_numpy(dtype=float, na_value=np.nan)).any(axis=0)
    if InfMask.any():
        raise ValueError(f"Features contain inf or -inf values: {[Column for Column, IsInf in zip(Features, InfMask) if IsInf]}")

    return Features
