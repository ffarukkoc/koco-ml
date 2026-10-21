from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._helpers import (
    _Alert, _CheckColumns, _CheckUniqueColumns, _Done, _Info, _OutputPath, _Step, _ValidatePlan, _ValidateTargetAndFeatures,
)
from .analysis import DataSummary
from .general import ExportExcel

__all__ = [
    "ProfileNullColumns", "FitNullPlan", "ApplyNullPlan",
    "ApplyOutlierPlan", "FitWOEPlan", "ApplyWOEPlan", "FitScalingPlan", "ApplyScalingPlan",
]

#********* Preprocessing Functions *********#

# Strategies available in a NullPlan; descriptions are in NullStrategyGuide.
# KeepNaN leaves the column untouched, so it may only be used by NaN-tolerant algorithms (see ModelRegistryGuide).
NULL_STRATEGIES = ("Drop", "DropRows", "Median", "Mean", "Mode", "FixValue", "KeepNaN")

# Methods available in FitScalingPlan; descriptions are in ScalingMethodGuide.
SCALING_METHODS = ("Standardization", "Normalization", "Robust", "MaxAbs")

def _ScalerClasses() -> dict[str, type]:
    """
    Map every scaling method to its scikit-learn scaler.

    Arg:
       None.

    Returns:
       Dict keyed by SCALING_METHODS.

    Raises:
       ImportError: If scikit-learn is not installed.
    """
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

    return {
        "Standardization": StandardScaler,
        "Normalization": MinMaxScaler,
        "Robust": RobustScaler,
        "MaxAbs": MaxAbsScaler,
    }

def _IsNumber(
    Value,
) -> bool:
    """
    Check that a value is a real number and not a bool.

    Arg:
       Value: Value to check.

    Returns:
       True for int, float and numpy numbers.

    Raises:
       None.
    """
    return isinstance(Value, (int, float, np.number)) and not isinstance(Value, bool)

def ProfileNullColumns(
    Table: pd.DataFrame,
    FilePath: str | Path | None = None,
) -> pd.DataFrame:
    """
    Summarize only the columns that contain nulls so a per-column strategy can be chosen.

    Diagnostic step that does not modify Table. Its output feeds the NullPlan passed to
    FitNullPlan; nothing is resolved automatically. Use NullStrategyGuide(Print=True) to review
    the available actions when needed.

    Arg:
       Table: Source table to inspect for null values.
       FilePath: None does not export. A path with an extension (e.g. "Data/NullAnalysis/Profile.xlsx")
          is the Excel file itself; a folder (existing, or a path without an extension, e.g.
          "Data/NullAnalysis") receives <yymmdd>_NullProfile.xlsx. Missing folders are created.

    Returns:
       DataFrame with the null-relevant statistics, empty if no column contains nulls.

    Raises:
       TypeError: If FilePath is not None, a string or a Path.
       OSError: If the workbook cannot be written.

    Notes:
       A file with the same name is updated and a message is printed. Nothing is exported when no
          column contains nulls.
    """
    if FilePath is not None and not isinstance(FilePath, (str, Path)):
        raise TypeError("FilePath must be a string, a Path or None.")

    NullColumns = Table.columns[Table.isna().any()].tolist()
    if not NullColumns:
        _Info("No null values found in the table columns.")
        return pd.DataFrame()

    _Info(f"{len(NullColumns)} / {len(Table.columns)} columns contain nulls — review and define a NullPlan for these:")
    _Info("Create a list of tuples like [('ColumnName1', 'Drop'), ('ColumnName2', 'Median'), ('ColumnName3', 'KeepNaN'), ('ColumnName4', 'FixValue', 10)]")

    NullProfileColumns = ["ColumnName", "TotalCount", "DistinctCount", "MinValue", "MaxValue", "AvgValue", "ModValue", "MedValue", "NullCount", "NullRate"]
    NullProfile = DataSummary(Table[NullColumns])[NullProfileColumns]

    if FilePath is not None:
        # ExportExcel reports an existing file, so the default name is resolved silently here.
        ExportExcel({"NullProfile": NullProfile}, _OutputPath(FilePath, "NullProfile", ".xlsx", Notify=False))

    return NullProfile

def _ValidateNullRules(
    Table: pd.DataFrame,
    NullPlan: list[tuple],
) -> list[tuple[str, str, list]]:
    """
    Validate the rules of FitNullPlan before anything is printed or fitted.

    Arg:
       Table: Training table the rules refer to.
       NullPlan: Rules such as ("Income", "Median") or ("Score", "FixValue", 0).

    Returns:
       List of (Column, Strategy, Rest) where Rest holds the optional fixed value.

    Raises:
       ValueError: If a rule is malformed, a strategy is unknown, a column has more than one rule,
          FixValue has no value, or Median / Mean / Mode cannot be calculated for a column.
    """
    Rules = []
    SeenColumns = set()
    for Rule in NullPlan:
        if not isinstance(Rule, (tuple, list)) or len(Rule) not in (2, 3) or not isinstance(Rule[0], str):
            raise ValueError(f"Each rule must be (Column, Strategy) or (Column, 'FixValue', Value), got {Rule!r}.")
        Column, Strategy, *Rest = Rule

        if Strategy not in NULL_STRATEGIES:
            raise ValueError(f"Unknown Strategy '{Strategy}' for column '{Column}'. Must be one of {sorted(NULL_STRATEGIES)}.")
        if Column in SeenColumns:
            raise ValueError(f"Column '{Column}' has more than one rule in NullPlan.")
        SeenColumns.add(Column)
        if Strategy == "FixValue" and (not Rest or Rest[0] is None):
            raise ValueError(f"FixValue needs a value for column '{Column}': ('{Column}', 'FixValue', 0).")

        if Column in Table.columns and Strategy in ("Median", "Mean", "Mode"):
            Series = Table[Column]
            if Strategy != "Mode" and (not pd.api.types.is_numeric_dtype(Series) or pd.api.types.is_bool_dtype(Series)):
                raise ValueError(f"{Strategy} needs a numeric column, but '{Column}' is {Series.dtype}.")
            if Series.dropna().empty:
                raise ValueError(f"Column '{Column}' has only null values, so {Strategy} cannot be calculated; use Drop or FixValue.")

        Rules.append((Column, Strategy, Rest))
    return Rules

def FitNullPlan(
    Table: pd.DataFrame,
    NullPlan: list[tuple[str, str] | tuple[str, str, int | float | str]],
) -> tuple[dict, pd.DataFrame]:
    """
    Fit a per-column null-handling plan on the training table.

    The returned fitted plan is the artifact that must be persisted with ExportJson and
    re-applied identically at scoring time with ApplyNullPlan, so production imputes the
    exact same values. Plan columns that exist but have no nulls in Table are still fitted,
    so nulls that appear later in production are handled.

    Arg:
       Table: Training table the fill values are calculated from.
       NullPlan: Rules like [("A", "Drop"), ("B", "Median"), ("D", "FixValue", 10)]; see NullStrategyGuide.

    Returns:
       Tuple of (FittedPlan, ColumnValidation).
         FittedPlan: JSON-serializable dict {"Columns": {Column: {"Strategy", "Value"}}}.
         ColumnValidation: DataFrame listing columns with nulls missing from NullPlan, plan columns
            not found in Table (skipped), and plan columns without nulls in Table (still fitted).

    Raises:
       ValueError: If a rule is malformed, a strategy is unknown, a column has more than one rule,
          FixValue has no value, or Median / Mean / Mode cannot be calculated for a column
          (non-numeric or all-null column).
    """
    Rules = _ValidateNullRules(Table, NullPlan)

    NullColumns = set(Table.columns[Table.isna().any()])
    PlanColumns = [Column for Column, _, _ in Rules]

    ColumnValidation = [
        {"ColumnName": Column, "Status": "Has nulls but missing in NullPlan"}
        for Column in Table.columns
        if Column in NullColumns and Column not in PlanColumns
    ]
    ColumnValidation.extend(
        {
            "ColumnName": Column,
            "Status": "Missing in Source Table; skipped" if Column not in Table.columns
                      else "No nulls in Source Table; still fitted",
        }
        for Column in PlanColumns
        if Column not in NullColumns
    )

    if not ColumnValidation:
        ColumnValidation = [{
            "ColumnName": "Passed",
            "Status": "All columns match between NullPlan and the table's null-value columns.",
        }]

    ColumnValidation = pd.DataFrame(ColumnValidation, columns=["ColumnName", "Status"])
    print(ColumnValidation.to_string(index=False))

    FittedPlan = {"Columns": {}}
    for Column, Strategy, Rest in Rules:
        if Column not in Table.columns:
            continue

        if Strategy == "Median":
            Value = float(Table[Column].median())
        elif Strategy == "Mean":
            Value = float(Table[Column].mean())
        elif Strategy == "Mode":
            Value = Table[Column].dropna().mode().iloc[0]
        elif Strategy == "FixValue":
            Value = Rest[0]
        else:
            Value = None  # Drop / DropRows / KeepNaN carry no fill value

        # Numpy scalars are unwrapped so the plan stays JSON-serializable.
        FittedPlan["Columns"][Column] = {"Strategy": Strategy, "Value": Value.item() if isinstance(Value, np.generic) else Value}

    return FittedPlan, ColumnValidation

def ApplyNullPlan(
    Table: pd.DataFrame,
    Plan: dict,
    NaNHandling: str = "Full",
    Verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply a previously fitted null-handling plan to a table.

    Works on Train, Test, or a future production batch — the same columns are dropped and
    imputed with the exact same values every time.

    Arg:
       Table: Source table the plan is applied to.
       Plan: Null plan produced by FitNullPlan or loaded with ImportJson.
       NaNHandling: "Full" leaves KeepNaN columns untouched, "Safe" drops them for a NaN-free result.
       Verbose: If False, nothing is printed (useful for scoring and loops).

    Returns:
       New DataFrame with the plan applied; Table itself is not modified. DropRows keeps the
       original index so removed rows can be traced back.

    Raises:
       ValueError: If NaNHandling is not Full or Safe, or Plan is not a valid Null plan.

    Notes:
       Columns that are not changed share memory with Table; only changed columns are copied.
       With DropRows the printed row ids are index labels of Table.
    """
    if NaNHandling not in ("Full", "Safe"):
        raise ValueError("NaNHandling must be 'Full' or 'Safe'.")
    _ValidatePlan(Plan)
    PlanColumns = Plan["Columns"]
    InvalidSpecs = [Column for Column, Spec in PlanColumns.items() if not isinstance(Spec, dict) or Spec.get("Strategy") not in NULL_STRATEGIES]
    if InvalidSpecs:
        raise ValueError(f"Plan columns without a valid null Strategy: {InvalidSpecs}")

    Data = Table.copy(deep=False)

    NullColumns = Data.columns[Data.isna().any()].tolist()
    NotInPlan = [Column for Column in NullColumns if Column not in PlanColumns]
    NotInTable = [Column for Column in PlanColumns if Column not in Data.columns]
    WithoutNulls = [Column for Column in PlanColumns if Column in Data.columns and Column not in NullColumns]

    if Verbose:
        if NotInPlan:
            _Alert(f"{len(NotInPlan)} features have NULL values but are missing in the Plan: {NotInPlan}")
        if NotInTable:
            _Alert(f"{len(NotInTable)} features are in the Plan but not found in the table: {NotInTable}")
        if WithoutNulls:
            _Info(f"{len(WithoutNulls)} features are in the Plan but do not have NULL values in the table: {WithoutNulls}")
        if not (NotInPlan or NotInTable or WithoutNulls):
            _Done("PASSED: All NULL columns match the Plan.")

    DropRowColumns = [Column for Column, Spec in PlanColumns.items() if Spec["Strategy"] == "DropRows" and Column in Data.columns]
    if DropRowColumns:
        Before = len(Data)
        DroppedIds = Data.index[Data[DropRowColumns].isna().any(axis=1)]
        Data = Data.dropna(subset=DropRowColumns)
        if Verbose:
            _Step(f"DropRows ({DropRowColumns}): {Before:,} → {len(Data):,} rows; index is kept.")
            if len(DroppedIds):
                Shown = DroppedIds[:10].tolist()
                _Step(f"Dropped row ids: {Shown}{' ...' if len(DroppedIds) > len(Shown) else ''}")

    for Column, Spec in PlanColumns.items():
        if Column in Data.columns and Spec["Strategy"] in ("Median", "Mean", "Mode", "FixValue"):
            Data[Column] = Data[Column].fillna(Spec["Value"])

    DropColumns = [Column for Column, Spec in PlanColumns.items() if Spec["Strategy"] == "Drop" and Column in Data.columns]
    if NaNHandling == "Safe":
        DropColumns += [Column for Column, Spec in PlanColumns.items() if Spec["Strategy"] == "KeepNaN" and Column in Data.columns]

    if DropColumns:
        Data = Data.drop(columns=DropColumns)

    return Data

def ApplyOutlierPlan(
    Table: pd.DataFrame,
    Plan: dict,
) -> pd.DataFrame:
    """
    Cap outliers of selected columns with fixed upper or lower limits.

    Arg:
       Table: Source table the limits are applied to.
       Plan: Plan such as {"Columns": {"Income": ["U", 250000], "Age": ["L", 18]}}. "U" caps
          values above the limit, "L" raises values below the limit. Columns missing in Table and
          methods other than "U" / "L" are reported and skipped.

    Returns:
       New DataFrame with the limits applied; Table itself is not modified.

    Raises:
       ValueError: If Plan is not a dict with a Columns dict, an entry is not [Method, Value]
          with a numeric Value, or Table has duplicated column names.

    Notes:
       Columns that are not changed share memory with Table; only changed columns are copied.
    """
    _ValidatePlan(Plan)
    _CheckUniqueColumns(Table)

    Data = Table.copy(deep=False)
    for Column, Spec in Plan["Columns"].items():
        if not isinstance(Spec, (list, tuple)) or len(Spec) != 2 or not _IsNumber(Spec[1]):
            raise ValueError(f"Entry for '{Column}' must be [Method, Value] with a numeric Value, got {Spec!r}.")
        Method, Value = Spec

        if Column not in Data.columns:
            _Alert(f"{Column} exists in the outlier plan but not found in the DataFrame.")
            continue
        if Method not in ("U", "L"):
            _Alert(f"{Column}: Method must be 'U' or 'L', got {Method!r}; skipped.")
            continue

        Data[Column] = Data[Column].clip(upper=Value) if Method == "U" else Data[Column].clip(lower=Value)

    return Data

def FitScalingPlan(
    Table: pd.DataFrame,
    Scaling: dict[str, list[str]],
) -> dict:
    """
    Fit a scaling plan for selected columns using one or more scaling methods.

    Supported methods are Standardization, Normalization, Robust and MaxAbs; use
    ScalingMethodGuide(Print=True) to compare them.

    Arg:
       Table: Training table the scaling parameters are calculated from; it is not modified.
       Scaling: Mapping of scaling methods to their column names,
          e.g. {"Standardization": ["Age"], "Robust": ["Income"]}.

    Returns:
       Fitted plan {"Columns": {Column: {"Strategy", "Offset", "Scale"}}}. Apply it to Train, Test
       or production data with ApplyScalingPlan and save it with ExportJson.

    Raises:
       ImportError: If scikit-learn is not installed.
       TypeError: If Scaling is not a dict or its values are not lists of strings.
       ValueError: If a scaling method is unknown, a column is missing, or a column is assigned
          to more than one method.
    """
    ScalerClasses = _ScalerClasses()

    if not isinstance(Scaling, dict):
        raise TypeError("Scaling must be a dict.")

    UnknownMethods = [Method for Method in Scaling if Method not in SCALING_METHODS]
    if UnknownMethods:
        raise ValueError(f"Unknown scaling methods: {UnknownMethods}. Must be one of {list(SCALING_METHODS)}.")

    InvalidValues = [Method for Method, Columns in Scaling.items() if not isinstance(Columns, list) or not all(isinstance(Column, str) for Column in Columns)]
    if InvalidValues:
        raise TypeError(f"Scaling values must be lists of strings. Invalid methods: {InvalidValues}")

    AllColumns = [Column for Columns in Scaling.values() for Column in Columns]
    _CheckColumns(Table, AllColumns)
    DuplicateColumns = list(dict.fromkeys(Column for Column in AllColumns if AllColumns.count(Column) > 1))
    if DuplicateColumns:
        raise ValueError(f"Columns assigned to multiple scaling methods: {DuplicateColumns}")

    Plan = {"Columns": {}}
    for Method, Columns in Scaling.items():
        if Columns:
            Scaler = ScalerClasses[Method]().fit(Table[Columns])

            if Method == "Standardization":
                Offsets = Scaler.mean_
                Scales = Scaler.scale_
            elif Method == "Normalization":
                Offsets = Scaler.data_min_
                Scales = [Range if Range != 0 else 1.0 for Range in Scaler.data_range_]
            elif Method == "Robust":
                Offsets = Scaler.center_
                Scales = Scaler.scale_
            else:
                Offsets = [0.0] * len(Columns)
                Scales = Scaler.scale_

            for Column, Offset, Scale in zip(Columns, Offsets, Scales):
                Plan["Columns"][Column] = {
                    "Strategy": Method,
                    "Offset": float(Offset),
                    "Scale": float(Scale),
                }

    return Plan

def ApplyScalingPlan(
    Table: pd.DataFrame,
    Plan: dict,
) -> pd.DataFrame:
    """
    Apply a fitted scaling plan to train, test or production data.

    Arg:
       Table: Source table the fitted plan is applied to.
       Plan: Scaling plan produced by FitScalingPlan or loaded with ImportJson.

    Returns:
       New DataFrame scaled with the parameters fitted on the training data; Table is not modified.

    Raises:
       ValueError: If Plan is not a dict with a Columns dict, an entry has no valid Strategy,
          Offset and Scale (Scale must not be 0), a plan column is missing in Table, or a plan
          column is not numeric.

    Notes:
       Columns that are not changed share memory with Table; only changed columns are copied.
    """
    _ValidatePlan(Plan)

    PlanColumns = Plan["Columns"]
    InvalidSpecs = [
        Column for Column, Spec in PlanColumns.items()
        if not isinstance(Spec, dict) or Spec.get("Strategy") not in SCALING_METHODS
        or not _IsNumber(Spec.get("Offset")) or not _IsNumber(Spec.get("Scale")) or not np.isfinite(Spec["Offset"])
        or not np.isfinite(Spec["Scale"]) or Spec["Scale"] == 0
    ]
    if InvalidSpecs:
        raise ValueError(f"Plan columns without a valid scaling Strategy, finite Offset and non-zero Scale: {InvalidSpecs}")

    _CheckColumns(Table, list(PlanColumns))
    NonNumericColumns = [Column for Column in PlanColumns if not pd.api.types.is_numeric_dtype(Table[Column])]
    if NonNumericColumns:
        raise ValueError(f"Columns to scale must be numeric: {NonNumericColumns}")

    Data = Table.copy(deep=False)
    for Column, Spec in PlanColumns.items():
        Data[Column] = (Data[Column] - Spec["Offset"]) / Spec["Scale"]

    return Data

#********* WOE (Weight of Evidence) *********#

def _WOECandidateBounds(
    Values: np.ndarray,
    Events: np.ndarray,
    RowCount: int,
    CoverBound: float,
) -> list[float]:
    """
    Choose the candidate upper bounds of the bins of one feature.

    The candidates are the distinct feature values of the event rows. When there are more than
    CoverBound * RowCount of them, the distinct values of all rows are cut into about
    CoverBound ** 2 * RowCount groups and the maximum of each group is used.

    Arg:
       Values: Non-null feature values.
       Events: 0/1 target of the same rows.
       RowCount: Number of rows of the whole table, nulls included.
       CoverBound: Minimum coverage of a bin.

    Returns:
       Increasing candidate bounds; empty if there are no candidates.

    Raises:
       None.
    """
    Bounds = np.unique(Values[Events == 1])
    if len(Bounds) > round(RowCount * CoverBound):
        GroupNumber = max(1, round(RowCount * CoverBound ** 2))
        Unique = np.unique(Values)
        Group = np.floor(np.arange(1, len(Unique) + 1) / (len(Unique) / GroupNumber)) + 1
        Aggregated = pd.Series(Unique).groupby(Group).agg(["count", "max"])
        if Aggregated["count"].iloc[-1] == 1:
            Aggregated = Aggregated.iloc[:-1]
        Bounds = Aggregated["max"].unique()
    return list(Bounds)

def _MergeWOEBins(
    Counts: list[int],
    Events: list[float],
    Edges: list[float],
    CoverBound: float,
    MeanDiffBound: float,
) -> None:
    """
    Merge neighbouring bins in place until every bin is large enough and differs from its neighbours.

    A bin is merged when its coverage is at most CoverBound or its event rate differs from a
    neighbour by at most MeanDiffBound; a middle bin joins the neighbour with the closer rate.
    After every merge the scan restarts from the first bin.

    Arg:
       Counts: Row count of each bin; updated.
       Events: Event count of each bin; updated.
       Edges: Upper bound of every bin except the last one; updated.
       CoverBound: Minimum coverage of a bin.
       MeanDiffBound: Minimum difference between the event rates of neighbouring bins.

    Returns:
       None.

    Raises:
       None.
    """
    Total = sum(Counts)
    Position = 0
    while Position < len(Counts) and len(Counts) > 1:
        Coverage = Counts[Position] / Total
        if Position == 0:
            Merge = Coverage <= CoverBound or abs(Events[0] / Counts[0] - Events[1] / Counts[1]) <= MeanDiffBound
            Pair = 0
        elif Position == len(Counts) - 1:
            Merge = Coverage <= CoverBound or abs(Events[Position] / Counts[Position] - Events[Position - 1] / Counts[Position - 1]) <= MeanDiffBound
            Pair = Position - 1
        else:
            Rate = Events[Position] / Counts[Position]
            ToNext = abs(Rate - Events[Position + 1] / Counts[Position + 1])
            ToPrevious = abs(Rate - Events[Position - 1] / Counts[Position - 1])
            Merge = Coverage <= CoverBound or ToPrevious <= MeanDiffBound or ToNext <= MeanDiffBound
            Pair = Position if ToNext < ToPrevious else Position - 1

        if not Merge:
            Position += 1
            continue

        Counts[Pair + 1] += Counts[Pair]
        Events[Pair + 1] += Events[Pair]
        del Counts[Pair], Events[Pair], Edges[Pair]
        Position = 0

def _WOEFeatureBins(
    Feature: str,
    Values: np.ndarray,
    Events: np.ndarray,
    CoverBound: float,
    MeanDiffBound: float,
    ZeroAdjustment: float,
) -> tuple[pd.DataFrame, list[float], list[int], list[float], int | None, float | None, float]:
    """
    Bin one feature and calculate the WOE and IV of its bins.

    Arg:
       Feature: Feature name.
       Values: Feature values of all rows; NaN is the null bin.
       Events: 0/1 target of all rows.
       CoverBound: Minimum coverage of a bin.
       MeanDiffBound: Minimum difference between the event rates of neighbouring bins.
       ZeroAdjustment: Count used instead of 0 when a bin has no events or no non-events.

    Returns:
       Tuple of (Binning, Edges, BinIds, BinWOE, NullBin, NullWOE, IV). Binning has one row per
       interval (plus the null row); Edges, BinIds and BinWOE describe the intervals; NullBin and
       NullWOE are None when the feature has no nulls.

    Raises:
       None.
    """
    IsNull = np.isnan(Values)
    NonNull, NonNullEvents = Values[~IsNull], Events[~IsNull]
    RowCount = len(Values)
    NullCount, NullEvents = int(IsNull.sum()), float(Events[IsNull].sum())

    if len(NonNull):
        Bounds = _WOECandidateBounds(NonNull, NonNullEvents, RowCount, CoverBound)
        Edges = [float(Bound) for Bound in Bounds[:-1]]
        Index = np.searchsorted(Edges, NonNull, side="left")
        Counts = np.bincount(Index, minlength=len(Edges) + 1).astype(int).tolist()
        EventCounts = np.bincount(Index, weights=NonNullEvents, minlength=len(Edges) + 1).tolist()
        _MergeWOEBins(Counts, EventCounts, Edges, CoverBound, MeanDiffBound)
    else:
        Edges, Counts, EventCounts = [], [], []

    IntervalCount = len(Counts)
    Rows = [
        {"LowerBound": (-np.inf if Position == 0 else Edges[Position - 1]), "UpperBound": (np.inf if Position == IntervalCount - 1 else Edges[Position]),
         "Freq": Counts[Position], "Target": EventCounts[Position], "Bin": Position + 1}
        for Position in range(IntervalCount)
    ]

    NullBin = None
    if NullCount:
        NullBin = IntervalCount + 1
        if NullCount < round(RowCount * CoverBound) and IntervalCount:
            # A small null group joins the bin whose event rate is closest to its own.
            NullRate = NullEvents / NullCount
            Differences = [abs(EventCounts[Position] / Counts[Position] - NullRate) for Position in range(IntervalCount)]
            NullBin = int(np.argmin(Differences)) + 1
        Rows.append({"LowerBound": np.nan, "UpperBound": np.nan, "Freq": NullCount, "Target": NullEvents, "Bin": NullBin})

    Binning = pd.DataFrame(Rows)
    Binning["NonTarget"] = Binning["Freq"] - Binning["Target"]

    BinTable = Binning.groupby("Bin").agg(BinFreq=("Freq", "sum"), BinTarget=("Target", "sum")).reset_index()
    BinTable["BinNonTarget"] = BinTable["BinFreq"] - BinTable["BinTarget"]
    BinTable["BinPercTarget"] = np.where(BinTable["BinTarget"] == 0, ZeroAdjustment, BinTable["BinTarget"]) / BinTable["BinTarget"].sum()
    BinTable["BinPercNonTarget"] = np.where(BinTable["BinNonTarget"] == 0, ZeroAdjustment, BinTable["BinNonTarget"]) / BinTable["BinNonTarget"].sum()
    BinTable["WOE"] = np.log(BinTable["BinPercNonTarget"] / BinTable["BinPercTarget"])
    BinTable["IV"] = (BinTable["BinPercNonTarget"] - BinTable["BinPercTarget"]) * BinTable["WOE"]

    # The lowest third of the bins by WOE is class -1 and the highest third is class 1.
    BinTable["BinClass"] = 0
    Proportion = round(len(BinTable) / 3)
    if Proportion >= 1:
        Order = np.argsort(BinTable["WOE"].to_numpy(), kind="stable")
        BinTable.loc[Order[:Proportion], "BinClass"] = -1
        BinTable.loc[Order[-Proportion:], "BinClass"] = 1

    Binning = Binning.merge(BinTable, on="Bin", how="left")
    Binning["Coverage"] = Binning["Freq"] / Binning["Freq"].sum()
    Binning["Rate"] = Binning["Target"] / Binning["Freq"]
    Binning.insert(0, "ColumnName", Feature)
    Binning = Binning[[
        "ColumnName", "Bin", "LowerBound", "UpperBound", "Coverage", "Rate", "Freq", "Target", "NonTarget",
        "BinFreq", "BinTarget", "BinNonTarget", "BinPercTarget", "BinPercNonTarget", "WOE", "IV", "BinClass",
    ]]
    for Column in ("Freq", "Target", "NonTarget", "BinFreq", "BinTarget", "BinNonTarget", "BinClass"):
        Binning[Column] = Binning[Column].astype(int)

    WOEOfBin = dict(zip(BinTable["Bin"], BinTable["WOE"]))
    BinIds = list(range(1, IntervalCount + 1))
    return (
        Binning, Edges, BinIds, [float(WOEOfBin[BinId]) for BinId in BinIds],
        NullBin, None if NullBin is None else float(WOEOfBin[NullBin]), float(BinTable["IV"].sum()),
    )

def FitWOEPlan(
    Table: pd.DataFrame,
    Target: str,
    Features: list[str] | None = None,
    CoverBound: float = 0.01,
    MeanDiffBound: float = 0.01,
    IVBound: float = 0.02,
    ZeroAdjustment: float = 0.5,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Bin the features of a binary target, calculate WOE and IV, and fit a reusable WOE plan.

    Weight of Evidence is ln(share of non-events / share of events) of a bin; Information Value
    (IV) sums (share of non-events - share of events) * WOE over the bins of a feature. Bins are
    built from the distinct values of the event rows, then neighbouring bins are merged until each
    bin covers more than CoverBound of the rows and neighbouring event rates differ by more than
    MeanDiffBound. Null values form their own bin; a null group smaller than CoverBound joins the
    bin with the closest event rate. The lowest third of the bins by WOE gets BinClass -1 and the
    highest third gets 1.

    Arg:
       Table: Training table; fit it on Train only. Features must be numeric and finite, and the
          Target must contain only 0 and 1 without nulls.
       Target: Binary target column.
       Features: Features to bin. If None, every column except Target is used.
       CoverBound: Minimum share of the rows a bin must cover, between 0 and 1.
       MeanDiffBound: Minimum difference between the event rates of neighbouring bins.
       IVBound: A feature is selected when its IV is above this value.
       ZeroAdjustment: Count used instead of 0 when a bin has no events or no non-events, so that
          WOE and IV stay finite. Bins without zero counts are not affected.

    Returns:
       Tuple of (Plan, Binning, FeatureSelection).
         Plan: JSON-serializable {"Columns": {Feature: {"Cuts", "Bins", "WOE", "NullBin", "NullWOE"}}}
            of the selected features only. Interval j covers (Cuts[j - 1], Cuts[j]], the first one has
            no lower bound and the last one no upper bound. Apply it with ApplyWOEPlan.
         Binning: One row per bin of every feature with ColumnName, Bin, LowerBound, UpperBound, Coverage,
            Rate, Freq, Target, NonTarget, BinFreq, BinTarget, BinNonTarget, BinPercTarget,
            BinPercNonTarget, WOE, IV and BinClass. A merged null group appears as an extra row with
            the Bin of the bin it joined.
         FeatureSelection: ColumnName, IV and Selection (1 if IV > IVBound) sorted by IV descending.

    Raises:
       TypeError: If Table, Target or Features have the wrong type.
       ValueError: If a column is missing, not numeric, or contains inf values, Target is not 0/1 or
          has nulls, or a bound is out of range.

    Notes:
       The binning rules follow the original WOETransformation function; infinite WOE values are
          replaced through ZeroAdjustment, and a feature without any non-null value gets IV 0.
    """
    if not 0 < CoverBound < 1:
        raise ValueError("CoverBound must be between 0 and 1.")
    if MeanDiffBound < 0 or IVBound < 0:
        raise ValueError("MeanDiffBound and IVBound must be 0 or greater.")
    if ZeroAdjustment <= 0:
        raise ValueError("ZeroAdjustment must be greater than 0.")
    Features = _ValidateTargetAndFeatures(Table, Target, Features, "Binary")
    if Table[Target].isna().any():
        raise ValueError(f"Target must not contain nulls: {Target}")

    EventValues = Table[Target].to_numpy(dtype=float)
    Binnings, Selection, PlanColumns = [], [], {}
    for Feature in Features:
        Values = Table[Feature].to_numpy(dtype=float, na_value=np.nan)
        if np.isnan(Values).all():
            _Alert(f"{Feature} has only null values; IV is 0.")
            Selection.append({"ColumnName": Feature, "IV": 0.0, "Selection": 0})
            continue

        Binning, Edges, BinIds, BinWOE, NullBin, NullWOE, IV = _WOEFeatureBins(
            Feature, Values, EventValues, CoverBound, MeanDiffBound, ZeroAdjustment,
        )
        Binnings.append(Binning)
        Selection.append({"ColumnName": Feature, "IV": IV, "Selection": int(IV > IVBound)})
        if IV > IVBound:
            PlanColumns[Feature] = {"Cuts": Edges, "Bins": BinIds, "WOE": BinWOE, "NullBin": NullBin, "NullWOE": NullWOE}

    FeatureSelection = pd.DataFrame(Selection, columns=["ColumnName", "IV", "Selection"]).sort_values("IV", ascending=False, kind="stable").reset_index(drop=True)
    Binning = pd.concat(Binnings, ignore_index=True) if Binnings else pd.DataFrame()

    _Step(f"WOE: {len(PlanColumns)} of {len(Features)} features selected (IV > {IVBound}): {list(PlanColumns)}")
    return {"Columns": PlanColumns}, Binning, FeatureSelection

def _ValidateWOEPlan(
    Plan: dict,
) -> None:
    """
    Check the entries of a WOE plan.

    Arg:
       Plan: Plan produced by FitWOEPlan or loaded with ImportJson.

    Returns:
       None.

    Raises:
       ValueError: If Plan has no Columns dict or an entry is not a valid WOE entry.
    """
    _ValidatePlan(Plan)
    Invalid = []
    for Column, Spec in Plan["Columns"].items():
        Valid = isinstance(Spec, dict) and all(isinstance(Spec.get(Key), list) for Key in ("Cuts", "Bins", "WOE"))
        if Valid:
            Valid = (
                len(Spec["Bins"]) == len(Spec["WOE"]) == len(Spec["Cuts"]) + 1
                and all(_IsNumber(Value) and np.isfinite(Value) for Value in [*Spec["Cuts"], *Spec["WOE"]])
                and all(Low < High for Low, High in zip(Spec["Cuts"], Spec["Cuts"][1:]))
                and (Spec.get("NullBin") is None) == (Spec.get("NullWOE") is None)
                and (Spec.get("NullWOE") is None or (_IsNumber(Spec["NullWOE"]) and np.isfinite(Spec["NullWOE"])))
            )
        if not Valid:
            Invalid.append(Column)
    if Invalid:
        raise ValueError(f"Plan columns without a valid WOE entry (Cuts, Bins, WOE, NullBin, NullWOE): {Invalid}")

def ApplyWOEPlan(
    Table: pd.DataFrame,
    Plan: dict,
) -> pd.DataFrame:
    """
    Add the bin and the WOE of every planned feature to a table.

    Arg:
       Table: Source table containing the planned features.
       Plan: WOE plan produced by FitWOEPlan or loaded with ImportJson.

    Returns:
       New DataFrame with the original columns plus <Feature>_Bin and <Feature>_WOE for each
       planned feature; Table itself is not modified.

    Raises:
       ValueError: If Plan is not a valid WOE plan, a planned feature is missing or not numeric, or
          Table has duplicated column names.

    Notes:
       Null values that did not exist when the plan was fitted get Bin 0 and WOE 0 (neutral), and
          a message lists the features.
       Existing <Feature>_Bin and <Feature>_WOE columns are replaced.
    """
    _ValidateWOEPlan(Plan)
    _CheckUniqueColumns(Table)
    PlanColumns = Plan["Columns"]
    _CheckColumns(Table, list(PlanColumns))
    NonNumeric = [Column for Column in PlanColumns if not pd.api.types.is_numeric_dtype(Table[Column])]
    if NonNumeric:
        raise ValueError(f"Columns to transform must be numeric: {NonNumeric}")

    Data = Table.copy(deep=False)
    for Column, Spec in PlanColumns.items():
        Values = Table[Column].to_numpy(dtype=float, na_value=np.nan)
        IsNull = np.isnan(Values)
        Index = np.searchsorted(np.asarray(Spec["Cuts"], dtype=float), np.where(IsNull, 0.0, Values), side="left")
        BinIds = np.asarray(Spec["Bins"], dtype=np.int64)[Index]
        WOE = np.asarray(Spec["WOE"], dtype=float)[Index]

        if Spec.get("NullBin") is None:
            BinIds[IsNull], WOE[IsNull] = 0, 0.0
            if IsNull.any():
                _Alert(f"{Column}: {int(IsNull.sum())} null values were not seen when the plan was fitted; they get Bin 0 and WOE 0.")
        else:
            BinIds[IsNull], WOE[IsNull] = Spec["NullBin"], Spec["NullWOE"]

        Data[f"{Column}_Bin"] = BinIds
        Data[f"{Column}_WOE"] = WOE

    return Data
