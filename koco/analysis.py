from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ._helpers import _Alert, _CheckColumns, _Done, _Info, _OutputPath, _Rate, _Step

__all__ = [
    "AggregationReport", "ColumnBinning", "SampleBinning", "DataSummary", "CrossCheck",
    "OutlierReport", "OutlierCountByVariable", "TargetDivergenceByColumn", "RepresentationReport", "DrawChart",
]

#********* Analysis Functions *********#

_IQR_MULTIPLIER = 1.5
_ZSCORE_LIMIT = 3
# Statistics of DataSummary that RepresentationReport can compare.
REPRESENTATION_STATISTICS = (
    "AvgValue", "ModValue", "MedValue", "NullRate", "ZeroRate", "PositiveRate", "NegativeRate",
    "MinValue", "MaxValue", "StdValue", "P01Value", "P05Value", "P95Value", "P99Value",
)
_DEFAULT_REPRESENTATION_STATISTICS = REPRESENTATION_STATISTICS[:7]
_QUOTED_TOKEN = re.compile(r"""'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`[^`]*`""")

def AggregationReport(
    Table: pd.DataFrame,
    GroupColumns: list[str],
    Agg: dict[str, str | list[str] | tuple[str, ...]] | None = None,
    Sort: list[tuple[str, str | bool]] | None = None,
) -> pd.DataFrame:
    """
    Build a grouped aggregation report with observation rates.

    Arg:
       Table: Source table to aggregate.
       GroupColumns: Columns used for grouping.
       Agg: Aggregation mapping such as {"Amount": ["mean", "sum"]}; output columns are named
          <Func><Column>, e.g. meanAmount.
       Sort: Optional (Column, Direction) tuples; Direction is "Asc" / "Desc" or a bool
          (True = ascending). Without Sort the report is ordered by Observation descending.

    Returns:
       DataFrame containing grouped counts, aggregations, and rate values.

    Raises:
       ValueError: If Table is empty, GroupColumns is empty, or columns or sort rules are invalid.
       TypeError: If a Sort item is not a (Column, Direction) tuple.
    """
    if Table.empty:
        raise ValueError("Table must not be empty.")

    if not GroupColumns:
        raise ValueError("GroupColumns must contain at least one column name.")

    _CheckColumns(Table, list(GroupColumns) + list(Agg or {}))

    Report = Table.groupby(GroupColumns, dropna=False).size().reset_index(name="Observation").sort_values(["Observation"], ascending=[False]).reset_index(drop=True)

    if Agg is not None:
        Aggregations = {}
        for Column, FuncSpec in Agg.items():
            FuncList = FuncSpec if isinstance(FuncSpec, (list, tuple)) else [FuncSpec]
            for Func in FuncList:
                if not isinstance(Func, str) or not Func:
                    raise ValueError(f"Invalid aggregation function for column {Column}: {Func}")
                Aggregations[f"{Func}{Column}"] = (Column, Func)

        Report = Report.merge(Table.groupby(GroupColumns, dropna=False).agg(**Aggregations).reset_index(), on=GroupColumns, how="left")

    Report["Rate"] = _Rate(Report["Observation"])

    if Sort:
        SortColumns: list[str] = []
        SortAscending: list[bool] = []

        for Item in Sort:
            if not isinstance(Item, tuple):
                raise TypeError(f"Sort items must be (Column, Direction) tuples: {Item!r}")
            if len(Item) != 2:
                raise ValueError(f"Sort tuple must have 2 items: {Item}")

            Column, Direction = Item
            if Column not in Report.columns:
                raise ValueError(f"Sort column not found in report: {Column}")

            if isinstance(Direction, bool):
                Ascending = Direction
            elif isinstance(Direction, str):
                DirectionLower = Direction.strip().lower()
                if DirectionLower in ["asc", "ascending"]:
                    Ascending = True
                elif DirectionLower in ["desc", "descending"]:
                    Ascending = False
                else:
                    raise ValueError(
                        f"Sort direction must be Asc/Desc or bool for column {Column}: {Direction}"
                    )
            else:
                raise ValueError(
                    f"Sort direction must be string or bool for column {Column}: {Direction}"
                )

            SortColumns.append(Column)
            SortAscending.append(Ascending)

        Report = Report.sort_values(by=SortColumns, ascending=SortAscending).reset_index(drop=True)

    return Report

def _RangeLabels(
    Edges: Sequence[float],
) -> list[str]:
    """
    Build interval labels for consecutive bin edges.

    Arg:
       Edges: Increasing bin edges.

    Returns:
       One label per interval; the first interval is closed on both sides, the rest on the right.

    Raises:
       None.
    """
    return [
        f"{'[' if Index == 0 else '('}{round(Edges[Index], 2)} - {round(Edges[Index + 1], 2)}]"
        for Index in range(len(Edges) - 1)
    ]

def ColumnBinning(
    Table: pd.DataFrame,
    Column: str,
    Bins: int | tuple[float, float, int] | list[float] | None = None,
) -> pd.DataFrame:
    """
    Bin a numeric column into equal-width or custom ranges and return its distribution.

    Arg:
       Table: Source table containing the target column.
       Column: Numeric column to bin.
       Bins: None for 10 equal-width bins, an int for that many equal-width bins, a
          (Start, Stop, Count) tuple for Count equal-width bins between Start and Stop, or a
          strictly increasing list of int / float edges. Values outside a tuple or list range
          are not counted.

    Returns:
       DataFrame with bin labels, observation counts, and rates. A constant column is reported
       as a single bin.

    Raises:
       ValueError: If Column is missing, not numeric, has no non-null values, or Bins has an
          unsupported format.
    """
    _CheckColumns(Table, [Column])
    if not pd.api.types.is_numeric_dtype(Table[Column]) or pd.api.types.is_bool_dtype(Table[Column]):
        raise ValueError(f"Column must be numeric: {Column}")

    Values = Table[Column].dropna()
    if Values.empty:
        raise ValueError(f"Column {Column} has no non-null values.")

    MinValue, MaxValue = Values.min(), Values.max()
    _Step(f"{Column} min: {MinValue}, max: {MaxValue}")

    if Bins is None or (isinstance(Bins, int) and not isinstance(Bins, bool)):
        BinCount = 10 if Bins is None else Bins
        if BinCount < 1:
            raise ValueError("Bins must be a positive integer.")
        # A constant column would produce identical edges, so it is reported as one bin below.
        Edges = None if MinValue == MaxValue else np.linspace(MinValue, MaxValue, BinCount + 1)
    elif isinstance(Bins, tuple) and len(Bins) == 3:
        Start, Stop, Count = Bins
        if not isinstance(Count, int) or isinstance(Count, bool) or Count < 1 or not Start < Stop:
            raise ValueError("Bins tuple must be (Start, Stop, Count) with Start < Stop and a positive int Count.")
        Edges = np.linspace(Start, Stop, Count + 1)
    elif isinstance(Bins, list) and len(Bins) >= 2 and all(
        isinstance(Edge, (int, float, np.number)) and not isinstance(Edge, bool) for Edge in Bins
    ):
        if not (np.diff(np.asarray(Bins, dtype=float)) > 0).all():
            raise ValueError("Bins edges must be strictly increasing.")
        Edges = Bins
    else:
        raise ValueError("Bins must be None, a positive int, a (Start, Stop, Count) tuple, or a list of at least 2 numeric edges.")

    if Edges is None:
        BinDist = pd.DataFrame({f"{Column}Range": _RangeLabels([MinValue, MaxValue]), "Observation": [len(Values)]})
    else:
        BinTable = pd.cut(Table[Column], bins=Edges, labels=_RangeLabels(Edges), include_lowest=True)
        BinDist = BinTable.value_counts().sort_index().reset_index()
        BinDist.columns = [f"{Column}Range", "Observation"]

    BinDist["Rate"] = _Rate(BinDist["Observation"])

    return BinDist

def SampleBinning(
    Table: pd.DataFrame,
    Column: str,
    Bins: int = 10,
) -> pd.DataFrame:
    """
    Create equal-sample bins for a sorted column.

    Arg:
       Table: Source table containing the target column.
       Column: Column to split into sample-based bins.
       Bins: Number of bins to create.

    Returns:
       DataFrame with observation counts, min and max values, and rates per bin.

    Raises:
       ValueError: If Column is missing, Bins is invalid, or no usable values exist.
    """
    _CheckColumns(Table, [Column])
    if not isinstance(Bins, int) or Bins <= 0:
        raise ValueError("Bins must be a positive integer.")

    NullCount = Table[Column].isnull().sum()
    if NullCount > 0:
        _Alert(f"{Column} column has {NullCount} null values. They will be excluded from binning.")

    WorkingTable = Table[[Column]].dropna().sort_values(Column).reset_index(drop=True)

    SampleCount = len(WorkingTable)
    if SampleCount == 0:
        raise ValueError(f"Column {Column} has no non-null values.")

    BinSize = int(np.ceil(SampleCount / Bins))
    WorkingTable["Bin"] = (WorkingTable.index // BinSize) + 1
    WorkingTable["Bin"] = WorkingTable["Bin"].clip(upper=Bins)

    Report = (
        WorkingTable.groupby("Bin", as_index=False)
        .agg(
            Observation=(Column, "count"),
            MinValue=(Column, "min"),
            MaxValue=(Column, "max"),
        )
        .rename(columns={"MinValue": f"Min{Column}", "MaxValue": f"Max{Column}"})
    )

    Report["Rate"] = _Rate(Report["Observation"])

    return Report

def DataSummary(
    Table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize each column in a DataFrame with descriptive statistics.

    Arg:
       Table: Source table to summarize.

    Returns:
       DataFrame with one row per column containing type, count, null, and distribution metrics.
       Numeric statistics are NaN for non-numeric columns.

    Raises:
       None.
    """
    NumericCols = set(Table.select_dtypes(include=np.number).columns)
    TotalCount = len(Table)
    Rows = []

    for Col in Table.columns:
        Series        = Table[Col]
        IsNumeric     = Col in NumericCols
        NullCount     = int(Series.isnull().sum())
        DistinctCount = int(Series.nunique())
        ModeValues    = Series.mode()
        ModValue      = ModeValues.iloc[0] if not ModeValues.empty else np.nan

        if IsNumeric:
            MinValue      = Series.min()
            MaxValue      = Series.max()
            AvgValue      = Series.mean()
            MedValue      = Series.median()
            StdValue      = Series.std()
            VarValue      = Series.var()
            Percentiles   = Series.quantile([0.01, 0.05, 0.95, 0.99])
            P01Value, P05Value, P95Value, P99Value = Percentiles.tolist()
            ZeroCount     = int((Series == 0).sum())
            PositiveCount = int((Series > 0).sum())
            NegativeCount = int((Series < 0).sum())
        else:
            try:
                MinValue = Series.min()
                MaxValue = Series.max()
            except TypeError:
                MinValue = np.nan
                MaxValue = np.nan
            AvgValue  = MedValue = StdValue = VarValue = P01Value = P05Value = P95Value = P99Value = np.nan
            ZeroCount = PositiveCount = NegativeCount = np.nan

        Rows.append({
            "ColumnName"    : Col,
            "DataType"      : str(Series.dtype),
            "TotalCount"    : TotalCount,
            "DistinctCount" : DistinctCount,
            "MinValue"      : MinValue,
            "MaxValue"      : MaxValue,
            "AvgValue"      : AvgValue,
            "ModValue"      : ModValue,
            "MedValue"      : MedValue,
            "P01Value"      : P01Value,
            "P05Value"      : P05Value,
            "P95Value"      : P95Value,
            "P99Value"      : P99Value,
            "StdValue"      : StdValue,
            "VarValue"      : VarValue,
            "NullCount"     : NullCount,
            "NullRate"      : round(NullCount / TotalCount, 4) if TotalCount > 0 else np.nan,
            "ZeroCount"     : ZeroCount,
            "ZeroRate"      : round(ZeroCount / TotalCount, 4) if IsNumeric and TotalCount > 0 else np.nan,
            "PositiveCount" : PositiveCount,
            "PositiveRate"  : round(PositiveCount / TotalCount, 4) if IsNumeric and TotalCount > 0 else np.nan,
            "NegativeCount" : NegativeCount,
            "NegativeRate"  : round(NegativeCount / TotalCount, 4) if IsNumeric and TotalCount > 0 else np.nan,
        })

    return pd.DataFrame(Rows)

def _NormalizeRule(
    Expr: str,
) -> str:
    """
    Normalize a SQL-like rule into a pandas-evaluable expression.

    Quoted text and backticked column names are left untouched.

    Arg:
       Expr: Raw rule expression.

    Returns:
       Normalized expression string.

    Raises:
       None.
    """
    Protected: list[str] = []

    def Protect(Match: re.Match) -> str:
        Protected.append(Match.group(0))
        return f"__KocoToken{len(Protected) - 1}__"

    Normalized = _QUOTED_TOKEN.sub(Protect, Expr.strip())
    Normalized = re.sub(r"\bAND\b", "and", Normalized, flags=re.IGNORECASE)
    Normalized = re.sub(r"\bOR\b", "or", Normalized, flags=re.IGNORECASE)
    Normalized = re.sub(r"\bNOT\b", "not", Normalized, flags=re.IGNORECASE)
    Normalized = re.sub(r"(?<![=!<>])=(?!=)", "==", Normalized)
    Normalized = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NOT\s+NULL\b", r"\1.notna()", Normalized, flags=re.IGNORECASE)
    Normalized = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\b", r"\1.isna()", Normalized, flags=re.IGNORECASE)
    for Index, Original in enumerate(Protected):
        Normalized = Normalized.replace(f"__KocoToken{Index}__", Original)
    return Normalized

def CrossCheck(
    Table: pd.DataFrame,
    CheckList: dict[str, str],
) -> pd.DataFrame:
    """
    Evaluate boolean rules on a table and count matching rows.

    Arg:
       Table: Source table used for rule evaluation.
       CheckList: Mapping of control names to SQL-like rule expressions, e.g.
          {"NegativeAge": "Age < 0 AND Segment IS NOT NULL"}. Wrap column names that contain
          spaces in backticks, e.g. "`Net Income` IS NULL".

    Returns:
       DataFrame containing each rule and its matching row count.

    Raises:
       ValueError: If the rule list is empty or a rule is invalid.

    Notes:
       Rules are evaluated as code by DataFrame.eval; use only rules written by a trusted person.
       AND, OR, NOT, = and IS [NOT] NULL are translated; text inside quotes is never changed.
    """
    if not CheckList:
        raise ValueError("CheckList must contain at least one rule.")

    Rows = []
    for ControlName, Rule in CheckList.items():
        if not isinstance(Rule, str) or not Rule.strip():
            raise ValueError(f"Rule must be a non-empty string for control: {ControlName}")

        Expr = _NormalizeRule(Rule)
        _Step(f"{ControlName} Normalized: {Expr}")

        try:
            Mask = Table.eval(Expr, engine="python")
        except Exception as Error:
            raise ValueError(f"Failed to evaluate rule for '{ControlName}': {Rule!r} ({Error})") from Error

        if not pd.api.types.is_bool_dtype(Mask):
            raise ValueError(f"Rule does not evaluate to a boolean condition for '{ControlName}': {Rule!r}")

        Rows.append({
            "Control": ControlName,
            "Detail" : Rule.strip(),
            "Count"  : int(Mask.fillna(False).sum()),
        })

    return pd.DataFrame(Rows)

def _OutlierStats(
    Series: pd.Series,
    Method: str,
) -> dict:
    """
    Calculate the outlier bounds, count and rate of one non-null numeric series.

    Arg:
       Series: Non-null numeric values.
       Method: "IQR" or "ZScore".

    Returns:
       Dict with LowerBound, UpperBound, OutlierCount and OutlierRate.

    Raises:
       None.
    """
    Total = len(Series)
    if Method == "IQR":
        Q1, Q3 = Series.quantile(0.25), Series.quantile(0.75)
        IQR = Q3 - Q1
        Lower, Upper = Q1 - _IQR_MULTIPLIER * IQR, Q3 + _IQR_MULTIPLIER * IQR
        Count = int(((Series < Lower) | (Series > Upper)).sum())
    else:
        Std = Series.std()
        if Std == 0:
            Lower = Upper = float(Series.iloc[0]) if Total > 0 else np.nan
            Count = 0
        else:
            Mean = Series.mean()
            Lower, Upper = Mean - _ZSCORE_LIMIT * Std, Mean + _ZSCORE_LIMIT * Std
            Count = int((((Series - Mean) / Std).abs() > _ZSCORE_LIMIT).sum())

    return {
        "LowerBound"  : round(Lower, 4),
        "UpperBound"  : round(Upper, 4),
        "OutlierCount": Count,
        "OutlierRate" : round(Count / Total, 4) if Total > 0 else np.nan,
    }

def OutlierReport(
    Table: pd.DataFrame,
    Method: str | None = None,
) -> pd.DataFrame:
    """
    Report the outliers of each numeric column.

    Arg:
       Table: Source table containing the numeric columns.
       Method: None for both methods, "IQR" for the 1.5x IQR rule, or "ZScore" for the |z| > 3 rule.

    Returns:
       DataFrame with the bounds, outlier count, and outlier rate per numeric column.

    Raises:
       ValueError: If Method is not None, "IQR" or "ZScore", or Table has no numeric columns.
    """
    if Method not in (None, "IQR", "ZScore"):
        raise ValueError(f"Method must be None, 'IQR' or 'ZScore', got {Method!r}.")

    Columns = Table.select_dtypes(include=np.number).columns.tolist()
    if not Columns:
        raise ValueError("Table has no numeric columns.")

    Methods = ["IQR", "ZScore"] if Method is None else [Method]
    Rows = []
    for Col in Columns:
        Series = Table[Col].dropna()
        Row = {"Column": Col}
        for Name in Methods:
            Stats = _OutlierStats(Series, Name)
            if Method is None:
                Row.update({f"{Name}{Key}": Value for Key, Value in Stats.items()})
            else:
                Row.update({"Method": Name, **Stats})
        Rows.append(Row)

    return pd.DataFrame(Rows).sort_values("Column", ascending=False).reset_index(drop=True)

def OutlierCountByVariable(
    Table: pd.DataFrame,
    VariableName: str,
    Thresholds: list[tuple[str, float]] | None = None,
    AddStats: bool = False,
) -> None:
    """
    Print how many observations lie beyond each threshold of a numeric variable.

    Thresholds below the average count observations at or below them; thresholds at or above
    the average count observations at or above them. Useful for choosing capping values for
    ApplyOutlierPlan.

    Arg:
       Table: Source table containing the numeric column.
       VariableName: The name of the variable to analyze.
       Thresholds: (Label, Value) pairs to check, e.g. [("Cap", 250000)].
       AddStats: Add the statistical thresholds P1, P5, P95, P99 (percentiles), Mode and Median;
          they are always added when Thresholds is empty.

    Returns:
       None. The counts are printed.

    Raises:
       ValueError: If VariableName is not a numeric column of Table or has no non-null values.
    """
    _CheckColumns(Table, [VariableName])
    if not pd.api.types.is_numeric_dtype(Table[VariableName]) or pd.api.types.is_bool_dtype(Table[VariableName]):
        raise ValueError(f"Column must be numeric: {VariableName}")

    Series = Table[VariableName].dropna()
    if Series.empty:
        raise ValueError(f"Column {VariableName} has no non-null values.")

    NullCount = int(Table[VariableName].isnull().sum())
    RowCount = len(Table)
    Average = float(Series.mean())

    Thresholds = list(Thresholds or [])
    if not Thresholds or AddStats:
        Thresholds += [
            ("P1", float(Series.quantile(0.01))),
            ("P5", float(Series.quantile(0.05))),
            ("P95", float(Series.quantile(0.95))),
            ("P99", float(Series.quantile(0.99))),
            ("Mode", float(Series.mode().iloc[0])),
            ("Median", float(Series.median())),
        ]

    Thresholds = sorted(Thresholds, key=lambda Item: Item[1])
    MaxLabelLen = max(len(Label) for Label, _ in Thresholds)
    MaxValueLen = max(len(f"{Value:,.2f}") for _, Value in Thresholds)

    print(f"Variable: {VariableName}")
    print(f"Variable has {NullCount} ({NullCount / RowCount:.2%}) nulls")
    print(f"Average: {Average:,.2f}")
    print()
    print("Observation count out of thresholds:")
    for Label, Value in Thresholds:
        PrintLabel = f"{Label} ({Value:,.2f})"
        if Value < Average:
            Count = (Series <= Value).sum()
            print(f"Observation count equal or below {PrintLabel:<{MaxLabelLen + MaxValueLen + 3}}: {Count:,}")
        else:
            Count = (Series >= Value).sum()
            print(f"Observation count equal or above {PrintLabel:<{MaxLabelLen + MaxValueLen + 3}}: {Count:,}")

def TargetDivergenceByColumn(
    Table: pd.DataFrame,
    Target: str,
    Divergence: str,
    Alpha: float = 0.05,
    EffectThreshold: float = 0.01,
    MinObservationSize: int = 1000,
) -> dict:
    """
    Test whether the target distribution differs across groups using Kruskal-Wallis.

    Rows where Target or Divergence is null are excluded from the test.

    Arg:
       Table: Source table containing the target and grouping columns.
       Target: Numeric target column whose distributions are compared.
       Divergence: Column defining the groups to compare.
       Alpha: Significance level for the p-value, between 0 and 1.
       EffectThreshold: Minimum epsilon-squared effect size for a practical difference.
       MinObservationSize: Minimum number of observations required in every group.

    Returns:
       Dict with the keys:
         Statistic: Kruskal-Wallis H statistic.
         PValue: Probability of a difference at least this large arising by chance.
         EpsilonSquared: Effect size of the difference.
         GroupCount: Number of groups compared.
         ObservationCount: Number of rows used in the test.
         GroupSizes: Observation count per group.
         Significant: PValue < Alpha.
         Divergent: True when both the significance and the effect size conditions are met.

    Raises:
       ImportError: If scipy is not installed.
       ValueError: If a column is missing, Target is not numeric or contains inf values,
          Alpha / EffectThreshold / MinObservationSize are invalid, fewer than 2 groups exist,
          a group is smaller than MinObservationSize, or Target has a single distinct value.

    Notes:
       A different target distribution does not prove that a separate model is needed: groups can
          have different target levels while the feature-target relationship stays the same, so
          the result is supporting evidence only.

    Notes (TR):
       Hedef dağılımlarının farklı olması ayrı model gerektiğini kanıtlamaz. Gruplar farklı hedef
       seviyelerine sahip olabilir ama özellik-hedef ilişkisi aynı olabilir. Bu nedenle bu
       fonksiyon yalnızca destekleyici bilgi sağlar.
       PValue: Gözlenen farkın tesadüfen oluşma ihtimaline ilişkin test sonucu.
       EpsilonSquared: Farkın büyüklüğü.
       Significant: PValue < Alpha.
       Divergent: Hem anlamlılık hem etki büyüklüğü koşulu sağlanmış mı?
       GroupSizes: Grupların örneklem büyüklükleri.
    """
    from scipy.stats import kruskal

    _CheckColumns(Table, [Target, Divergence])
    if not pd.api.types.is_numeric_dtype(Table[Target]):
        raise ValueError(f"Target must be numeric: {Target}")
    if not np.isfinite(Table[Target].dropna().to_numpy()).all():
        raise ValueError(f"Target contains inf or -inf values: {Target}")
    if not 0 < Alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if EffectThreshold < 0:
        raise ValueError("EffectThreshold must be 0 or greater.")
    if MinObservationSize < 2:
        raise ValueError("MinObservationSize must be at least 2.")

    Data = Table[[Target, Divergence]].dropna()
    GroupSizes = Data.groupby(Divergence).size()
    Groups = [SubGroup[Target].to_numpy() for _, SubGroup in Data.groupby(Divergence)]

    if len(Groups) < 2:
        raise ValueError(f"At least 2 valid groups are required for '{Divergence}'.")
    SmallGroups = GroupSizes[GroupSizes < MinObservationSize].index.tolist()
    if SmallGroups:
        raise ValueError(f"Groups need at least {MinObservationSize} observations: {SmallGroups}")
    if Data[Target].nunique() <= 1:
        raise ValueError(f"Target must contain at least 2 distinct values: {Target}")

    Statistic, PValue = kruskal(*Groups)
    GroupCount = len(Groups)
    ObservationCount = len(Data)
    EpsilonSquared = max(0.0, (float(Statistic) - GroupCount + 1) / (ObservationCount - GroupCount))

    return {
        "Statistic": round(float(Statistic), 4),
        "PValue": round(float(PValue), 6),
        "EpsilonSquared": round(EpsilonSquared, 4),
        "GroupCount": GroupCount,
        "ObservationCount": ObservationCount,
        "GroupSizes": GroupSizes.to_dict(),
        "Significant": bool(PValue < Alpha),
        "Divergent": bool(PValue < Alpha and EpsilonSquared >= EffectThreshold),
    }

def RepresentationReport(
    Population: pd.DataFrame,
    Sample: pd.DataFrame,
    Statistics: list[str] | None = None,
    Detail: bool = False,
) -> pd.DataFrame:
    """
    Measure how well a sample (for example Train) represents a population.

    For every numeric column that exists in both tables, the DataSummary statistics of the sample
    are compared with those of the population. The relative difference r = (sample - population) /
    population of each statistic is squared and averaged, and the representation power of the
    column is 1 - sqrt(mean(r ** 2)). Statistics are rounded to 6 decimals first.

    Arg:
       Population: Table of the whole population (for example all customers).
       Sample: Table of the sample (for example Train).
       Statistics: DataSummary statistics to compare, a subset of REPRESENTATION_STATISTICS. None uses
          AvgValue, ModValue, MedValue, NullRate, ZeroRate, PositiveRate and NegativeRate.
       Detail: If True, the relative difference of every statistic is returned as well.

    Returns:
       DataFrame sorted by Rep descending with ColumnName, Rep and, when Detail is True, one column
       per statistic. Rep is 1 for identical statistics and gets smaller as they diverge (it can be
       negative). The mean Rep of all columns is printed as the representation power of the sample.

    Raises:
       TypeError: If Population or Sample is not a DataFrame.
       ValueError: If Statistics is empty or contains an unknown name, or the two tables share no
          numeric column.

    Notes:
       When the population value is 0, the difference is 0 if the sample value is 0 too, and 1
          otherwise. A statistic that is missing in only one of the two tables counts as a
          difference of 1; missing in both counts as 0.
       Columns that are not numeric (including bool) are ignored.
       A statistic close to 0 in the population (for example the mean of a standardized column) makes the
          relative difference very large, so Rep can become strongly negative; compare such columns with care.

    Notes (TR):
       Örneklemin (ör. Train) popülasyonu ne kadar iyi temsil ettiğini ölçer: Rep = 1 tam temsil
       anlamına gelir, değer düştükçe temsil gücü azalır. Popülasyon değeri 0 ise fark, örneklem
       değeri de 0 ise 0, değilse 1 kabul edilir.
    """
    if not isinstance(Population, pd.DataFrame) or not isinstance(Sample, pd.DataFrame):
        raise TypeError("Population and Sample must be DataFrames.")
    Statistics = list(_DEFAULT_REPRESENTATION_STATISTICS if Statistics is None else Statistics)
    UnknownStatistics = [Name for Name in Statistics if Name not in REPRESENTATION_STATISTICS]
    if not Statistics or UnknownStatistics:
        raise ValueError(f"Statistics must be a non-empty subset of {list(REPRESENTATION_STATISTICS)}; unknown: {UnknownStatistics}")

    SampleNumeric = set(Sample.select_dtypes(include=np.number).columns)
    Columns = [Column for Column in Population.select_dtypes(include=np.number).columns if Column in SampleNumeric]
    if not Columns:
        raise ValueError("Population and Sample share no numeric column.")

    PopulationStats = DataSummary(Population[Columns]).set_index("ColumnName")[Statistics].astype(float)
    SampleStats = DataSummary(Sample[Columns]).set_index("ColumnName")[Statistics].astype(float)

    PopulationValues, SampleValues = np.round(PopulationStats.to_numpy(), 6), np.round(SampleStats.to_numpy(), 6)
    Missing = np.isnan(PopulationValues) | np.isnan(SampleValues)
    PopulationZero = PopulationValues == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        Relative = (SampleValues - PopulationValues) / PopulationValues
    Relative = np.where(PopulationZero, np.where(SampleValues == 0, 0.0, 1.0), Relative)
    Relative = np.where(Missing, np.where(np.isnan(PopulationValues) & np.isnan(SampleValues), 0.0, 1.0), Relative)

    Report = pd.DataFrame({"ColumnName": Columns, "Rep": 1 - np.sqrt((Relative ** 2).mean(axis=1))})
    if Detail:
        Report[Statistics] = Relative
    Report = Report.sort_values("Rep", ascending=False, kind="stable").reset_index(drop=True)

    _Info(f"Representation power of the sample: {Report['Rep'].mean() * 100:.2f}")
    return Report

def DrawChart(
    Table: pd.DataFrame,
    BarColumn: str | None = None,
    BinCount: int | None = None,
    LineColumn: str | None = None,
    IndexColumn: str | None = None,
    Title: str | None = None,
    FigureSize: tuple[float, float] = (5, 3),  # (11, 6) for presentations
    DPI: int = 100,  # 150 for presentations
    SaveImage: bool = False,
    FilePath: str | Path | None = None,
) -> str | None:
    """
    Draw a bar, histogram, line, or combined chart from a table.

    Arg:
       Table: Source table containing the chart data.
       BarColumn: Column plotted as bars or histogram.
       BinCount: Bin count used for numeric histograms.
       LineColumn: Column plotted as a line.
       IndexColumn: Optional column used as the x-axis.
       Title: Optional chart title.
       FigureSize: Figure size in inches as width and height.
       DPI: Figure resolution.
       SaveImage: Whether to save the chart as a PNG file; saving also happens when FilePath is given.
       FilePath: Image file. None saves <yymmdd>_DrawChart.png to the current folder; a path with an
          extension is the file itself; a folder (existing, or a path without an extension) receives
          the default file name. Missing folders are created.

    Returns:
       Path of the saved PNG file (usable in ExportExcel), or None when the chart is not saved.
       The chart is displayed and its figure is closed afterwards.

    Raises:
       ValueError: If required columns are missing or contain no valid data.
       TypeError: If FilePath has the wrong type.

    Notes:
       A file with the same name is updated and a message is printed.
    """
    import matplotlib.pyplot as plt

    if FilePath is not None and not isinstance(FilePath, (str, Path)):
        raise TypeError("FilePath must be a string, a Path or None.")
    if not BarColumn and not LineColumn:
        raise ValueError("Column not found: provide at least one of BarColumn or LineColumn")

    _CheckColumns(Table, [Column for Column in (BarColumn, LineColumn, IndexColumn) if Column])

    Figure, Axis1 = plt.subplots(figsize=FigureSize, dpi=DPI)
    try:
        ChartColumn = BarColumn or LineColumn
        IndexValues = Table[IndexColumn].astype(str) if IndexColumn else None

        if BarColumn:
            if IndexColumn:
                BarValues = pd.to_numeric(Table[BarColumn], errors="coerce")
                if BarValues.dropna().empty:
                    raise ValueError(f"No valid numeric data found in column: {BarColumn}")
                Axis1.bar(IndexValues, BarValues, edgecolor="black", color="#5B8FF9", alpha=0.85)
                Axis1.set_xlabel(IndexColumn)
                Axis1.set_ylabel(BarColumn)
            else:
                BarSeries = Table[BarColumn].dropna()
                if BarSeries.empty:
                    raise ValueError(f"No valid data found in column: {BarColumn}")

                if pd.api.types.is_numeric_dtype(BarSeries):
                    NumericSeries = pd.to_numeric(BarSeries, errors="coerce").dropna()
                    if NumericSeries.empty:
                        raise ValueError(f"No valid numeric data found in column: {BarColumn}")
                    Axis1.hist(NumericSeries, bins=BinCount, edgecolor="black", color="#5B8FF9", alpha=0.85)
                    Axis1.set_ylabel("Frequency")
                else:
                    CategoryFrequency = BarSeries.astype(str).value_counts()
                    Axis1.bar(CategoryFrequency.index, CategoryFrequency.values, edgecolor="black", color="#5B8FF9", alpha=0.85)
                    Axis1.set_ylabel("Count")

                Axis1.set_xlabel(BarColumn)

            Axis1.tick_params(axis="x", rotation=45)
            Axis1.grid(axis="y", alpha=0.3)

        if LineColumn:
            LineSeries = pd.to_numeric(Table[LineColumn], errors="coerce")
            if LineSeries.dropna().empty:
                raise ValueError(f"No valid numeric data found in column: {LineColumn}")

            LineX = IndexValues if IndexColumn else (Table[BarColumn].astype(str) if BarColumn else Table.index)

            if BarColumn:
                Axis2 = Axis1.twinx()
                Axis2.plot(LineX, LineSeries, color="#F08C00", marker=".")
                Axis2.set_ylabel(LineColumn)
            else:
                Axis1.plot(LineX, LineSeries, color="#F08C00", marker=".")
                Axis1.set_xlabel(IndexColumn or Table.index.name or "Index")
                Axis1.set_ylabel(LineColumn)
                Axis1.grid(axis="y", alpha=0.3)

        Axis1.set_title(Title if Title else f"Distribution of {ChartColumn}")
        Figure.tight_layout()

        ImagePath = None
        if SaveImage or FilePath is not None:
            ImagePath = str(_OutputPath(FilePath, "DrawChart", ".png"))
            Figure.savefig(ImagePath)
            _Done(f"Chart saved as image: {ImagePath}")

        plt.show()
    finally:
        plt.close(Figure)

    return ImagePath
