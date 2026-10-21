import numpy as np
import pandas as pd

__all__ = ["RegressionMetrics", "ClassificationMetrics"]

#********* Metric Functions *********#

# Column order of the Performance table of each model type; MetricGuide follows this order.
PERFORMANCE_COLUMNS = {
    "Regression": ["Label", "AlgorithmName", "SampleName", "N", "RMSE", "MAE", "MedAE", "MAPE", "WMAPE", "R2", "Bias", "FitTimeSec"],
    "Binary"    : ["Label", "AlgorithmName", "SampleName", "N", "TargetCount", "TargetRate", "MeanProb", "ZeroPred", "OnePred",
                   "AUC", "Gini", "KS", "LogLoss", "Brier", "Threshold", "TP", "TN", "FP", "FN", "Precision", "Recall", "F1", "FitTimeSec"],
}

def RegressionMetrics(
    YTrue: pd.Series | np.ndarray,
    YPred: pd.Series | np.ndarray,
) -> dict:
    """
    Calculate regression performance metrics.

    Arg:
       YTrue: Actual target values.
       YPred: Predicted values, in the same order as YTrue.

    Returns:
       Dict of metrics, rounded to 4 decimals.
         N: Number of observations.
         RMSE: Root mean squared error; penalises large errors more heavily.
         MAE: Mean absolute error.
         MedAE: Median absolute error; robust to outliers.
         MAPE: Mean absolute percentage error as a ratio, computed on rows where YTrue != 0.
         WMAPE: Total absolute error divided by total absolute actual; safe with zero targets.
         R2: Share of the Target variance explained by the model.
         Bias: Mean of (YPred - YTrue); positive means systematic over-prediction.

    Raises:
       ValueError: If the inputs are empty, differ in length, or contain NaN / inf values.
    """
    YTrue = np.asarray(YTrue, dtype=float).ravel()
    YPred = np.asarray(YPred, dtype=float).ravel()

    if len(YTrue) == 0:
        raise ValueError("YTrue must not be empty.")
    if len(YTrue) != len(YPred):
        raise ValueError(f"YTrue and YPred lengths differ: {len(YTrue)} vs {len(YPred)}")
    if not np.isfinite(YTrue).all() or not np.isfinite(YPred).all():
        raise ValueError("YTrue and YPred must not contain NaN or inf values.")

    Errors = YPred - YTrue
    AbsErrors = np.abs(Errors)
    NonZero = YTrue != 0
    SumAbsTrue = np.abs(YTrue).sum()
    TotalSS = ((YTrue - YTrue.mean()) ** 2).sum()

    Metrics = {
        "RMSE" : np.sqrt(np.mean(Errors ** 2)),
        "MAE"  : AbsErrors.mean(),
        "MedAE": np.median(AbsErrors),
        "MAPE" : (AbsErrors[NonZero] / np.abs(YTrue[NonZero])).mean() if NonZero.any() else np.nan,
        "WMAPE": AbsErrors.sum() / SumAbsTrue if SumAbsTrue > 0 else np.nan,
        "R2"   : 1 - (Errors ** 2).sum() / TotalSS if TotalSS > 0 else np.nan,
        "Bias" : Errors.mean(),
    }
    return {"N": len(YTrue), **{Name: round(float(Value), 4) for Name, Value in Metrics.items()}}

def ClassificationMetrics(
    YTrue: pd.Series | np.ndarray,
    YProb: pd.Series | np.ndarray,
    Threshold: float = 0.5,
) -> dict:
    """
    Calculate binary classification performance metrics.

    Arg:
       YTrue: Actual 0/1 target values.
       YProb: Predicted probability of class 1, in the same order as YTrue.
       Threshold: Probability cut-off used to turn YProb into 0/1 predictions.

    Returns:
       Dict of metrics; ratios are rounded to 4 decimals. AUC, Gini and KS are NaN when YTrue
       contains a single class.
         N: Number of observations.
         TargetCount: Number of class 1 observations.
         TargetRate: Share of class 1.
         MeanProb: Mean predicted probability; close to TargetRate when the model is calibrated.
         ZeroPred: Mean predicted probability of class 0 observations.
         OnePred: Mean predicted probability of class 1 observations.
         AUC: Area under the ROC curve; ranking power, independent of Threshold.
         Gini: 2 * AUC - 1.
         KS: Maximum gap between the cumulative distributions of class 1 and class 0.
         LogLoss: Cross-entropy of the probabilities; lower is better.
         Brier: Mean squared error of the probabilities; lower is better.
         Threshold: Cut-off used for the confusion matrix metrics.
         TP, TN, FP, FN: Confusion matrix counts.
         Precision: Share of predicted 1s that are actually 1.
         Recall: Share of actual 1s that are predicted as 1.
         F1: Harmonic mean of Precision and Recall.

    Raises:
       ValueError: If the inputs are empty, differ in length, YTrue is not 0/1, YProb is outside
          [0, 1], or Threshold is outside [0, 1].
    """
    from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve

    YTrue = np.asarray(YTrue, dtype=float).ravel()
    YProb = np.asarray(YProb, dtype=float).ravel()

    if len(YTrue) == 0:
        raise ValueError("YTrue must not be empty.")
    if len(YTrue) != len(YProb):
        raise ValueError(f"YTrue and YProb lengths differ: {len(YTrue)} vs {len(YProb)}")
    if not np.isin(YTrue, [0, 1]).all():
        raise ValueError("YTrue must contain only 0 and 1.")
    if not np.isfinite(YProb).all() or ((YProb < 0) | (YProb > 1)).any():
        raise ValueError("YProb must contain probabilities between 0 and 1.")
    if not 0 <= Threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    YTrue = YTrue.astype(int)
    YPred = (YProb >= Threshold).astype(int)
    TN, FP, FN, TP = confusion_matrix(YTrue, YPred, labels=[0, 1]).ravel()

    AUC = KS = np.nan
    if len(np.unique(YTrue)) == 2:
        AUC = roc_auc_score(YTrue, YProb)
        FPR, TPR, _ = roc_curve(YTrue, YProb)
        KS = np.max(TPR - FPR)

    Ratios = {
        "TargetRate": YTrue.mean(),
        "MeanProb"  : YProb.mean(),
        "ZeroPred"  : YProb[YTrue == 0].mean() if (YTrue == 0).any() else np.nan,
        "OnePred"   : YProb[YTrue == 1].mean() if (YTrue == 1).any() else np.nan,
        "AUC"       : AUC,
        "Gini"      : 2 * AUC - 1,
        "KS"        : KS,
        "LogLoss"   : log_loss(YTrue, YProb, labels=[0, 1]),
        "Brier"     : np.mean((YProb - YTrue) ** 2),
        "Threshold" : Threshold,
    }
    Scores = {
        "Precision": precision_score(YTrue, YPred, zero_division=0),
        "Recall"   : recall_score(YTrue, YPred, zero_division=0),
        "F1"       : f1_score(YTrue, YPred, zero_division=0),
    }
    return {
        "N": len(YTrue),
        "TargetCount": int(YTrue.sum()),
        **{Name: round(float(Value), 4) for Name, Value in Ratios.items()},
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        **{Name: round(float(Value), 4) for Name, Value in Scores.items()},
    }
