from __future__ import annotations

import importlib.util

import pandas as pd

from .metrics import PERFORMANCE_COLUMNS
from .preprocessing import NULL_STRATEGIES, SCALING_METHODS
from .registry import REGISTRIES

__all__ = ["ModelRegistryGuide", "NullStrategyGuide", "ScalingMethodGuide", "MetricGuide"]

#********* Guides *********#
# Leaf module: it may import any koco module, and no other koco module imports it.

_LANGUAGES = ("ENG", "TR")

def _CheckLanguage(
    Language: str,
) -> str:
    """
    Normalise and validate the language of a guide.

    Arg:
       Language: "ENG" or "TR"; case-insensitive.

    Returns:
       Upper-case language code.

    Raises:
       ValueError: If the language is not supported.
    """
    Language = str(Language).upper()
    if Language not in _LANGUAGES:
        raise ValueError(f"Language must be one of {list(_LANGUAGES)}, got {Language!r}")
    return Language

def _PrintBlocks(
    Guide: pd.DataFrame,
) -> None:
    """
    Print a guide as one readable block per row.

    Arg:
       Guide: Guide whose first column is the name and second column a short tag; the other
          columns are printed as indented lines.

    Returns:
       None.

    Raises:
       None.
    """
    # Long text columns do not fit a wide table, so each row is printed as a block.
    Width = max(len(Column) for Column in Guide.columns[2:])
    for Row in Guide.itertuples(index=False):
        print(f"{Row[0]}  [{Row[1]}]")
        for Column, Value in zip(Guide.columns[2:], Row[2:]):
            print(f"  {Column:<{Width}} : {Value}")
        print()

#********* Model Registry Guide *********#

def ModelRegistryGuide(
    ModelType: str | None = None,
    Print: bool = False,
) -> pd.DataFrame:
    """
    Describe the available algorithms and their data sensitivities.

    Arg:
       ModelType: "Regression" or "Binary" to list a single registry; None lists both.
       Print: If True, print the registry and its scaling / NaN sensitivity lists.

    Returns:
       DataFrame with one row per algorithm key showing its model type, estimator class, whether it
       needs scaled features, whether it accepts NaN values, and whether its package is installed.

    Raises:
       ValueError: If ModelType is unknown.
    """
    if ModelType is not None and ModelType not in REGISTRIES:
        raise ValueError(f"ModelType must be one of {list(REGISTRIES)}.")

    Rows = []
    for RegistryType, Registry in REGISTRIES.items():
        if ModelType is not None and RegistryType != ModelType:
            continue
        for Key, Spec in Registry.items():
            Rows.append({
                "ModelType"       : RegistryType,
                "Key"             : Key,
                "Algorithm"       : Spec.ClassName,
                "Module"          : Spec.Module,
                "ScalingSensitive": "Yes" if Spec.ScalingSensitive else "No",
                "AcceptsNaN"      : "Yes" if Spec.NaNTolerant else "No",
                "Installed"       : "Yes" if importlib.util.find_spec(Spec.Module.split(".")[0]) else "No",
            })

    RegistryTable = pd.DataFrame(Rows)
    if Print:
        print(RegistryTable.to_string(index=False))
        print(f"\nScaling sensitive models: {[Row['Key'] for Row in Rows if Row['ScalingSensitive'] == 'Yes']}")
        print(f"NaN intolerant models   : {[Row['Key'] for Row in Rows if Row['AcceptsNaN'] == 'No']}")
    return RegistryTable

#********* Null Strategy Guide *********#

_NULL_GUIDE_COLUMNS = {
    "ENG": ["Strategy", "Description"],
    "TR" : ["Strateji", "Açıklama"],
}
# Strategy: {Language: Description}
_NULL_GUIDE = {
    "Drop": {
        "ENG": "Drop the column entirely — high null rate or the column isn't worth keeping.",
        "TR" : "Sütunu tamamen siler — null oranı yüksek veya sütun tutmaya değmez.",
    },
    "DropRows": {
        "ENG": "Drop rows containing a null — few nulls, losing those rows is acceptable.",
        "TR" : "Null içeren satırları siler — null sayısı azdır ve bu satırların kaybı kabul edilebilir.",
    },
    "Median": {
        "ENG": "Fill nulls with the median — numeric, robust to outliers.",
        "TR" : "Null değerleri medyanla doldurur — sayısal, aykırı değerlere dayanıklı.",
    },
    "Mean": {
        "ENG": "Fill nulls with the mean — numeric, roughly symmetric distribution.",
        "TR" : "Null değerleri ortalamayla doldurur — sayısal, dağılımı yaklaşık simetrik.",
    },
    "Mode": {
        "ENG": "Fill nulls with the mode (most frequent value) — categorical / low-cardinality.",
        "TR" : "Null değerleri mod (en sık değer) ile doldurur — kategorik / düşük kardinaliteli.",
    },
    "FixValue": {
        "ENG": "Fill nulls with a fixed value like Zero or 'Student' — the user should directly write the value.",
        "TR" : "Null değerleri 0 veya 'Student' gibi sabit bir değerle doldurur — değeri kullanıcı doğrudan yazar.",
    },
    "KeepNaN": {
        "ENG": "Leave null as-is — only usable by NaN-tolerant algorithms like LGBMR/XGBR/CatBoostR.",
        "TR" : "Null değeri olduğu gibi bırakır — yalnızca LGBMR/XGBR/CatBoostR gibi NaN toleranslı algoritmalarla kullanılabilir.",
    },
}

def NullStrategyGuide(
    Language: str = "ENG",
    Print: bool = False,
) -> pd.DataFrame:
    """
    Describe the null-handling strategies available in a NullPlan.

    Arg:
       Language: "ENG" (English) or "TR" (Turkish). Case-insensitive.
       Print: If True, also print the guide.

    Returns:
       DataFrame with one row per strategy: Strategy and Description (Strateji and Açıklama in Turkish).

    Raises:
       ValueError: If Language is not supported.
    """
    Language = _CheckLanguage(Language)
    Guide = pd.DataFrame(
        [[Strategy, _NULL_GUIDE[Strategy][Language]] for Strategy in NULL_STRATEGIES],
        columns=_NULL_GUIDE_COLUMNS[Language],
    )
    if Print:
        print("Available NullPlan actions:" if Language == "ENG" else "Kullanılabilir NullPlan aksiyonları:")
        print(Guide.to_string(index=False))
    return Guide

#********* Scaling Method Guide *********#

_SCALING_GUIDE_COLUMNS = {
    "ENG": ["Method", "Scaler", "Result Range", "Outlier Sensitivity", "Best For", "Description"],
    "TR" : ["Yöntem", "Scaler", "Sonuç Aralığı", "Aykırı Değer Hassasiyeti", "Uygun Kullanım", "Açıklama"],
}
# Method: (Scaler, {Language: (Result Range, Outlier Sensitivity, Best For, Description)})
_SCALING_GUIDE = {
    "Standardization": ("StandardScaler", {
        "ENG": ("Mean 0, standard deviation 1", "High",
                "Ridge, Lasso and Linear Regression; data close to a normal distribution.",
                "Z-score scaling. The mean and standard deviation are affected by outliers, so it works best when the data is close to normal."),
        "TR" : ("Ortalama 0, standart sapma 1", "Yüksek",
                "Ridge, Lasso ve Linear Regression; normal dağılıma yakın veri.",
                "Standardizasyon (Z-Score Scaling): Verinin ortalamasını 0, standart sapmasını 1 yapar. Ortalama ve standart sapma aykırı değerlerden etkilenir; veri normal (gauss) dağılıma yakınsa en iyi sonucu verir."),
    }),
    "Normalization": ("MinMaxScaler", {
        "ENG": ("[0, 1]", "High",
                "Distance based or gradient based algorithms (KNN, SVM, MLP).",
                "Min-max scaling squeezes the data into [0, 1] so everything stays within a fixed bound. Outliers can push all the other values into a narrow band."),
        "TR" : ("[0, 1]", "Yüksek",
                "Uzaklık temelli veya gradyan tabanlı algoritmalar (KNN, SVM, MLP).",
                "Normalizasyon (Min-Max Scaling): Verileri 0 ile 1 aralığına sıkıştırır; tüm verilerin belirli bir sınır içinde olmasını sağlar. Aykırı değerler diğer değerleri dar bir aralığa toplayabilir."),
    }),
    "Robust": ("RobustScaler", {
        "ENG": ("Centred on the median, scaled by the IQR", "Low",
                "Data with many outliers.",
                "Uses the median and the interquartile range (IQR), so outliers hardly change the scaling. The safest choice when the data has many outliers."),
        "TR" : ("Medyan merkezli, IQR ile ölçekli", "Düşük",
                "Çok sayıda aykırı değer içeren veri.",
                "Robust Scaler: Medyan (ortanca) ve çeyrekler açıklığını (IQR) kullanır. Veri setinde çok fazla aykırı değer (outlier) varsa en güvenli yöntemdir; aykırı değerlerin modeli bozmasını engeller."),
    }),
    "MaxAbs": ("MaxAbsScaler", {
        "ENG": ("[-1, 1]", "High",
                "Sparse data.",
                "Divides each value by the maximum absolute value. Zeros are kept, so sparse structure is preserved."),
        "TR" : ("[-1, 1]", "Yüksek",
                "Seyrek (sparse) veri setleri.",
                "MaxAbs Scaler: Değeri maksimum mutlak değerine bölerek -1 ile 1 aralığına ölçekler. Sıfır değerlerini korur, bu yüzden seyrek veri setleri için uygundur."),
    }),
}

def ScalingMethodGuide(
    Language: str = "ENG",
    Print: bool = False,
) -> pd.DataFrame:
    """
    Describe the scaling methods available in ScalingPlan and when to use each.

    Arg:
       Language: "ENG" (English) or "TR" (Turkish). Case-insensitive.
       Print: If True, also print the guide as one readable block per method.

    Returns:
       DataFrame with one row per method: Method, Scaler, Result Range, Outlier Sensitivity, Best For
       and Description (translated column names in Turkish).

    Raises:
       ValueError: If Language is not supported.
    """
    Language = _CheckLanguage(Language)
    Rows = [[Method, _SCALING_GUIDE[Method][0], *_SCALING_GUIDE[Method][1][Language]] for Method in SCALING_METHODS]
    Guide = pd.DataFrame(Rows, columns=_SCALING_GUIDE_COLUMNS[Language])
    if Print:
        _PrintBlocks(Guide)
    return Guide

#********* Metric Guide *********#

_GUIDE_COLUMNS = {
    "ENG": ["Metric", "Model Type", "Value Range", "Better When", "Purpose & Significance", "Interpretation"],
    "TR" : ["Metrik", "Model Tipi", "Değer Aralığı", "İyi Yön", "Amaç ve Önem", "Yorumlama"],
}
_GUIDE_MODEL_TYPES = {
    "ENG": {"Regression": "Regression", "Binary": "Binary Classification"},
    "TR" : {"Regression": "Regresyon", "Binary": "İkili Sınıflandırma"},
}
_GUIDE_DIRECTIONS = {
    "ENG": {"Higher": "Higher", "Lower": "Lower", "Zero": "Closer to 0", "Rate": "Closer to TargetRate", "Info": "Informational"},
    "TR" : {"Higher": "Yüksek", "Lower": "Düşük", "Zero": "0'a yakın", "Rate": "TargetRate'e yakın", "Info": "Bilgi amaçlı"},
}
# Metric: (Range, Direction, {Language: (Purpose, Interpretation)})
_METRIC_GUIDE = {
    "N": ("[1, +∞)", "Info", {
        "ENG": ("Number of observations the metrics are calculated on.",
                "Metrics calculated on small samples are less reliable."),
        "TR" : ("Metriklerin hesaplandığı gözlem sayısı.",
                "Küçük örneklemlerde hesaplanan metrikler daha az güvenilirdir."),
    }),
    "RMSE": ("[0, +∞)", "Lower", {
        "ENG": ("Typical prediction error in target units; large errors are penalised more heavily.",
                "0 means perfect predictions. Compare with the typical size of the target; RMSE much larger than MAE points to a few large errors."),
        "TR" : ("Hedef değişken biriminde tipik tahmin hatası; büyük hatalar daha ağır cezalandırılır.",
                "0 mükemmel tahmindir. Hedefin tipik büyüklüğüyle kıyaslanmalıdır; RMSE'nin MAE'den çok büyük olması birkaç büyük hatanın varlığına işaret eder."),
    }),
    "MAE": ("[0, +∞)", "Lower", {
        "ENG": ("Average absolute deviation between prediction and actual, in target units.",
                "Predictions miss the actual value by MAE units on average."),
        "TR" : ("Tahmin ile gerçekleşen arasındaki ortalama mutlak sapma (hedef biriminde).",
                "Tahminler gerçekleşen değerden ortalama MAE birim sapar."),
    }),
    "MedAE": ("[0, +∞)", "Lower", {
        "ENG": ("Median absolute error; typical error that is not affected by outliers.",
                "Half of the predictions miss by less than this value."),
        "TR" : ("Medyan mutlak hata; aykırı değerlerden etkilenmeyen tipik hata.",
                "Tahminlerin yarısı bu değerden daha az sapar."),
    }),
    "MAPE": ("[0, +∞)", "Lower", {
        "ENG": ("Average error as a percentage of the actual value; unit-free, comparable across targets.",
                "0.10 means predictions are off by 10% on average. Inflated when actual values are close to 0; rows with actual = 0 are excluded."),
        "TR" : ("Gerçekleşen değere oranla ortalama yüzde hata; birimden bağımsızdır, farklı hedefler arasında kıyaslanabilir.",
                "0.10, tahminlerin ortalama %10 saptığı anlamına gelir. Gerçekleşen değerler 0'a yakınsa şişer; gerçekleşeni 0 olan satırlar hesaba katılmaz."),
    }),
    "WMAPE": ("[0, +∞)", "Lower", {
        "ENG": ("Total absolute error relative to total actual volume; robust to small or zero actual values.",
                "0.10 means the total error equals 10% of the total actual volume."),
        "TR" : ("Toplam mutlak hatanın toplam gerçekleşen hacme oranı; küçük veya 0 gerçekleşen değerlerden etkilenmez.",
                "0.10, toplam hatanın toplam gerçekleşen hacmin %10'u olduğu anlamına gelir."),
    }),
    "R2": ("(-∞, 1]", "Higher", {
        "ENG": ("Share of the variation in the target explained by the model.",
                "1 is perfect, 0 is no better than always predicting the average, negative is worse than the average."),
        "TR" : ("Hedef değişkendeki değişkenliğin model tarafından açıklanan payı.",
                "1 mükemmeldir, 0 her zaman ortalamayı tahmin etmekten farksızdır, negatif değer ortalamadan daha kötüdür."),
    }),
    "Bias": ("(-∞, +∞)", "Zero", {
        "ENG": ("Average signed error; shows systematic over- or under-prediction.",
                "Positive means the model over-predicts on average, negative means it under-predicts."),
        "TR" : ("Ortalama işaretli hata; sistematik olarak fazla veya eksik tahmin yapılıp yapılmadığını gösterir.",
                "Pozitif değer modelin ortalamada fazla, negatif değer eksik tahmin ettiğini gösterir."),
    }),
    "TargetCount": ("[0, N]", "Info", {
        "ENG": ("Number of actual events (class 1).",
                "With few events, AUC, Gini and KS become unstable."),
        "TR" : ("Gerçekleşen olay (sınıf 1) sayısı.",
                "Olay sayısı azsa AUC, Gini ve KS değerleri istikrarsız olur."),
    }),
    "TargetRate": ("[0, 1]", "Info", {
        "ENG": ("Event rate in the sample; the baseline a random selection would achieve.",
                "0.05 means 5% of the observations are events. Precision should be compared against this value."),
        "TR" : ("Örneklemdeki olay oranı; rastgele seçimle ulaşılacak temel seviye.",
                "0.05, gözlemlerin %5'inin olay olduğu anlamına gelir. Precision bu değerle kıyaslanmalıdır."),
    }),
    "MeanProb": ("[0, 1]", "Rate", {
        "ENG": ("Average predicted probability; checks whether the probability level is realistic (calibration).",
                "Should be close to TargetRate. Higher means the model overestimates, lower means it underestimates the event rate."),
        "TR" : ("Ortalama tahmin olasılığı; olasılık seviyesinin gerçekçi olup olmadığını (kalibrasyon) kontrol eder.",
                "TargetRate'e yakın olmalıdır. Daha yüksekse model olay oranını olduğundan fazla, düşükse eksik tahmin eder."),
    }),
    "ZeroPred": ("[0, 1]", "Lower", {
        "ENG": ("Average probability given to actual non-events (class 0).",
                "Should be low and clearly below OnePred."),
        "TR" : ("Gerçekte olay olmayan (sınıf 0) gözlemlere verilen ortalama olasılık.",
                "Düşük ve OnePred'in belirgin şekilde altında olmalıdır."),
    }),
    "OnePred": ("[0, 1]", "Higher", {
        "ENG": ("Average probability given to actual events (class 1).",
                "The larger the gap between OnePred and ZeroPred, the better the model separates the classes."),
        "TR" : ("Gerçekte olay olan (sınıf 1) gözlemlere verilen ortalama olasılık.",
                "OnePred ile ZeroPred arasındaki fark büyüdükçe model sınıfları daha iyi ayırır."),
    }),
    "AUC": ("[0, 1]", "Higher", {
        "ENG": ("Probability that a random event is scored higher than a random non-event; ranking power, independent of the threshold.",
                "0.5 random, 0.6-0.7 weak, 0.7-0.8 acceptable, 0.8-0.9 good, > 0.9 very strong (check for overfitting / data leakage)."),
        "TR" : ("Rastgele seçilen bir olayın, rastgele seçilen bir olay olmayana göre daha yüksek skor alma olasılığı; eşikten bağımsız sıralama gücü.",
                "0.5 rastgele, 0.6-0.7 zayıf, 0.7-0.8 kabul edilebilir, 0.8-0.9 iyi, > 0.9 çok güçlü (aşırı öğrenme / veri sızıntısı kontrol edilmeli)."),
    }),
    "Gini": ("[-1, 1]", "Higher", {
        "ENG": ("Rescaled AUC (2 * AUC - 1); standard in credit risk and scoring.",
                "0 random, 1 perfect. 0.2-0.4 weak, 0.4-0.6 acceptable, 0.6-0.8 good, > 0.8 very strong (check for leakage)."),
        "TR" : ("Yeniden ölçeklenmiş AUC (2 * AUC - 1); kredi riski ve skorlamada standarttır.",
                "0 rastgele, 1 mükemmel. 0.2-0.4 zayıf, 0.4-0.6 kabul edilebilir, 0.6-0.8 iyi, > 0.8 çok güçlü (sızıntı kontrol edilmeli)."),
    }),
    "KS": ("[0, 1]", "Higher", {
        "ENG": ("Maximum separation between the cumulative score distributions of events and non-events.",
                "0 no separation. < 0.2 weak, 0.2-0.4 acceptable, 0.4-0.6 good, > 0.6 very strong (check for leakage)."),
        "TR" : ("Olay ve olay olmayanların kümülatif skor dağılımları arasındaki en büyük ayrışma.",
                "0 ayrışma yok. < 0.2 zayıf, 0.2-0.4 kabul edilebilir, 0.4-0.6 iyi, > 0.6 çok güçlü (sızıntı kontrol edilmeli)."),
    }),
    "LogLoss": ("[0, +∞)", "Lower", {
        "ENG": ("Quality of the predicted probabilities; confident wrong predictions are heavily penalised.",
                "0 is perfect. Has no absolute scale; use it to compare models on the same data."),
        "TR" : ("Tahmin edilen olasılıkların kalitesi; emin olunan yanlış tahminler ağır cezalandırılır.",
                "0 mükemmeldir. Mutlak bir ölçeği yoktur; aynı veri üzerinde modelleri kıyaslamak için kullanılır."),
    }),
    "Brier": ("[0, 1]", "Lower", {
        "ENG": ("Mean squared error of the probabilities; combines accuracy and calibration.",
                "0 is perfect. Should be clearly below TargetRate * (1 - TargetRate), the score of always predicting the event rate."),
        "TR" : ("Olasılıkların ortalama karesel hatası; doğruluk ve kalibrasyonu birlikte ölçer.",
                "0 mükemmeldir. Her zaman olay oranını tahmin etmenin skoru olan TargetRate * (1 - TargetRate) değerinin belirgin şekilde altında olmalıdır."),
    }),
    "Threshold": ("[0, 1]", "Info", {
        "ENG": ("Probability cut-off above which an observation is labelled as an event.",
                "Raising it increases Precision and lowers Recall; it should be set according to business costs."),
        "TR" : ("Bir gözlemin olay olarak etiketlendiği olasılık eşiği.",
                "Eşik yükseldikçe Precision artar, Recall düşer; iş maliyetlerine göre belirlenmelidir."),
    }),
    "TP": ("[0, N]", "Higher", {
        "ENG": ("True positives: actual events correctly flagged.",
                "Events captured by the model."),
        "TR" : ("Doğru pozitif: doğru şekilde işaretlenen gerçek olaylar.",
                "Modelin yakaladığı olaylar."),
    }),
    "TN": ("[0, N]", "Higher", {
        "ENG": ("True negatives: actual non-events correctly not flagged.",
                "Non-events the model correctly left out."),
        "TR" : ("Doğru negatif: doğru şekilde işaretlenmeyen gerçek olay olmayanlar.",
                "Modelin doğru şekilde dışarıda bıraktığı olay olmayanlar."),
    }),
    "FP": ("[0, N]", "Lower", {
        "ENG": ("False positives: non-events flagged as events (false alarms).",
                "Represents the cost of unnecessary action."),
        "TR" : ("Yanlış pozitif: olay olarak işaretlenen olay olmayanlar (yanlış alarm).",
                "Gereksiz aksiyonun maliyetini temsil eder."),
    }),
    "FN": ("[0, N]", "Lower", {
        "ENG": ("False negatives: events the model missed.",
                "Represents the cost of missed risk or opportunity."),
        "TR" : ("Yanlış negatif: modelin kaçırdığı olaylar.",
                "Kaçırılan risk veya fırsatın maliyetini temsil eder."),
    }),
    "Precision": ("[0, 1]", "Higher", {
        "ENG": ("Share of flagged observations that are actual events.",
                "0.30 means 30 of every 100 flagged cases are real events. Precision / TargetRate gives the lift over random selection."),
        "TR" : ("İşaretlenen gözlemler içinde gerçekten olay olanların payı.",
                "0.30, işaretlenen her 100 vakadan 30'unun gerçek olay olduğu anlamına gelir. Precision / TargetRate rastgele seçime göre kazancı (lift) verir."),
    }),
    "Recall": ("[0, 1]", "Higher", {
        "ENG": ("Share of actual events captured by the model.",
                "0.70 means the model catches 70% of all events."),
        "TR" : ("Gerçek olayların model tarafından yakalanan payı.",
                "0.70, modelin tüm olayların %70'ini yakaladığı anlamına gelir."),
    }),
    "F1": ("[0, 1]", "Higher", {
        "ENG": ("Harmonic mean of Precision and Recall; a single balance score.",
                "Low if either Precision or Recall is low; useful when both false alarms and misses matter."),
        "TR" : ("Precision ve Recall'un harmonik ortalaması; tek bir denge skoru.",
                "Precision veya Recall'dan biri düşükse düşüktür; hem yanlış alarm hem kaçırma önemliyse kullanışlıdır."),
    }),
    "FitTimeSec": ("[0, +∞)", "Lower", {
        "ENG": ("Model training time in seconds.",
                "Operational cost indicator; not a measure of accuracy."),
        "TR" : ("Saniye cinsinden model eğitim süresi.",
                "Operasyonel maliyet göstergesidir; doğruluk ölçüsü değildir."),
    }),
}

def MetricGuide(
    ModelType: str = "All",
    Language: str = "ENG",
    Print: bool = False,
) -> pd.DataFrame:
    """
    Build a business-friendly description table of the performance metrics.

    Arg:
       ModelType: "Regression", "Binary" or "All".
       Language: "ENG" (English) or "TR" (Turkish). Case-insensitive.
       Print: If True, also print the guide as one readable block per metric.

    Returns:
       DataFrame with one row per metric, in the column order of the performance tables:
       Metric, Model Type, Value Range, Better When, Purpose & Significance, Interpretation.

    Raises:
       ValueError: If ModelType or Language is not supported.

    Notes:
       Quality bands (e.g. for AUC, Gini, KS) are rules of thumb and vary by domain.
    """
    Language = _CheckLanguage(Language)
    if ModelType != "All" and ModelType not in PERFORMANCE_COLUMNS:
        raise ValueError(f"ModelType must be 'All' or one of {list(PERFORMANCE_COLUMNS)}, got {ModelType!r}")

    Types = list(PERFORMANCE_COLUMNS) if ModelType == "All" else [ModelType]
    Metrics = list(dict.fromkeys(Column for Type in Types for Column in PERFORMANCE_COLUMNS[Type] if Column in _METRIC_GUIDE))

    Rows = []
    for Metric in Metrics:
        Range, Direction, Texts = _METRIC_GUIDE[Metric]
        MetricTypes = ", ".join(_GUIDE_MODEL_TYPES[Language][Type] for Type in Types if Metric in PERFORMANCE_COLUMNS[Type])
        Rows.append([Metric, MetricTypes, Range, _GUIDE_DIRECTIONS[Language][Direction], *Texts[Language]])
    Guide = pd.DataFrame(Rows, columns=_GUIDE_COLUMNS[Language])

    if Print:
        _PrintBlocks(Guide)

    return Guide
