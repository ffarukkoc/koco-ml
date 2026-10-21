import importlib
import inspect
from dataclasses import dataclass, field

from ._helpers import DEFAULT_RANDOM_STATE

__all__: list[str] = []

#********* Model Registry *********#

@dataclass(frozen=True)
class ModelSpec:
    """
    Everything koco knows about one algorithm key.

    Arg:
       Module: Module that provides the estimator class.
       ClassName: Estimator class name.
       ModelType: "Regression" or "Binary".
       Params: Default constructor parameters.
       ScalingSensitive: True if results depend on feature scale (features are assumed to be scaled beforehand).
       NaNTolerant: False if the algorithm raises on NaN inputs (verified with scikit-learn 1.8).

    Returns:
       Immutable specification stored in MODEL_SPECS.

    Raises:
       None.
    """
    Module: str
    ClassName: str
    ModelType: str
    Params: dict = field(default_factory=dict)
    ScalingSensitive: bool = False
    NaNTolerant: bool = True

_REG, _BIN = "Regression", "Binary"
_SEED = {"random_state": DEFAULT_RANDOM_STATE}
_FOREST = {"n_estimators": 200, "random_state": DEFAULT_RANDOM_STATE, "n_jobs": -1}
_MLP = {"max_iter": 500, "early_stopping": True, "random_state": DEFAULT_RANDOM_STATE}

# Key -> ModelSpec. Regression keys end with "R", classification keys end with "C".
MODEL_SPECS: dict[str, ModelSpec] = {
    # ---- Regression ----
    # Baseline
    "DummyR"      : ModelSpec("sklearn.dummy",          "DummyRegressor",                _REG, {"strategy": "median"}),
    # Linear
    "LinearR"     : ModelSpec("sklearn.linear_model",   "LinearRegression",              _REG, {}, ScalingSensitive=True, NaNTolerant=False),
    "RidgeR"      : ModelSpec("sklearn.linear_model",   "Ridge",                         _REG, {}, ScalingSensitive=True, NaNTolerant=False),
    "LassoR"      : ModelSpec("sklearn.linear_model",   "Lasso",                         _REG, {}, ScalingSensitive=True, NaNTolerant=False),
    "ElasticNetR" : ModelSpec("sklearn.linear_model",   "ElasticNet",                    _REG, dict(_SEED), ScalingSensitive=True, NaNTolerant=False),
    "HuberR"      : ModelSpec("sklearn.linear_model",   "HuberRegressor",                _REG, {}, ScalingSensitive=True, NaNTolerant=False),
    "BayesRidgeR" : ModelSpec("sklearn.linear_model",   "BayesianRidge",                 _REG, {}, ScalingSensitive=True, NaNTolerant=False),
    # Tree / Ensemble
    "DecTR"       : ModelSpec("sklearn.tree",           "DecisionTreeRegressor",         _REG, dict(_SEED)),
    "RandFR"      : ModelSpec("sklearn.ensemble",       "RandomForestRegressor",         _REG, dict(_FOREST)),
    "ExtraTreesR" : ModelSpec("sklearn.ensemble",       "ExtraTreesRegressor",           _REG, dict(_FOREST)),
    "GradBR"      : ModelSpec("sklearn.ensemble",       "GradientBoostingRegressor",     _REG, dict(_SEED), NaNTolerant=False),
    "HistGBR"     : ModelSpec("sklearn.ensemble",       "HistGradientBoostingRegressor", _REG, dict(_SEED)),
    "AdaBoostR"   : ModelSpec("sklearn.ensemble",       "AdaBoostRegressor",             _REG, dict(_SEED), NaNTolerant=False),
    "BaggingR"    : ModelSpec("sklearn.ensemble",       "BaggingRegressor",              _REG, dict(_SEED)),
    # Boosting (optional packages)
    "XGBR"        : ModelSpec("xgboost",                "XGBRegressor",                  _REG, dict(_SEED)),
    "LGBMR"       : ModelSpec("lightgbm",               "LGBMRegressor",                 _REG, {**_SEED, "verbosity": -1}),
    "CatBoostR"   : ModelSpec("catboost",               "CatBoostRegressor",             _REG, {"random_seed": DEFAULT_RANDOM_STATE, "verbose": 0}),
    # Other
    "SVR"         : ModelSpec("sklearn.svm",            "SVR",                           _REG, {}, ScalingSensitive=True, NaNTolerant=False),
    "KNNR"        : ModelSpec("sklearn.neighbors",      "KNeighborsRegressor",           _REG, {}, ScalingSensitive=True, NaNTolerant=False),
    "MLPR"        : ModelSpec("sklearn.neural_network", "MLPRegressor",                  _REG, dict(_MLP), ScalingSensitive=True, NaNTolerant=False),

    # ---- Binary classification ----
    # Baseline
    "DummyC"      : ModelSpec("sklearn.dummy",          "DummyClassifier",                _BIN, {"strategy": "prior"}),
    # Linear
    "LogisticC"   : ModelSpec("sklearn.linear_model",   "LogisticRegression",             _BIN, {"max_iter": 1000}, ScalingSensitive=True, NaNTolerant=False),
    # Tree / Ensemble
    "DecTC"       : ModelSpec("sklearn.tree",           "DecisionTreeClassifier",         _BIN, dict(_SEED)),
    "RandFC"      : ModelSpec("sklearn.ensemble",       "RandomForestClassifier",         _BIN, dict(_FOREST)),
    "ExtraTreesC" : ModelSpec("sklearn.ensemble",       "ExtraTreesClassifier",           _BIN, dict(_FOREST)),
    "GradBC"      : ModelSpec("sklearn.ensemble",       "GradientBoostingClassifier",     _BIN, dict(_SEED), NaNTolerant=False),
    "HistGBC"     : ModelSpec("sklearn.ensemble",       "HistGradientBoostingClassifier", _BIN, dict(_SEED)),
    "AdaBoostC"   : ModelSpec("sklearn.ensemble",       "AdaBoostClassifier",             _BIN, dict(_SEED), NaNTolerant=False),
    "BaggingC"    : ModelSpec("sklearn.ensemble",       "BaggingClassifier",              _BIN, dict(_SEED)),
    # Boosting (optional packages)
    "XGBC"        : ModelSpec("xgboost",                "XGBClassifier",                  _BIN, dict(_SEED)),
    "LGBMC"       : ModelSpec("lightgbm",               "LGBMClassifier",                 _BIN, {**_SEED, "verbosity": -1}),
    "CatBoostC"   : ModelSpec("catboost",               "CatBoostClassifier",             _BIN, {"random_seed": DEFAULT_RANDOM_STATE, "verbose": 0}),
    # Other
    "GaussNBC"    : ModelSpec("sklearn.naive_bayes",    "GaussianNB",                     _BIN, {}, NaNTolerant=False),
    "SVC"         : ModelSpec("sklearn.svm",            "SVC",                            _BIN, {"probability": True, **_SEED}, ScalingSensitive=True, NaNTolerant=False),
    "KNNC"        : ModelSpec("sklearn.neighbors",      "KNeighborsClassifier",           _BIN, {}, ScalingSensitive=True, NaNTolerant=False),
    "MLPC"        : ModelSpec("sklearn.neural_network", "MLPClassifier",                  _BIN, dict(_MLP), ScalingSensitive=True, NaNTolerant=False),
}

REGISTRIES: dict[str, dict[str, ModelSpec]] = {
    ModelType: {Key: Spec for Key, Spec in MODEL_SPECS.items() if Spec.ModelType == ModelType}
    for ModelType in (_REG, _BIN)
}

def LoadEstimatorClass(Key: str) -> type:
    """
    Import and return the estimator class registered under Key.

    Arg:
       Key: Registry key, e.g. "RandFR".

    Returns:
       Estimator class.

    Raises:
       KeyError: If Key is not registered.
       ImportError: If the package of the estimator is not installed.
    """
    Spec = MODEL_SPECS[Key]
    return getattr(importlib.import_module(Spec.Module), Spec.ClassName)

def BuildModelKwargs(
    Key: str,
    EstimatorClass: type,
    RandomState: int,
    UserParams: dict | None,
) -> dict:
    """
    Build the constructor arguments of an estimator.

    Priority: registry defaults < RandomState < user parameters. RandomState is written to the
    seed parameter already in the registry (random_state or CatBoost's random_seed), otherwise
    to random_state when the estimator accepts it.

    Arg:
       Key: Registry key.
       EstimatorClass: Class returned by LoadEstimatorClass.
       RandomState: Seed to apply.
       UserParams: Optional parameters overriding everything else.

    Returns:
       Dict of constructor keyword arguments.

    Raises:
       None.
    """
    Kwargs = dict(MODEL_SPECS[Key].Params)
    SeedKeys = [Name for Name in ("random_state", "random_seed") if Name in Kwargs]
    if not SeedKeys and "random_state" in inspect.signature(EstimatorClass).parameters:
        SeedKeys = ["random_state"]
    for Name in SeedKeys:
        Kwargs[Name] = RandomState
    Kwargs.update(UserParams or {})
    return Kwargs
