# koco

koco is a Python toolkit for regression and binary classification on tabular data. It covers the
whole model life cycle: **Analysis → Preprocessing → Feature selection → Modelling → Scoring**.

Preprocessing steps are stored as reusable *plans*: a plan is fitted once on training data and
applied unchanged to test data and new data. A trained model and its plans can be saved together in
one file and used to score raw data.

📘 **Workflow, end-to-end examples and function reference: [GUIDE.md](https://github.com/ffarukkoc/koco-ml/blob/main/GUIDE.md)**

## Requirements

- Python 3.10 or later
- pandas, numpy, matplotlib, openpyxl, scikit-learn, scipy and joblib (installed automatically)
- Optional features need extra packages (see [Optional dependencies](#optional-dependencies))

## Installation

> **Note:** koco is published on PyPI as **`koco-ml`** and imported as `koco`. The PyPI project
> named `koco` is an unrelated package; do not install it.

> **Note for macOS:** the commands below use `python`. On macOS, `python` is often unavailable outside
> a virtual environment; use `python3` instead.

A virtual environment keeps koco and its dependencies separate from other projects:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### From PyPI

```bash
python -m pip install koco-ml             # core features
python -m pip install "koco-ml[all]"      # core features + all optional dependencies
```

### Installing from the source folder

Run the commands from the folder that contains `pyproject.toml`:

```bash
python -m pip install ".[all]"         # standard install: the code is copied into the environment
python -m pip install -e ".[all]"      # editable install: the environment uses the code in this folder
```

- A **standard install** works independently of the folder; the folder can be moved or deleted
  afterwards.
- An **editable install** reflects every change in the folder immediately. It suits development and
  `git pull` updates. The folder must stay in place (see [Troubleshooting](#troubleshooting)).
- pip installs the dependencies, which requires internet access. On an offline machine where all
  dependencies are already installed, add `--no-deps --no-build-isolation`.
- Copying the `koco/` folder into `site-packages` by hand is not supported: it installs no
  dependencies and cannot be updated or removed with pip.

### Optional dependencies

Each extra is imported only by the features that need it.

| Extra | Packages | Needed by |
|---|---|---|
| `sql` | SQLAlchemy, pyodbc | `Connection` |
| `parquet` | pyarrow | `ExportParquet`, `ImportParquet` |
| `stats` | statsmodels | `VIFReport`, `BackwardSelection` |
| `boosting` | xgboost, lightgbm, catboost | `XGB*`, `LGBM*`, `CatBoost*` algorithm keys |
| `dev` | pytest | the test suite |
| `all` | sql + parquet + stats + boosting | everything except `dev` |

`Connection` also needs the *ODBC Driver 17 for SQL Server*. Connection settings are described in
[GUIDE.md](https://github.com/ffarukkoc/koco-ml/blob/main/GUIDE.md#sql-server-configuration). On macOS / Linux, also follow
[ODBC and Kerberos setup](https://github.com/ffarukkoc/koco-ml/blob/main/GUIDE.md#macos--linux-odbc-and-kerberos-setup) (driver registration and `krb5.conf`).

### Checking the install

```bash
python -c "import koco; print(koco.__version__, koco.__file__)"
```

The printed path shows which copy is in use: `site-packages` for a standard install, the source
folder for an editable install. In a notebook, the kernel must use the same interpreter; check it
with `import sys; print(sys.executable)` and restart the kernel after installing.

### Updating

- From PyPI: `python -m pip install --upgrade koco-ml`
- Standard install from a folder: run the install command again from the new version of the folder.
- Editable install: update the folder (for example with `git pull`). If the version number changed,
  run the install command again so that `pip list` shows the new version.

### Uninstalling

```bash
python -m pip uninstall koco-ml
```

The source folder is not affected.

## Quickstart

Fit a null-handling plan on training data and apply the same values to new data:

```python
import numpy as np
import pandas as pd
import koco

Train = pd.DataFrame({"Income": [3200, np.nan, 4100, 5000, 2800],
                      "Age":    [25, 40, np.nan, 35, 52]})
New = pd.DataFrame({"Income": [np.nan, 3900],
                    "Age":    [30, np.nan]})

Plan, Check = koco.FitNullPlan(Train, [("Income", "Median"), ("Age", "Mean")])
koco.ExportJson(Plan, "Plans/NullPlan.json")   # store the plan for later scoring

print(koco.ApplyNullPlan(New, Plan))
#    Income   Age
# 0  3650.0  30.0     <- Income median from Train
# 1  3900.0  38.0     <- Age mean from Train
```

Complete regression and binary classification examples, from loading data to scoring with a model
bundle, are in [GUIDE.md](https://github.com/ffarukkoc/koco-ml/blob/main/GUIDE.md#end-to-end-example).

## Conventions

- Functions, parameters and variables use PascalCase. The input DataFrame parameter is always `Table`.
- `ModelType` takes `"Regression"` or `"Binary"`.
- Algorithm keys end with `R` (regression) or `C` (binary classification), e.g. `RandFR` / `RandFC`.
  `koco.ModelRegistryGuide()` lists them.
- Fitted plans are plain dicts of the form `{"Columns": {...}}` and can be saved with `ExportJson`.
- Every `Export*` function takes `FilePath`: `None` writes `<yymmdd>_<FunctionName>.<ext>` to the
  current folder; a path with an extension is used as the file name; a folder receives the default
  file name. See [Output files and messages](https://github.com/ffarukkoc/koco-ml/blob/main/GUIDE.md#output-files-and-messages) for details.
- Printed messages use one symbol set: `❗` needs attention, `ℹ` information, `✓` completed,
  `•` progress.

## Troubleshooting

### `ModuleNotFoundError: No module named 'koco'`

- **Notebook started before the install:** an installed package is detected only when the
  interpreter starts. Restart the kernel; clearing the outputs is not enough.
- **Different interpreter:** compare `sys.executable` in the notebook with the `python` used for the
  install.
- **Local file named `koco.py`:** a file with this name next to the notebook or script shadows the
  package. Rename it and delete its `__pycache__` folder.
- **Editable install after moving the folder:** the environment still points to the old location.
  Run the editable install again from the new location.
- **New Python version:** packages are installed per Python version. Install koco again with the new
  interpreter.

### `import koco` succeeds, but the functions are missing

The outer project folder (the one with `pyproject.toml`) is being imported instead of the package.
This happens when that folder is copied into `site-packages` or added to `sys.path`. Remove the copy
and install with pip.

### A notebook does not see code changes (editable install)

A running kernel keeps the old code in memory. Restart the kernel, or reload the changed module:

```python
import importlib, koco
importlib.reload(koco.analysis)
importlib.reload(koco)
```

### `pip list` shows an old version

The version in the package metadata is written at install time. Run the install command again.

### `Connection` fails on macOS / Linux

- **`Can't open lib 'ODBC Driver 17 for SQL Server' : file not found`:** unixODBC cannot find the driver registration.
  Check with `odbcinst -q -d`, then add `"ODBCSYSINI"` (folder of `odbcinst.ini`) to the SQL configuration file.
- **`Cannot generate SSPI context` or other GSSAPI / Kerberos errors:** check the ticket with `klist` (renew with `kinit`)
  and, if `krb5.conf` is not in `/etc`, add `"KRB5_CONFIG"` to the SQL configuration file.
- Restart the kernel after changing these settings. See
  [ODBC and Kerberos setup](https://github.com/ffarukkoc/koco-ml/blob/main/GUIDE.md#macos--linux-odbc-and-kerberos-setup).

## Development

Development uses an editable install with the test dependencies:

```bash
python -m pip install -e ".[dev,all]"
python -m pytest
```

The test suite also checks that every null strategy, scaling method, algorithm key and metric has
an entry in the guides.

### Project layout

```
koco/                        <- project root
├── pyproject.toml           <- package metadata and dependencies
├── README.md
├── GUIDE.md                 <- workflow and function reference
├── tests/                   <- pytest suite
└── koco/                    <- the package
    ├── __init__.py          <- version; re-exports the public names of every module
    ├── _helpers.py          <- private helpers shared by the modules
    ├── general.py           <- database connection, import/export, model bundles
    ├── analysis.py          <- data summaries, binning, outlier and representation reports, charts
    ├── preprocessing.py     <- null, outlier, WOE and scaling plans
    ├── feature_selection.py <- correlation, VIF, feature selection
    ├── registry.py          <- algorithm registry (ModelSpec, MODEL_SPECS)
    ├── metrics.py           <- regression and classification metrics
    ├── modelling.py         <- input checks, train/test split, model training
    ├── scoring.py           <- predictions and model bundle scoring
    └── guides.py            <- explanatory guides for strategies, methods, algorithms and metrics
```

Dependency rules (no circular imports):

- `_helpers` imports nothing from koco; `registry` and `metrics` import only `_helpers`.
- `guides` may import any module; no module imports `guides`.

### Extending koco

- **Function:** add it to the matching module and to that module's `__all__`, then document it in
  `GUIDE.md`. `__init__.py` re-exports every `__all__`, so `koco.NewFunction()` works without
  further changes. If two modules export the same name, `import koco` raises an `ImportError`.
- **Algorithm:** add one `ModelSpec(...)` entry to `MODEL_SPECS` in `koco/registry.py`. The
  registries, `ModelRegistryGuide` and the checks in `CheckModellingInputs` are derived from it.
  Keys end with `R` or `C`.
- **Metric:** add its column to `PERFORMANCE_COLUMNS` in `koco/metrics.py` and its English and
  Turkish text to `_METRIC_GUIDE` in `koco/guides.py`.
- **Module:** create the module (for example `koco/newgroup.py`) with an `__all__` list and add these lines to
  `koco/__init__.py`:

  ```python
  from . import newgroup
  from .newgroup import *
  __all__ += newgroup.__all__
  ```

### Coding conventions

- One parameter per line in function definitions.
- `pandas`, `numpy` and `openpyxl` are imported at module level. Heavy or optional packages
  (sqlalchemy, matplotlib, scikit-learn, statsmodels, scipy, joblib) are imported inside the
  functions that use them.
- Names used by several modules (`MODEL_SPECS`, `PERFORMANCE_COLUMNS`, `NULL_STRATEGIES`, ...) have
  no leading `_`; names used inside one module start with `_`.
- Docstrings use these sections in this order. `Raises` is `None.` when nothing is raised; `Notes`
  and `Notes (TR)` appear only when needed. Text is in English; Turkish text appears only in
  `Notes (TR)`. Headings are indented 4 spaces and entries 7 spaces.

  ```
  """
  One-line summary.

  Optional longer description.

  Arg:
     Name: Description.

  Returns:
     Description.

  Raises:
     ValueError: When ...

  Notes:
     Optional English notes.

  Notes (TR):
     Optional Turkish notes.
  """
  ```

### Versioning

The version is defined only in `koco/__init__.py` (`__version__`); `pyproject.toml` reads it from
there. After changing it, run the install command again so that the package metadata is updated.
