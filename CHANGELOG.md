# Changelog

All notable changes to koco-ml are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version may contain breaking changes;
they are always listed under **Changed** or **Removed**.

## [Unreleased]

### Fixed

- `BootstrapCompare` no longer raises `NameError` when it reports progress on large inputs.

### Added

- GitHub Actions workflow running ruff and the test suite on Python 3.10 to 3.13.
- ruff configuration in `pyproject.toml`; `ruff` added to the `dev` extra.

### Changed

- Packaging is configured only in `pyproject.toml`; `setup.py` was removed. The built wheel and
  source archive keep the same contents.

## [0.1.1] - 2026-10-30

### Added

- `model_selection` module: `SelectBestModel`, `EvaluateOnHoldout` and `BootstrapCompare`, with
  English and Turkish documentation.

### Changed

- Development-only files are excluded from Git and from the source and wheel distributions.

## [0.1.0] - 2026-10-21

First release on PyPI as `koco-ml`.

### Added

- `general`: SQL Server connection, Excel / Parquet / Joblib / JSON import and export, model
  artifacts and model bundles with SHA-256 checks.
- `analysis`: data summaries, binning, cross checks, outlier, target divergence and
  representation reports, charts.
- `preprocessing`: null, outlier, WOE and scaling plans (fit on Train, apply unchanged).
- `feature_selection`: correlation, VIF, model-based feature selection, backward selection.
- `modelling`: input checks, train/test split, regression and binary classification training.
- `metrics`, `scoring`, `registry` and `guides` modules.

[Unreleased]: https://github.com/ffarukkoc/koco-ml/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ffarukkoc/koco-ml/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ffarukkoc/koco-ml/releases/tag/v0.1.0
