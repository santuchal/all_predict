# Changelog

## 0.3.0 - 2026-05-06

- Rebuilt `all_predict` around a registry-driven comparison engine with safe preprocessing, richer metrics, optional tuning, persistence helpers, plotting utilities, and a package-local CLI.
- Added public aliases `AllClassifier`, `AllRegressor`, `AutoClassifier`, and `AutoRegressor` while keeping `LazyClassifierPlus` and `LazyRegressorPlus` import-compatible.
- Switched packaging to a modern `pyproject.toml` configuration for the `all-predict` PyPI distribution and added the `all-predict` console script entry point.
- Added reporting utilities, CSV and JSON artifact outputs, optional best-model saving, and graceful skipping of missing optional dependencies.
- Rewrote the README, examples, notebook, tests, and GitHub Actions workflows to match the 0.3.0 feature set and release process.
