# all_predict

[![PyPI version](https://img.shields.io/pypi/v/all-predict.svg)](https://pypi.org/project/all-predict/)
[![Python versions](https://img.shields.io/pypi/pyversions/all-predict.svg)](https://pypi.org/project/all-predict/)
[![Tests](https://github.com/santuchal/all_predict/actions/workflows/tests.yml/badge.svg)](https://github.com/santuchal/all_predict/actions/workflows/tests.yml)
[![License](https://img.shields.io/pypi/l/all-predict.svg)](LICENSE)

`all_predict` is a lightweight but robust automated model comparison library for supervised machine learning. It benchmarks many classifiers and regressors with safe preprocessing, useful metrics, optional tuning, CLI support, and PyPI-ready packaging.

It is inspired by LazyPredict. It is not a replacement for careful feature engineering, external validation, calibration, or domain-specific model review. It is useful for fast baseline discovery, teaching, rapid screening, and early model selection. It does not guarantee the best final accuracy.

## Installation

```bash
pip install all-predict
pip install "all-predict[boost]"
pip install "all-predict[all]"
```

## Quick Classification Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from all_predict import AllClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

clf = AllClassifier(
    verbose=True,
    random_state=42,
    n_jobs=-1,
    ignore_warnings=True,
    sort_by="roc_auc",
    predictions=True,
    tune=False,
)

models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models.head())
```

## Quick Regression Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from all_predict import AllRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

reg = AllRegressor(
    verbose=True,
    random_state=42,
    n_jobs=-1,
    ignore_warnings=True,
    sort_by="r2",
    predictions=True,
    tune=False,
)

models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models.head())
```

## Mixed-Type DataFrame Example

```python
import pandas as pd

from all_predict import AllClassifier

frame = pd.DataFrame(
    {
        "age": [35, 41, 29, None, 50, 38],
        "bmi": [22.1, None, 31.2, 27.5, 29.1, 24.7],
        "city": ["Kolkata", "Delhi", None, "Mumbai", "Delhi", "Kolkata"],
        "smoker": pd.Series([True, False, True, None, False, True], dtype="boolean"),
        "target": [1, 0, 1, 0, 0, 1],
    }
)

clf = AllClassifier(
    verbose=False,
    predictions=True,
    include_models=["LogisticRegression", "RandomForestClassifier", "DummyClassifier"],
)

results, predictions = clf.fit(frame.drop(columns=["target"]), frame["target"], test_size=0.33)
print(results[["Model", "Accuracy", "ROC AUC"]])
print(predictions.head())
```

## CLI

```bash
all-predict classify --file data.csv --target diagnosis --output runs/breast_cancer --sort-by roc_auc --predictions --save-best
all-predict regress --file data.csv --target target --output runs/diabetes --sort-by r2 --predictions --save-best
all-predict infer --file data.csv --target target --output runs/auto
```

The CLI writes:

- `model_results.csv`
- `predictions.csv` when `--predictions` is enabled
- `failed_models.csv` when failures occur
- `skipped_models.csv` when optional models are unavailable or a timeout skips work
- `run_summary.json`
- `best_model.joblib` when `--save-best` is enabled

## Metrics

### Classification Metrics

| Metric | Meaning |
| --- | --- |
| Accuracy | Overall label accuracy |
| Balanced Accuracy | Mean recall across classes |
| ROC AUC | Binary or multiclass OVR ROC AUC when scores are available |
| Average Precision | Binary or multiclass average precision when probabilities are available |
| F1 | Binary F1 for binary tasks, weighted F1 for multiclass |
| F1 Macro | Macro-averaged F1 |
| F1 Weighted | Weighted F1 |
| Precision | Binary precision for binary tasks, weighted precision for multiclass |
| Recall | Binary recall for binary tasks, weighted recall for multiclass |
| MCC | Matthews correlation coefficient |
| Log Loss | Probability-based log loss when available |
| Time | Train, predict, and total runtime columns |

### Regression Metrics

| Metric | Meaning |
| --- | --- |
| R2 | Coefficient of determination |
| Adjusted R2 | Adjusted R2 with safe NaN handling for small samples |
| RMSE | Root mean squared error |
| MAE | Mean absolute error |
| Median AE | Median absolute error |
| MAPE | Mean absolute percentage error with zero-safe handling |
| Explained Variance | Explained variance score |
| Max Error | Worst absolute error |
| Time | Train, predict, and total runtime columns |

## Tuning

Fast comparison is the default. Tuning is optional and intentionally limited.

```python
from all_predict import AllClassifier

clf = AllClassifier(
    tune=True,
    tune_top_n=3,
    tuner="randomized",
    cv=5,
    sort_by="roc_auc",
)
```

Use tuning carefully. It increases runtime and can overfit the chosen validation split.

## Saving and Loading Models

```python
from all_predict import AllClassifier, load_model, save_model

clf = AllClassifier(save_best=False)
results, _ = clf.fit(X_train, X_test, y_train, y_test)

best_model = clf.best_model_
save_model(best_model, "runs/best_model.joblib")
loaded_model = load_model("runs/best_model.joblib")
preds = loaded_model.predict(X_test)
```

## Output Directory Example

```text
runs/breast_cancer/
├── best_model.joblib
├── failed_models.csv
├── model_results.csv
├── predictions.csv
├── run_summary.json
├── skipped_models.csv
└── tuned_results.csv
```

## Public API

```python
from all_predict import AllClassifier, AllRegressor, AutoClassifier, AutoRegressor
from all_predict.classification import LazyClassifierPlus
from all_predict.regression import LazyRegressorPlus
```

The old imports remain valid. The new aliases are the preferred names for new code.

## Comparison with LazyPredict

LazyPredict is mature and popular. `all_predict` is a smaller project with a narrower goal. This release focuses on:

- safer preprocessing with `Pipeline` and `ColumnTransformer`
- richer classification and regression metrics
- optional tuning instead of always-on heavy search
- CSV and JSON reporting utilities
- CLI outputs and model persistence
- clearer examples and tests

It does not claim to outperform LazyPredict on accuracy across datasets. That would require controlled benchmarking.

## PyPI Release Note

This repository is intended for publication as `all-predict` version `0.3.0`.

## Development

```bash
python -m pip install -e ".[dev,boost,notebook]"
pytest
ruff check .
black .
python -m build
twine check dist/*
```

## Repository Examples

- `examples/classification_breast_cancer.py`
- `examples/regression_diabetes.py`
- `examples/all_predict_0_3_0_brutal_walkthrough.ipynb`

## License

MIT

## Author

Santu Chall  
santuchal@gmail.com
