"""Small utility helpers shared across the package."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .exceptions import InvalidTaskError

CLASSIFICATION_METRIC_ALIASES = {
    "accuracy": "Accuracy",
    "balancedaccuracy": "Balanced Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "balanced accuracy": "Balanced Accuracy",
    "rocauc": "ROC AUC",
    "roc_auc": "ROC AUC",
    "roc auc": "ROC AUC",
    "auc": "ROC AUC",
    "averageprecision": "Average Precision",
    "average_precision": "Average Precision",
    "average precision": "Average Precision",
    "f1": "F1",
    "f1macro": "F1 Macro",
    "f1_macro": "F1 Macro",
    "f1 macro": "F1 Macro",
    "f1weighted": "F1 Weighted",
    "f1_weighted": "F1 Weighted",
    "f1 weighted": "F1 Weighted",
    "precision": "Precision",
    "recall": "Recall",
    "mcc": "MCC",
    "logloss": "Log Loss",
    "log_loss": "Log Loss",
    "log loss": "Log Loss",
    "traintime": "Train Time",
    "train_time": "Train Time",
    "train time": "Train Time",
    "predicttime": "Predict Time",
    "predict_time": "Predict Time",
    "predict time": "Predict Time",
    "totaltime": "Total Time",
    "total_time": "Total Time",
    "total time": "Total Time",
}

REGRESSION_METRIC_ALIASES = {
    "r2": "R2",
    "adjustedr2": "Adjusted R2",
    "adjusted_r2": "Adjusted R2",
    "adjusted r2": "Adjusted R2",
    "rmse": "RMSE",
    "mae": "MAE",
    "medianae": "Median AE",
    "median_ae": "Median AE",
    "median ae": "Median AE",
    "mape": "MAPE",
    "explainedvariance": "Explained Variance",
    "explained_variance": "Explained Variance",
    "explained variance": "Explained Variance",
    "maxerror": "Max Error",
    "max_error": "Max Error",
    "max error": "Max Error",
    "traintime": "Train Time",
    "train_time": "Train Time",
    "predicttime": "Predict Time",
    "predict_time": "Predict Time",
    "totaltime": "Total Time",
    "total_time": "Total Time",
}

LOWER_IS_BETTER = {
    "Log Loss",
    "RMSE",
    "MAE",
    "Median AE",
    "MAPE",
    "Max Error",
    "Train Time",
    "Predict Time",
    "Total Time",
}


def ensure_dataframe(data: Any) -> pd.DataFrame:
    """Return the input as a DataFrame with stable column names."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        for column in frame.columns:
            if is_bool_dtype(frame[column].dtype):
                frame[column] = frame[column].astype(object)
        return frame.where(pd.notna(frame), np.nan)
    if isinstance(data, pd.Series):
        frame = data.to_frame()
        if is_bool_dtype(frame.iloc[:, 0].dtype):
            frame.iloc[:, 0] = frame.iloc[:, 0].astype(object)
        return frame.where(pd.notna(frame), np.nan)
    if hasattr(data, "toarray"):
        data = data.toarray()
    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    columns = [f"feature_{index}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns).where(pd.notna(array), np.nan)


def ensure_series(data: Any, name: str = "target") -> pd.Series:
    """Return the input as a Series."""

    if isinstance(data, pd.Series):
        series = data.copy()
        if is_bool_dtype(series.dtype):
            series = series.astype(object)
        return series.where(pd.notna(series), np.nan)
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Expected a 1-column DataFrame when converting to Series.")
        series = data.iloc[:, 0].copy()
        if is_bool_dtype(series.dtype):
            series = series.astype(object)
        return series.where(pd.notna(series), np.nan)
    series = pd.Series(data, name=name)
    if is_bool_dtype(series.dtype):
        series = series.astype(object)
    return series.where(pd.notna(series), np.nan)


def canonicalize_metric_name(task: str, metric: str) -> str:
    """Map a user-facing metric alias to the stored result column name."""

    normalized = metric.strip().lower().replace("-", " ")
    collapsed = normalized.replace(" ", "")
    aliases = (
        CLASSIFICATION_METRIC_ALIASES if task == "classification" else REGRESSION_METRIC_ALIASES
    )
    if normalized in aliases:
        return aliases[normalized]
    if collapsed in aliases:
        return aliases[collapsed]
    return metric


def metric_sort_ascending(metric_name: str) -> bool:
    """Return True when lower metric values are better."""

    return metric_name in LOWER_IS_BETTER


def parse_model_list(
    value: Sequence[str] | str | None,
    *,
    allow_all: bool = False,
) -> Sequence[str] | str | None:
    """Normalize model name inputs from strings or sequences."""

    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if allow_all and cleaned.lower() == "all":
            return "all"
        if not cleaned:
            return None
        return [item.strip() for item in cleaned.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def infer_task_from_target(target: Any) -> str:
    """Best-effort task inference for CLI auto mode."""

    y = ensure_series(target)
    if not is_numeric_dtype(y):
        return "classification"

    unique_count = int(y.nunique(dropna=False))
    sample_count = len(y)
    if unique_count <= min(20, max(2, sample_count // 10)):
        return "classification"
    if unique_count <= 10 and unique_count < sample_count:
        return "classification"
    return "regression"


def require_known_task(task: str) -> str:
    """Validate a task string."""

    normalized = task.strip().lower()
    aliases = {
        "classify": "classification",
        "classification": "classification",
        "classifier": "classification",
        "regress": "regression",
        "regression": "regression",
        "regressor": "regression",
        "infer": "infer",
        "auto": "infer",
    }
    if normalized not in aliases:
        raise InvalidTaskError(f"Unsupported task value: {task}")
    return aliases[normalized]


def format_exception(exc: Exception) -> str:
    """Render a compact exception message."""

    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"


def output_path(path_like: str | Path) -> Path:
    """Return a normalized Path instance."""

    return path_like if isinstance(path_like, Path) else Path(path_like)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Backward-compatible regression metric helper."""

    from .metrics import compute_regression_metrics

    values, _ = compute_regression_metrics(
        y_true=ensure_series(y_true),
        y_pred=np.asarray(y_pred),
        n_features=1,
    )
    return {key: values[key] for key in ["R2", "RMSE", "MAE"]}


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None,
) -> dict[str, float]:
    """Backward-compatible classification metric helper."""

    from .metrics import compute_classification_metrics

    values, _ = compute_classification_metrics(
        y_true=ensure_series(y_true),
        y_pred=np.asarray(y_pred),
        y_proba=None if y_proba is None else np.asarray(y_proba),
    )
    return {
        "Accuracy": values["Accuracy"],
        "F1": values["F1 Weighted"],
        "AUC": values["ROC AUC"],
    }


def check_missing_values(X: Any) -> bool:
    """Return True when the input appears to contain missing values."""

    frame = ensure_dataframe(X)
    return bool(frame.isna().any().any())
