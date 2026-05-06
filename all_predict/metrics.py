"""Metric helpers used across model comparisons and tuning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - fallback for older sklearn
    root_mean_squared_error = None


def adjusted_r2_score(r2: float, n_samples: int, n_features: int) -> float:
    """Return adjusted R2 or NaN when it is undefined."""

    if n_samples <= n_features + 1 or np.isnan(r2):
        return float("nan")
    return 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


def safe_mape(y_true: Any, y_pred: Any) -> float:
    """Compute MAPE without exploding on zero targets."""

    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    non_zero_mask = true_values != 0
    if not np.any(non_zero_mask):
        return float("nan")
    return float(
        mean_absolute_percentage_error(true_values[non_zero_mask], pred_values[non_zero_mask])
    )


def compute_classification_metrics(
    *,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    y_score: np.ndarray | None = None,
    custom_metrics: Mapping[str, Callable[..., float]] | None = None,
    model_classes: np.ndarray | list[Any] | None = None,
) -> tuple[dict[str, float], str]:
    """Compute robust binary or multiclass classification metrics."""

    classes = np.asarray(sorted(pd.Series(y_true).dropna().unique().tolist()))
    if model_classes is not None:
        classes = np.asarray(model_classes)
    is_binary = len(classes) == 2

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "F1": float(
            f1_score(y_true, y_pred, average="binary" if is_binary else "weighted", zero_division=0)
        ),
        "F1 Macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 Weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Precision": float(
            precision_score(
                y_true, y_pred, average="binary" if is_binary else "weighted", zero_division=0
            )
        ),
        "Recall": float(
            recall_score(
                y_true, y_pred, average="binary" if is_binary else "weighted", zero_division=0
            )
        ),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "ROC AUC": float("nan"),
        "Average Precision": float("nan"),
        "Log Loss": float("nan"),
    }
    note_parts: list[str] = []

    if is_binary:
        score_vector = None
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                score_vector = y_proba[:, 1]
            elif y_proba.ndim == 1:
                score_vector = y_proba
        if score_vector is None and y_score is not None:
            score_vector = np.asarray(y_score)

        if score_vector is not None:
            metrics["ROC AUC"] = float(roc_auc_score(y_true, score_vector))
            metrics["Average Precision"] = float(average_precision_score(y_true, score_vector))
        else:
            note_parts.append(
                "ROC AUC and Average Precision were unavailable for this binary model."
            )

        if y_proba is not None:
            try:
                metrics["Log Loss"] = float(log_loss(y_true, y_proba))
            except ValueError:
                metrics["Log Loss"] = float("nan")
    else:
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            try:
                metrics["ROC AUC"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
            except ValueError:
                metrics["ROC AUC"] = float("nan")
            try:
                y_true_bin = label_binarize(y_true, classes=classes)
                metrics["Average Precision"] = float(
                    average_precision_score(y_true_bin, y_proba, average="weighted")
                )
            except ValueError:
                metrics["Average Precision"] = float("nan")
            try:
                metrics["Log Loss"] = float(log_loss(y_true, y_proba, labels=classes))
            except ValueError:
                metrics["Log Loss"] = float("nan")
        else:
            note_parts.append(
                "Multiclass ROC AUC, Average Precision, and Log Loss require predict_proba."
            )

    if custom_metrics:
        for name, func in custom_metrics.items():
            metrics[name] = _call_custom_metric(
                func,
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                y_proba=y_proba,
            )

    return metrics, "; ".join(note_parts)


def compute_regression_metrics(
    *,
    y_true: pd.Series,
    y_pred: np.ndarray,
    n_features: int,
    custom_metrics: Mapping[str, Callable[..., float]] | None = None,
) -> tuple[dict[str, float], str]:
    """Compute regression metrics with safe adjusted R2 and MAPE handling."""

    r2 = float(r2_score(y_true, y_pred))
    metrics = {
        "R2": r2,
        "Adjusted R2": float(adjusted_r2_score(r2, len(y_true), n_features)),
        "RMSE": float(_rmse(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Median AE": float(median_absolute_error(y_true, y_pred)),
        "MAPE": float(safe_mape(y_true, y_pred)),
        "Explained Variance": float(explained_variance_score(y_true, y_pred)),
        "Max Error": float(max_error(y_true, y_pred)),
    }

    if custom_metrics:
        for name, func in custom_metrics.items():
            metrics[name] = _call_custom_metric(func, y_true=y_true, y_pred=y_pred)

    note = ""
    if np.isnan(metrics["Adjusted R2"]):
        note = "Adjusted R2 is undefined when n <= p + 1."
    return metrics, note


def _call_custom_metric(
    func: Callable[..., float],
    *,
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
    y_proba: Any | None = None,
) -> float:
    try:
        return float(func(y_true, y_pred))
    except TypeError:
        try:
            return float(func(y_true, y_pred, y_score))
        except TypeError:
            return float(func(y_true, y_pred, y_score, y_proba))


def _rmse(y_true: Any, y_pred: Any) -> float:
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
