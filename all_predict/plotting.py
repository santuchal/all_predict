"""Matplotlib-based plotting helpers for model comparison results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .utils import canonicalize_metric_name, metric_sort_ascending


def plot_classification_results(results: pd.DataFrame, metric: str = "roc_auc", top_n: int = 15):
    """Plot the top classification models for a metric."""

    return _plot_ranked_results(results, "classification", metric, top_n)


def plot_regression_results(results: pd.DataFrame, metric: str = "r2", top_n: int = 15):
    """Plot the top regression models for a metric."""

    return _plot_ranked_results(results, "regression", metric, top_n)


def plot_metric_comparison(results: pd.DataFrame, metrics: list[str] | None = None):
    """Plot several metrics side by side for the top models."""

    if results.empty:
        raise ValueError("results is empty")
    if metrics is None:
        metrics = [
            column
            for column in results.columns
            if column not in {"Model", "Status", "Error", "Notes"}
        ][:4]

    frame = results.head(min(10, len(results))).set_index("Model")
    fig, ax = plt.subplots(figsize=(12, 6))
    frame[metrics].plot(kind="bar", ax=ax)
    ax.set_ylabel("Metric Value")
    ax.set_title("Metric comparison across top models")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_time_vs_score(results: pd.DataFrame, score_metric: str):
    """Scatter plot of total runtime against a score metric."""

    if results.empty:
        raise ValueError("results is empty")

    task = (
        "classification"
        if score_metric.lower().startswith(("roc", "acc", "f1", "prec", "recall", "mcc"))
        else "regression"
    )
    metric_name = canonicalize_metric_name(task, score_metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results["Total Time"], results[metric_name], alpha=0.8)
    for _, row in results.iterrows():
        ax.annotate(row["Model"], (row["Total Time"], row[metric_name]), fontsize=8, alpha=0.8)
    ax.set_xlabel("Total Time (s)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Time vs {metric_name}")
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix_for_model(model, X_test, y_test, *, labels=None, normalize=None):
    """Plot a confusion matrix for a fitted classifier."""

    predictions = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=labels,
        normalize=normalize,
        ax=ax,
        colorbar=False,
    )
    fig.tight_layout()
    return fig, ax


def plot_roc_curves_for_top_models(
    models: Mapping[str, object] | object,
    X_test,
    y_test,
    *,
    top_model_names: Iterable[str] | None = None,
    max_models: int = 5,
):
    """Plot ROC curves for a set of fitted classifiers when score outputs are available."""

    model_mapping = _coerce_model_mapping(models)
    if top_model_names is None:
        top_model_names = list(model_mapping.keys())[:max_models]

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0
    for name in top_model_names:
        model = model_mapping.get(name)
        if model is None:
            continue
        try:
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                score = model.decision_function(X_test)
            else:
                continue
            RocCurveDisplay.from_predictions(y_test, score, name=name, ax=ax)
            plotted += 1
        except Exception:
            continue

    if plotted == 0:
        raise ValueError("No supplied models exposed usable ROC scores.")
    ax.set_title("ROC curves for top models")
    fig.tight_layout()
    return fig, ax


def plot_residuals(model, X_test, y_test):
    """Plot prediction residuals for a fitted regressor."""

    predictions = model.predict(X_test)
    residuals = np.asarray(y_test) - np.asarray(predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predictions, residuals, alpha=0.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual plot")
    fig.tight_layout()
    return fig, ax


def _plot_ranked_results(results: pd.DataFrame, task: str, metric: str, top_n: int):
    if results.empty:
        raise ValueError("results is empty")

    metric_name = canonicalize_metric_name(task, metric)
    ascending = metric_sort_ascending(metric_name)
    top = results.sort_values(by=metric_name, ascending=ascending, na_position="last").head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, min(12, 0.45 * len(top)))))
    ax.barh(top["Model"], top[metric_name])
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Model")
    ax.set_title(f"Top {len(top)} models by {metric_name}")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def _coerce_model_mapping(models: Mapping[str, object] | object) -> dict[str, object]:
    if isinstance(models, Mapping):
        return dict(models)
    if hasattr(models, "fitted_models_"):
        return dict(models.fitted_models_)
    raise TypeError("models must be a mapping or an object exposing fitted_models_.")
