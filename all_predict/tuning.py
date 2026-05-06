"""Optional hyperparameter tuning for the top-ranked models."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .metrics import compute_classification_metrics, compute_regression_metrics
from .preprocessing import build_model_pipeline, get_effective_feature_count
from .registry import ModelSpec, make_estimator
from .utils import canonicalize_metric_name, format_exception, metric_sort_ascending


@dataclass(frozen=True)
class TuningConfig:
    """Configuration passed into tuning helpers."""

    task: str
    sort_by: str | None
    tuner: str
    cv: int
    random_state: int | None
    n_jobs: int | None
    preprocess: bool
    categorical_encoder: str
    scale_numeric: str | bool
    return_train_score: bool
    timeout: float | None
    custom_metrics: Mapping[str, Callable[..., float]] | None
    use_gpu: bool = False
    ignore_warnings: bool = True
    verbose: bool = False


def tune_selected_models(
    *,
    specs: list[ModelSpec],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: TuningConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune a list of already-ranked model specs."""

    result_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    start_time = perf_counter()

    for spec in specs:
        if not spec.param_grid:
            result_rows.append(
                {
                    "Model": spec.name,
                    "Status": "skipped",
                    "Error": "",
                    "Search Type": "",
                    "Best CV Score": float("nan"),
                    "Best Params": "",
                }
            )
            continue

        if config.timeout is not None and (perf_counter() - start_time) >= config.timeout:
            result_rows.append(
                {
                    "Model": spec.name,
                    "Status": "skipped",
                    "Error": (
                        f"Skipped because timeout={config.timeout} seconds "
                        "was reached before tuning started."
                    ),
                    "Search Type": "",
                    "Best CV Score": float("nan"),
                    "Best Params": "",
                }
            )
            continue

        estimator = make_estimator(spec, config.random_state, config.n_jobs, config.use_gpu)
        pipeline = build_model_pipeline(
            estimator=estimator,
            X_train=X_train,
            preprocess=config.preprocess,
            needs_scaling=spec.needs_scaling,
            categorical_encoder=config.categorical_encoder,
            scale_numeric=config.scale_numeric,
        )
        prefixed_grid = {f"model__{key}": value for key, value in spec.param_grid.items()}
        search_type = config.tuner.lower()
        scoring = _resolve_scoring(config.task, config.sort_by, y_train.nunique(dropna=False))
        search = _build_search(
            pipeline=pipeline,
            prefixed_grid=prefixed_grid,
            scoring=scoring,
            config=config,
            search_type=search_type,
        )

        try:
            with warnings.catch_warnings():
                if config.ignore_warnings:
                    warnings.simplefilter("ignore")
                train_start = perf_counter()
                search.fit(X_train, y_train)
                train_time = perf_counter() - train_start
                tuned_model = search.best_estimator_
                predict_start = perf_counter()
                preds = tuned_model.predict(X_test)
                predict_time = perf_counter() - predict_start
        except Exception as exc:
            result_rows.append(
                {
                    "Model": spec.name,
                    "Status": "failed",
                    "Error": format_exception(exc),
                    "Search Type": search_type,
                    "Best CV Score": float("nan"),
                    "Best Params": "",
                }
            )
            continue

        if config.task == "classification":
            proba = None
            decision = None
            if hasattr(tuned_model, "predict_proba"):
                try:
                    proba = tuned_model.predict_proba(X_test)
                except Exception:
                    proba = None
            if hasattr(tuned_model, "decision_function"):
                try:
                    decision = tuned_model.decision_function(X_test)
                except Exception:
                    decision = None
            metric_values, note = compute_classification_metrics(
                y_true=y_test,
                y_pred=preds,
                y_proba=proba,
                y_score=decision,
                custom_metrics=config.custom_metrics,
                model_classes=(
                    getattr(tuned_model.named_steps["model"], "classes_", None)
                    if hasattr(tuned_model, "named_steps")
                    else getattr(tuned_model, "classes_", None)
                ),
            )
        else:
            metric_values, note = compute_regression_metrics(
                y_true=y_test,
                y_pred=preds,
                n_features=get_effective_feature_count(tuned_model, X_test),
                custom_metrics=config.custom_metrics,
            )

        result_rows.append(
            {
                "Model": spec.name,
                **metric_values,
                "Train Time": train_time,
                "Predict Time": predict_time,
                "Total Time": train_time + predict_time,
                "Best CV Score": float(search.best_score_),
                "Best Params": json.dumps(search.best_params_, sort_keys=True),
                "Search Type": search_type,
                "Status": "success",
                "Error": note,
            }
        )
        fitted_models[spec.name] = tuned_model

    results = pd.DataFrame(result_rows)
    if not results.empty:
        sort_metric = canonicalize_metric_name(
            config.task, config.sort_by or ("roc_auc" if config.task == "classification" else "r2")
        )
        if sort_metric in results.columns:
            results = results.sort_values(
                by=sort_metric,
                ascending=metric_sort_ascending(sort_metric),
                na_position="last",
            ).reset_index(drop=True)
    return results, fitted_models


def _build_search(
    *,
    pipeline,
    prefixed_grid: dict[str, Any],
    scoring: str,
    config: TuningConfig,
    search_type: str,
):
    common_kwargs = {
        "cv": config.cv,
        "n_jobs": config.n_jobs,
        "scoring": scoring,
        "return_train_score": config.return_train_score,
    }
    if search_type == "grid":
        return GridSearchCV(pipeline, prefixed_grid, **common_kwargs)
    if search_type != "randomized":
        raise ValueError("tuner must be either 'grid' or 'randomized'")

    discrete_space = max(1, math.prod(len(values) for values in prefixed_grid.values()))
    n_iter = min(10, discrete_space)
    return RandomizedSearchCV(
        pipeline,
        prefixed_grid,
        n_iter=n_iter,
        random_state=config.random_state,
        **common_kwargs,
    )


def _resolve_scoring(task: str, sort_by: str | None, unique_class_count: int) -> str:
    metric = canonicalize_metric_name(
        task, sort_by or ("roc_auc" if task == "classification" else "r2")
    )
    scoring_map = {
        "Accuracy": "accuracy",
        "Balanced Accuracy": "balanced_accuracy",
        "ROC AUC": "roc_auc_ovr_weighted" if unique_class_count > 2 else "roc_auc",
        "F1": "f1_weighted" if unique_class_count > 2 else "f1",
        "F1 Macro": "f1_macro",
        "F1 Weighted": "f1_weighted",
        "Precision": "precision_weighted" if unique_class_count > 2 else "precision",
        "Recall": "recall_weighted" if unique_class_count > 2 else "recall",
        "Log Loss": "neg_log_loss",
        "R2": "r2",
        "Adjusted R2": "r2",
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "Median AE": "neg_median_absolute_error",
        "MAPE": "neg_mean_absolute_percentage_error",
        "Explained Variance": "explained_variance",
    }
    return scoring_map.get(metric, "accuracy" if task == "classification" else "r2")
