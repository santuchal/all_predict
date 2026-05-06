"""Shared comparison engine used by classifiers and regressors."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, ClassVar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .exceptions import DataValidationError
from .metrics import compute_classification_metrics, compute_regression_metrics
from .persistence import save_model
from .preprocessing import build_model_pipeline, get_effective_feature_count
from .registry import ModelSpec, get_model_registry, is_dependency_available, make_estimator
from .reporting import save_dataframe, save_predictions, save_results, save_run_summary
from .tuning import TuningConfig, tune_selected_models
from .utils import (
    canonicalize_metric_name,
    ensure_dataframe,
    ensure_series,
    format_exception,
    metric_sort_ascending,
    parse_model_list,
)


@dataclass
class BasePredictor:
    """Base class for comparing many estimators on a single tabular dataset."""

    verbose: bool = True
    ignore_warnings: bool = True
    random_state: int = 42
    n_jobs: int = -1
    sort_by: str | None = None
    predictions: bool = False
    include_models: Sequence[str] | str | None = None
    exclude_models: Sequence[str] | str | None = None
    models: Sequence[str] | str = "all"
    max_models: int | None = None
    preprocess: bool = True
    categorical_encoder: str = "onehot"
    scale_numeric: str | bool = "auto"
    tune: bool = False
    tune_top_n: int = 3
    tuner: str = "randomized"
    cv: int = 5
    timeout: float | None = None
    save_best: bool = False
    output_dir: str | Path | None = None
    return_train_score: bool = False
    custom_metrics: Mapping[str, Callable[..., float]] | None = None
    use_gpu: bool = False
    progress: bool = True
    fail_fast: bool = False

    task: ClassVar[str] = ""

    fitted_models_: dict[str, Any] = field(init=False, default_factory=dict)
    results_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    predictions_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    failed_models_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    skipped_models_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    tuned_results_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    tuned_models_: dict[str, Any] = field(init=False, default_factory=dict)
    best_model_: Any = field(init=False, default=None)
    best_model_name_: str | None = field(init=False, default=None)
    best_model_source_: str = field(init=False, default="comparison")
    best_model_path_: Path | None = field(init=False, default=None)
    run_summary_: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError("Subclasses must define task as 'classification' or 'regression'.")
        self.include_models = parse_model_list(self.include_models)
        self.exclude_models = parse_model_list(self.exclude_models)
        self.models = parse_model_list(self.models, allow_all=True)
        self._logger = logging.getLogger(f"all_predict.{self.__class__.__name__}")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

    def fit(
        self,
        X: Any,
        X_test_or_y: Any,
        y_train: Any | None = None,
        y_test: Any | None = None,
        *,
        test_size: float = 0.2,
        random_state: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit and evaluate many models.

        Supports:
        - fit(X_train, X_test, y_train, y_test)
        - fit(X, y, test_size=0.2, random_state=42)
        """

        start_time = perf_counter()
        run_random_state = self.random_state if random_state is None else random_state
        X_train_df, X_test_df, y_train_sr, y_test_sr = self._resolve_fit_inputs(
            X,
            X_test_or_y,
            y_train,
            y_test,
            test_size=test_size,
            random_state=run_random_state,
        )

        specs = self._select_specs()
        result_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        skipped_rows: list[dict[str, Any]] = []
        prediction_columns: dict[str, pd.Series] = {}
        metric_notes: dict[str, str] = {}

        self.fitted_models_.clear()
        self.tuned_models_.clear()
        self.best_model_ = None
        self.best_model_name_ = None
        self.best_model_source_ = "comparison"
        self.best_model_path_ = None

        for spec in specs:
            if self._timed_out(start_time):
                skipped_rows.append(
                    {
                        "Model": spec.name,
                        "Status": "skipped",
                        "Reason": f"Skipped because timeout={self.timeout} seconds was reached.",
                        "Dependency": spec.optional_dependency or "",
                    }
                )
                continue

            if spec.optional_dependency and not is_dependency_available(spec.optional_dependency):
                skipped_rows.append(
                    {
                        "Model": spec.name,
                        "Status": "skipped",
                        "Reason": (
                            f"Optional dependency '{spec.optional_dependency}' " "is not installed."
                        ),
                        "Dependency": spec.optional_dependency,
                    }
                )
                continue

            try:
                row, fitted_model, preds, note = self._fit_single_model(
                    spec=spec,
                    X_train=X_train_df,
                    X_test=X_test_df,
                    y_train=y_train_sr,
                    y_test=y_test_sr,
                )
            except Exception as exc:  # pragma: no cover
                if self.fail_fast:
                    raise
                failure_rows.append(
                    {
                        "Model": spec.name,
                        "Status": "failed",
                        "Error": format_exception(exc),
                    }
                )
                if self.verbose:
                    self._logger.warning(f"{spec.name} failed: {format_exception(exc)}")
                continue

            result_rows.append(row)
            self.fitted_models_[spec.name] = fitted_model
            if note:
                metric_notes[spec.name] = note
            if self.predictions:
                prediction_columns[spec.name] = pd.Series(
                    preds, index=X_test_df.index, name=spec.name
                )

        self.results_ = pd.DataFrame(result_rows)
        if not self.results_.empty:
            if metric_notes:
                self.results_["Notes"] = self.results_["Model"].map(metric_notes).fillna("")
            self.results_["Error"] = self.results_.get("Error", "").fillna("")
            self.results_["Status"] = self.results_.get("Status", "success").fillna("success")
            self.results_ = self._sort_results(self.results_, y_test_sr)
        else:
            self.results_ = pd.DataFrame(columns=["Model", "Status", "Error"])

        self.failed_models_ = pd.DataFrame(failure_rows, columns=["Model", "Status", "Error"])
        self.skipped_models_ = pd.DataFrame(
            skipped_rows, columns=["Model", "Status", "Reason", "Dependency"]
        )
        self.predictions_ = (
            pd.DataFrame(prediction_columns, index=X_test_df.index)
            if prediction_columns
            else pd.DataFrame(index=X_test_df.index)
        )

        if self.tune and not self.results_.empty and not self._timed_out(start_time):
            self._run_tuning(X_train_df, X_test_df, y_train_sr, y_test_sr, start_time)
        else:
            self.tuned_results_ = pd.DataFrame()

        self._select_best_model(y_test_sr)
        self.run_summary_ = self._build_run_summary(X_train_df, X_test_df, y_test_sr, start_time)
        self._maybe_save_outputs()
        return self.results_, self.predictions_

    def _resolve_fit_inputs(
        self,
        X: Any,
        X_test_or_y: Any,
        y_train: Any | None,
        y_test: Any | None,
        *,
        test_size: float,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if y_train is None and y_test is None:
            X_df = ensure_dataframe(X)
            y_sr = ensure_series(X_test_or_y, name="target")
            if len(X_df) != len(y_sr):
                raise DataValidationError("X and y must contain the same number of rows.")
            stratify = y_sr if self.task == "classification" else None
            try:
                X_train, X_test, y_train_split, y_test_split = train_test_split(
                    X_df,
                    y_sr,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify,
                )
            except ValueError:
                if self.task == "classification":
                    warnings.warn(
                        "Stratified split was not possible for the provided labels; "
                        "falling back to an unstratified split.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                X_train, X_test, y_train_split, y_test_split = train_test_split(
                    X_df,
                    y_sr,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=None,
                )
            return (
                ensure_dataframe(X_train),
                ensure_dataframe(X_test),
                ensure_series(y_train_split),
                ensure_series(y_test_split),
            )

        if y_train is None or y_test is None:
            raise DataValidationError(
                "Use either fit(X, y, ...) or fit(X_train, X_test, y_train, y_test)."
            )

        X_train_df = ensure_dataframe(X)
        X_test_df = ensure_dataframe(X_test_or_y)
        y_train_sr = ensure_series(y_train, name="target")
        y_test_sr = ensure_series(y_test, name="target")
        if len(X_train_df) != len(y_train_sr):
            raise DataValidationError("X_train and y_train must contain the same number of rows.")
        if len(X_test_df) != len(y_test_sr):
            raise DataValidationError("X_test and y_test must contain the same number of rows.")
        return X_train_df, X_test_df, y_train_sr, y_test_sr

    def _select_specs(self) -> list[ModelSpec]:
        registry = get_model_registry(self.task)
        specs = [spec for spec in registry if spec.include_by_default]

        if self.models != "all":
            requested = {name.lower(): name for name in self.models}
            specs = [spec for spec in registry if spec.name.lower() in requested]
            missing = [
                name
                for name in requested.values()
                if name.lower() not in {spec.name.lower() for spec in registry}
            ]
            if missing:
                raise DataValidationError(f"Unknown model names: {', '.join(sorted(missing))}")

        if self.include_models:
            include = {name.lower() for name in self.include_models}
            specs = [spec for spec in registry if spec.name.lower() in include]
            missing = include.difference({spec.name.lower() for spec in registry})
            if missing:
                missing_display = ", ".join(sorted(missing))
                raise DataValidationError(
                    f"Unknown model names in include_models: {missing_display}"
                )

        if self.exclude_models:
            exclude = {name.lower() for name in self.exclude_models}
            specs = [spec for spec in specs if spec.name.lower() not in exclude]

        if self.max_models is not None:
            specs = specs[: self.max_models]

        if not specs:
            raise DataValidationError(
                "No models are selected. Check include_models, exclude_models, or max_models."
            )
        return specs

    def _fit_single_model(
        self,
        *,
        spec: ModelSpec,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> tuple[dict[str, Any], Any, np.ndarray, str]:
        estimator = make_estimator(
            spec, random_state=self.random_state, n_jobs=self.n_jobs, use_gpu=self.use_gpu
        )
        model = build_model_pipeline(
            estimator=estimator,
            X_train=X_train,
            preprocess=self.preprocess,
            needs_scaling=spec.needs_scaling,
            categorical_encoder=self.categorical_encoder,
            scale_numeric=self.scale_numeric,
        )

        with warnings.catch_warnings():
            if self.ignore_warnings:
                warnings.simplefilter("ignore")
            fit_start = perf_counter()
            model.fit(X_train, y_train)
            train_time = perf_counter() - fit_start
            predict_start = perf_counter()
            preds = model.predict(X_test)
            predict_time = perf_counter() - predict_start

        total_time = train_time + predict_time
        note = ""

        if self.task == "classification":
            proba = None
            decision = None
            note_parts: list[str] = []
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)
                except Exception as exc:
                    note_parts.append(f"predict_proba unavailable: {format_exception(exc)}")
            if hasattr(model, "decision_function"):
                try:
                    decision = model.decision_function(X_test)
                except Exception as exc:
                    note_parts.append(f"decision_function unavailable: {format_exception(exc)}")
            final_estimator = self._get_final_estimator(model)
            metric_values, metric_note = compute_classification_metrics(
                y_true=y_test,
                y_pred=preds,
                y_proba=proba,
                y_score=decision,
                custom_metrics=self.custom_metrics,
                model_classes=getattr(final_estimator, "classes_", None),
            )
            if metric_note:
                note_parts.append(metric_note)
            note = "; ".join(part for part in note_parts if part)
        else:
            feature_count = get_effective_feature_count(model, X_test)
            metric_values, note = compute_regression_metrics(
                y_true=y_test,
                y_pred=preds,
                n_features=feature_count,
                custom_metrics=self.custom_metrics,
            )

        row = {
            "Model": spec.name,
            **metric_values,
            "Train Time": train_time,
            "Predict Time": predict_time,
            "Total Time": total_time,
            "Status": "success",
            "Error": "",
        }
        return row, model, preds, note

    def _run_tuning(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        started_at: float,
    ) -> None:
        top_names = self.results_.head(self.tune_top_n)["Model"].tolist()
        specs = [spec for spec in self._select_specs() if spec.name in top_names]
        config = TuningConfig(
            task=self.task,
            sort_by=self._resolved_sort_metric(y_test),
            tuner=self.tuner,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            preprocess=self.preprocess,
            categorical_encoder=self.categorical_encoder,
            scale_numeric=self.scale_numeric,
            return_train_score=self.return_train_score,
            timeout=(
                None
                if self.timeout is None
                else max(self.timeout - (perf_counter() - started_at), 0.0)
            ),
            custom_metrics=self.custom_metrics,
            use_gpu=self.use_gpu,
            ignore_warnings=self.ignore_warnings,
            verbose=self.verbose,
        )
        self.tuned_results_, self.tuned_models_ = tune_selected_models(
            specs=specs,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            config=config,
        )

    def _select_best_model(self, y_test: pd.Series) -> None:
        if not self.results_.empty:
            top_name = self.results_.iloc[0]["Model"]
            self.best_model_ = self.fitted_models_.get(top_name)
            self.best_model_name_ = top_name
            self.best_model_source_ = "comparison"

        if self.tuned_results_.empty:
            return

        tuned_success = self.tuned_results_[self.tuned_results_["Status"] == "success"]
        if tuned_success.empty:
            return

        metric = self._resolved_sort_metric(y_test)
        if metric not in tuned_success.columns:
            return

        tuned_sorted = tuned_success.sort_values(
            by=metric, ascending=metric_sort_ascending(metric), na_position="last"
        )
        tuned_best_name = tuned_sorted.iloc[0]["Model"]
        tuned_best_model = self.tuned_models_.get(tuned_best_name)
        if tuned_best_model is None:
            return

        self.best_model_ = tuned_best_model
        self.best_model_name_ = tuned_best_name
        self.best_model_source_ = "tuned"

    def _sort_results(self, results: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        metric = self._resolved_sort_metric(y_test)
        if metric not in results.columns:
            return results.sort_values(by="Model", ascending=True).reset_index(drop=True)
        return results.sort_values(
            by=metric,
            ascending=metric_sort_ascending(metric),
            na_position="last",
        ).reset_index(drop=True)

    def _resolved_sort_metric(self, y_values: Any) -> str:
        if self.sort_by:
            return canonicalize_metric_name(self.task, self.sort_by)
        if self.task == "regression":
            return "R2"
        y_series = ensure_series(y_values) if not isinstance(y_values, pd.Series) else y_values
        unique_count = y_series.nunique(dropna=False)
        if unique_count <= 2:
            if (
                not self.results_.empty
                and "ROC AUC" in self.results_.columns
                and self.results_["ROC AUC"].notna().any()
            ):
                return "ROC AUC"
            return "F1 Weighted"
        return "F1 Weighted"

    def _maybe_save_outputs(self) -> None:
        if self.output_dir is None and not self.save_best:
            return

        output_dir = Path(self.output_dir or "all_predict_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        save_results(self.results_, output_dir)
        if not self.predictions_.empty:
            save_predictions(self.predictions_, output_dir)
        if not self.failed_models_.empty:
            save_dataframe(self.failed_models_, output_dir / "failed_models.csv")
        if not self.skipped_models_.empty:
            save_dataframe(self.skipped_models_, output_dir / "skipped_models.csv")
        if not self.tuned_results_.empty:
            save_dataframe(self.tuned_results_, output_dir / "tuned_results.csv")

        if self.save_best and self.best_model_ is not None:
            self.best_model_path_ = output_dir / "best_model.joblib"
            save_model(self.best_model_, self.best_model_path_)
            self.run_summary_["best_model_path"] = str(self.best_model_path_)

        save_run_summary(self.run_summary_, output_dir)

    def _build_run_summary(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        started_at: float,
    ) -> dict[str, Any]:
        summary = {
            "task": self.task,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_count": int(X_train.shape[1]),
            "successful_models": int(len(self.results_)),
            "failed_models": int(len(self.failed_models_)),
            "skipped_models": int(len(self.skipped_models_)),
            "tuned_models": int(
                (self.tuned_results_["Status"] == "success").sum()
                if not self.tuned_results_.empty
                else 0
            ),
            "sort_by": self._resolved_sort_metric(y_test),
            "best_model_name": self.best_model_name_,
            "best_model_source": self.best_model_source_,
            "predictions_requested": self.predictions,
            "tuning_enabled": self.tune,
            "tuner": self.tuner,
            "cv": self.cv,
            "timeout": self.timeout,
            "random_state": self.random_state,
            "runtime_seconds": round(perf_counter() - started_at, 6),
        }
        if self.best_model_path_ is not None:
            summary["best_model_path"] = str(self.best_model_path_)
        return summary

    def _timed_out(self, started_at: float) -> bool:
        return self.timeout is not None and (perf_counter() - started_at) >= self.timeout

    @staticmethod
    def _get_final_estimator(model: Any) -> Any:
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            return model.named_steps["model"]
        return model
