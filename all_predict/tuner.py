"""Backward-compatible tuning wrapper."""

from __future__ import annotations

import pandas as pd

from .registry import get_model_spec
from .tuning import TuningConfig, tune_selected_models
from .utils import ensure_dataframe, ensure_series


def tune_top_models(
    top_model_names, param_grids, X_train, y_train, X_test, y_test, task="regression"
):
    """Compatibility wrapper around the new tuning module."""

    del param_grids
    specs = [get_model_spec(task, name) for name in top_model_names]
    config = TuningConfig(
        task=task,
        sort_by=None,
        tuner="grid",
        cv=5,
        random_state=42,
        n_jobs=-1,
        preprocess=True,
        categorical_encoder="onehot",
        scale_numeric="auto",
        return_train_score=False,
        timeout=None,
        custom_metrics=None,
        use_gpu=False,
        ignore_warnings=True,
        verbose=False,
    )
    tuned, _ = tune_selected_models(
        specs=specs,
        X_train=ensure_dataframe(X_train),
        X_test=ensure_dataframe(X_test),
        y_train=ensure_series(y_train),
        y_test=ensure_series(y_test),
        config=config,
    )
    return tuned if isinstance(tuned, pd.DataFrame) else pd.DataFrame(tuned)
