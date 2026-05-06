"""Backward-compatible registry exports."""

from __future__ import annotations

from .registry import (
    CLASSIFICATION_REGISTRY,
    REGRESSION_REGISTRY,
    get_model_registry,
    is_dependency_available,
    make_estimator,
)

CLASSIFIER_PARAM_GRIDS = {spec.name: dict(spec.param_grid) for spec in CLASSIFICATION_REGISTRY}
REGRESSOR_PARAM_GRIDS = {spec.name: dict(spec.param_grid) for spec in REGRESSION_REGISTRY}

CLASSIFIERS = [
    make_estimator(spec, random_state=42, n_jobs=-1, use_gpu=False)
    for spec in CLASSIFICATION_REGISTRY
    if is_dependency_available(spec.optional_dependency)
]
REGRESSORS = [
    make_estimator(spec, random_state=42, n_jobs=-1, use_gpu=False)
    for spec in REGRESSION_REGISTRY
    if is_dependency_available(spec.optional_dependency)
]

__all__ = [
    "CLASSIFIERS",
    "REGRESSORS",
    "CLASSIFIER_PARAM_GRIDS",
    "REGRESSOR_PARAM_GRIDS",
    "CLASSIFICATION_REGISTRY",
    "REGRESSION_REGISTRY",
    "get_model_registry",
]
