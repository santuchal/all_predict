# __init__.py for all_predict package

"""All Predict - Enhanced LazyPredict-style library with extended model support and hyperparameter tuning."""

from .regression import LazyRegressorPlus
from .classification import LazyClassifierPlus
from .model_registry import (
    REGRESSORS,
    CLASSIFIERS,
    REGRESSOR_PARAM_GRIDS,
    CLASSIFIER_PARAM_GRIDS,
)

__all__ = [
    "LazyRegressorPlus",
    "LazyClassifierPlus",
    "REGRESSORS",
    "CLASSIFIERS",
    "REGRESSOR_PARAM_GRIDS",
    "CLASSIFIER_PARAM_GRIDS",
]
