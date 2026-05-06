"""Regression interfaces for all_predict."""

from .base import BasePredictor


class AllRegressor(BasePredictor):
    """Compare many regression models on a tabular dataset."""

    task = "regression"


class AutoRegressor(AllRegressor):
    """Alias for AllRegressor."""


class LazyRegressorPlus(AllRegressor):
    """Backward-compatible alias for the previous public regressor API."""
