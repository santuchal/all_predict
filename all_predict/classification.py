"""Classification interfaces for all_predict."""

from .base import BasePredictor


class AllClassifier(BasePredictor):
    """Compare many classification models on a tabular dataset."""

    task = "classification"


class AutoClassifier(AllClassifier):
    """Alias for AllClassifier."""


class LazyClassifierPlus(AllClassifier):
    """Backward-compatible alias for the previous public classifier API."""
