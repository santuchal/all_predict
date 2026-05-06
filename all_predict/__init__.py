"""Public package interface for all_predict."""

from .__version__ import __author__, __email__, __version__
from .classification import AllClassifier, AutoClassifier, LazyClassifierPlus
from .persistence import load_model, save_model
from .regression import AllRegressor, AutoRegressor, LazyRegressorPlus
from .reporting import save_predictions, save_results, save_run_summary

__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "AllClassifier",
    "AllRegressor",
    "AutoClassifier",
    "AutoRegressor",
    "LazyClassifierPlus",
    "LazyRegressorPlus",
    "load_model",
    "save_model",
    "save_predictions",
    "save_results",
    "save_run_summary",
]
