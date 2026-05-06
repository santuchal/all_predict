from all_predict import (
    AllClassifier,
    AllRegressor,
    LazyClassifierPlus,
    LazyRegressorPlus,
    load_model,
    save_model,
)
from all_predict.classification import LazyClassifierPlus as LazyClassifierPlusModule
from all_predict.regression import LazyRegressorPlus as LazyRegressorPlusModule


def test_public_imports():
    assert AllClassifier is not None
    assert AllRegressor is not None
    assert LazyClassifierPlus is LazyClassifierPlusModule
    assert LazyRegressorPlus is LazyRegressorPlusModule
    assert save_model is not None
    assert load_model is not None
