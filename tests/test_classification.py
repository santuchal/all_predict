# tests/test_classification.py
"""Unit tests for LazyClassifierPlus."""
import pytest
import pandas as pd
from sklearn.datasets import make_classification
from all_predict.classification import LazyClassifierPlus

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    return X, y

def test_classifier_runs(classification_data):
    X, y = classification_data
    clf = LazyClassifierPlus(verbose=False)
    df_results, tuned_results = clf.fit(X, y)
    assert isinstance(df_results, pd.DataFrame)
    assert not df_results.empty
    assert isinstance(tuned_results, pd.DataFrame)