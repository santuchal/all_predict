import numpy as np
import pandas as pd

from all_predict.metrics import adjusted_r2_score, compute_classification_metrics, safe_mape


def test_adjusted_r2_edge_cases():
    assert np.isnan(adjusted_r2_score(0.5, 5, 5))
    assert adjusted_r2_score(0.9, 100, 5) > 0.0


def test_safe_mape_with_zero_targets():
    value = safe_mape([0.0, 10.0, 20.0], [1.0, 12.0, 18.0])
    assert np.isfinite(value)
    assert np.isnan(safe_mape([0.0, 0.0], [1.0, 2.0]))


def test_classification_metrics_binary():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([[0.8, 0.2], [0.2, 0.8], [0.75, 0.25], [0.1, 0.9]])
    metrics, note = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba)

    assert metrics["Accuracy"] == 1.0
    assert metrics["ROC AUC"] >= 0.99
    assert note == ""


def test_classification_metrics_multiclass():
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
            [0.1, 0.2, 0.7],
        ]
    )
    metrics, note = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba)

    assert metrics["F1 Weighted"] > 0.7
    assert np.isfinite(metrics["ROC AUC"])
    assert "require predict_proba" not in note
