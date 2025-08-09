# utils.py
"""Utility functions for all_predict.
Includes model persistence, metrics helpers, and data checks.
"""

import os
import joblib
import logging
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    model = joblib.load(path)
    logging.info(f"Model loaded from {path}")
    return model

def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

def classification_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average='weighted')
    }
    if y_proba is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["AUC"] = None
    return metrics

def check_missing_values(X):
    if hasattr(X, 'isnull') and X.isnull().sum().sum() > 0:
        logging.warning("Input data contains missing values. Consider preprocessing.")
