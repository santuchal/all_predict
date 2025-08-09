# classification.py
"""LazyClassifierPlus implementation for all_predict.
Evaluates multiple classification models, ranks them, and tunes top performers.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .base import BasePredictor
from .model_registry import CLASSIFIERS, CLASSIFIER_PARAM_GRIDS
from .tuner import tune_top_models

class LazyClassifierPlus(BasePredictor):
    def fit(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        results = []
        for model in CLASSIFIERS:
            model = self._set_random_state(model)
            self.logger.debug(f"Training {model.__class__.__name__}")
            try:
                fitted_model, train_time = self._time_function(model.fit, X_train, y_train)
                preds, pred_time = self._time_function(fitted_model.predict, X_test)
                proba = None
                if hasattr(fitted_model, "predict_proba"):
                    try:
                        proba = fitted_model.predict_proba(X_test)[:, 1]
                    except Exception:
                        proba = None
                results.append({
                    "Model": model.__class__.__name__,
                    "Accuracy": accuracy_score(y_test, preds),
                    "F1": f1_score(y_test, preds, average='weighted'),
                    "AUC": roc_auc_score(y_test, proba) if proba is not None else None,
                    "Train Time": train_time,
                    "Predict Time": pred_time
                })
            except Exception as e:
                self.logger.warning(f"{model.__class__.__name__} failed: {e}")

        df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

        top3_models = df_results.head(3)["Model"].tolist()
        tuned_results = tune_top_models(top3_models, CLASSIFIER_PARAM_GRIDS, X_train, y_train, X_test, y_test, task="classification")

        return df_results, tuned_results
