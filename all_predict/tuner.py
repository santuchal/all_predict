# tuner.py
"""Hyperparameter tuning module for all_predict.
Uses GridSearchCV to tune top-performing models for regression and classification.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from .model_registry import REGRESSORS, CLASSIFIERS, REGRESSOR_PARAM_GRIDS, CLASSIFIER_PARAM_GRIDS

def tune_top_models(top_model_names, param_grids, X_train, y_train, X_test, y_test, task="regression"):
    results = []
    model_pool = REGRESSORS if task == "regression" else CLASSIFIERS

    for model_name in top_model_names:
        model_cls = next((m for m in model_pool if m.__class__.__name__ == model_name), None)
        if not model_cls:
            continue

        params = param_grids.get(model_name, {})
        if not params:
            continue

        try:
            grid = GridSearchCV(model_cls, params, n_jobs=-1, cv=5, scoring=_select_scoring(task))
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            score = grid.score(X_test, y_test)

            results.append({
                "Model": model_name,
                "Best Params": grid.best_params_,
                "Best CV Score": grid.best_score_,
                "Test Score": score
            })
        except Exception as e:
            results.append({"Model": model_name, "Error": str(e)})

    return pd.DataFrame(results)

def _select_scoring(task):
    if task == "regression":
        return "r2"
    elif task == "classification":
        return "accuracy"
    return None
