# regression.py
"""LazyRegressorPlus implementation for all_predict.
Evaluates multiple regression models, ranks them, and tunes top performers.
"""

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from .base import BasePredictor
from .model_registry import REGRESSORS, REGRESSOR_PARAM_GRIDS
from .tuner import tune_top_models

class LazyRegressorPlus(BasePredictor):

    def _filter_param_grid(self, model, param_grid):
        """Remove params not supported by the model."""
        valid_keys = set(model.get_params(deep=True).keys())
        return {k: v for k, v in param_grid.items() if k in valid_keys}

    def fit(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        results = []
        for model in REGRESSORS:
            model = self._set_random_state(model)
            self.logger.debug(f"Training {model.__class__.__name__}")
            try:
                fitted_model, train_time = self._time_function(model.fit, X_train, y_train)
                preds, pred_time = self._time_function(fitted_model.predict, X_test)
                results.append({
                    "Model": model.__class__.__name__,
                    "R2": r2_score(y_test, preds),
                    "RMSE": mean_squared_error(y_test, preds)**0.5,
                    "MAE": mean_absolute_error(y_test, preds),
                    "Train Time": train_time,
                    "Predict Time": pred_time
                })
            except Exception as e:
                self.logger.warning(f"{model.__class__.__name__} failed: {e}")
        

        if not results:
            return pd.DataFrame(), pd.DataFrame()

        df_results = pd.DataFrame(results).sort_values(by="R2", ascending=False)

        top3_models = df_results.head(3)["Model"].tolist()
        tuned_results = tune_top_models(top3_models, REGRESSOR_PARAM_GRIDS, X_train, y_train, X_test, y_test, task="regression")

        return df_results, tuned_results



