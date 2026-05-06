import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from all_predict import AllRegressor


def test_regression_real_dataset():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = AllRegressor(
        verbose=False,
        random_state=42,
        include_models=["LinearRegression", "RandomForestRegressor", "DummyRegressor"],
    )
    results, _ = reg.fit(X_train, X_test, y_train, y_test)

    assert not results.empty
    for column in ["R2", "RMSE", "MAE"]:
        assert pd.api.types.is_numeric_dtype(results[column])
