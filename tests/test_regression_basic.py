from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from all_predict import AllRegressor


def test_regression_basic():
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        noise=5.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = AllRegressor(
        verbose=False,
        random_state=42,
        include_models=[
            "LinearRegression",
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "DummyRegressor",
        ],
    )
    results, _ = reg.fit(X_train, X_test, y_train, y_test)

    required_columns = {
        "Model",
        "R2",
        "Adjusted R2",
        "RMSE",
        "MAE",
        "Median AE",
        "MAPE",
        "Explained Variance",
        "Max Error",
        "Train Time",
        "Predict Time",
        "Total Time",
        "Status",
        "Error",
    }
    assert not results.empty
    assert required_columns.issubset(results.columns)

    best_real = results[results["Model"] != "DummyRegressor"].iloc[0]
    dummy = results[results["Model"] == "DummyRegressor"].iloc[0]
    assert best_real["R2"] >= 0.85
    assert best_real["MAE"] < dummy["MAE"]
