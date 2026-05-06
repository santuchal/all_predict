"""Minimal regression example for all_predict 0.3.0."""

from pathlib import Path

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from all_predict import AllRegressor


def main() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    reg = AllRegressor(
        verbose=True,
        random_state=42,
        n_jobs=-1,
        sort_by="r2",
        predictions=True,
        include_models=[
            "LinearRegression",
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "GradientBoostingRegressor",
            "DummyRegressor",
        ],
        output_dir=Path("runs") / "diabetes_example",
        save_best=True,
    )
    results, predictions = reg.fit(X_train, X_test, y_train, y_test)
    print(results.head())
    print(predictions.head())


if __name__ == "__main__":
    main()
