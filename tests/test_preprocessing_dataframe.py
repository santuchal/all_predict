import pandas as pd

from all_predict import AllClassifier, AllRegressor


def _build_frame():
    frame = pd.DataFrame(
        {
            "num_a": [1.0, 2.5, None, 4.2, 5.1, 6.3, 7.4, None, 9.0, 10.2, 11.0, 12.3],
            "num_b": [3, None, 5, 6, 7, 8, 9, 10, None, 12, 13, 14],
            "city": [
                "Kolkata",
                "Delhi",
                "Mumbai",
                None,
                "Delhi",
                "Kolkata",
                "Mumbai",
                "Delhi",
                "Kolkata",
                None,
                "Delhi",
                "Mumbai",
            ],
            "segment": ["a", "a", "b", "b", "c", "c", "a", "b", "c", "a", None, "b"],
            "active": pd.Series(
                [True, False, True, None, False, True, False, True, None, False, True, False],
                dtype="boolean",
            ),
        }
    )
    return frame


def test_preprocessing_dataframe_classifier():
    frame = _build_frame()
    target = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], name="target")

    clf = AllClassifier(
        verbose=False,
        predictions=True,
        include_models=["LogisticRegression", "RandomForestClassifier", "DummyClassifier"],
    )
    results, predictions = clf.fit(frame, target, test_size=0.25, random_state=42)

    assert not results.empty
    assert len(predictions) == 3


def test_preprocessing_dataframe_regressor():
    frame = _build_frame()
    target = pd.Series(
        [10.0, 11.2, 12.3, 14.5, 15.0, 16.8, 18.1, 19.4, 21.0, 22.2, 24.1, 25.0], name="target"
    )

    reg = AllRegressor(
        verbose=False,
        predictions=True,
        include_models=["LinearRegression", "RandomForestRegressor", "DummyRegressor"],
    )
    results, predictions = reg.fit(frame, target, test_size=0.25, random_state=42)

    assert not results.empty
    assert len(predictions) == 3
