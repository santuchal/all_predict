import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from all_predict import AllClassifier


def test_classification_basic():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = AllClassifier(
        verbose=False,
        random_state=42,
        predictions=True,
        include_models=[
            "LogisticRegression",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "DummyClassifier",
        ],
    )
    results, predictions = clf.fit(X_train, X_test, y_train, y_test)

    required_columns = {
        "Model",
        "Accuracy",
        "Balanced Accuracy",
        "ROC AUC",
        "Average Precision",
        "F1",
        "F1 Macro",
        "F1 Weighted",
        "Precision",
        "Recall",
        "MCC",
        "Log Loss",
        "Train Time",
        "Predict Time",
        "Total Time",
        "Status",
        "Error",
    }
    assert not results.empty
    assert required_columns.issubset(results.columns)
    assert len(predictions) == len(y_test)

    non_dummy = results[results["Model"] != "DummyClassifier"].iloc[0]
    dummy = results[results["Model"] == "DummyClassifier"].iloc[0]
    assert non_dummy["Accuracy"] >= 0.90
    if not np.isnan(non_dummy["ROC AUC"]):
        assert non_dummy["ROC AUC"] >= 0.90
    assert dummy["Accuracy"] < non_dummy["Accuracy"]
