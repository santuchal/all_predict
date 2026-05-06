import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from all_predict import AllClassifier


def test_classification_multiclass():
    X, y = load_wine(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    clf = AllClassifier(
        verbose=False,
        random_state=42,
        include_models=[
            "LogisticRegression",
            "RandomForestClassifier",
            "KNeighborsClassifier",
            "DummyClassifier",
        ],
    )
    results, _ = clf.fit(X_train, X_test, y_train, y_test)

    non_dummy = results[results["Model"] != "DummyClassifier"].iloc[0]
    assert non_dummy["F1 Weighted"] >= 0.85
    assert "ROC AUC" in results.columns
    assert results["ROC AUC"].dtype.kind in {"f", "i"}
    assert not np.isnan(non_dummy["F1 Weighted"])
