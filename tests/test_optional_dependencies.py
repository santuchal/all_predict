from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from all_predict import AllClassifier


def test_optional_dependencies_are_skipped(monkeypatch):
    monkeypatch.setattr(
        "all_predict.base.is_dependency_available",
        lambda dependency: False if dependency in {"xgboost", "lightgbm", "catboost"} else True,
    )

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
        include_models=["LogisticRegression", "XGBClassifier"],
    )
    results, _ = clf.fit(X_train, X_test, y_train, y_test)

    assert "LogisticRegression" in results["Model"].tolist()
    assert "XGBClassifier" in clf.skipped_models_["Model"].tolist()
