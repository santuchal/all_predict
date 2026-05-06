from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from all_predict import AllClassifier, load_model, save_model


def test_persistence_round_trip(tmp_path):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = AllClassifier(verbose=False, include_models=["LogisticRegression", "DummyClassifier"])
    clf.fit(X_train, X_test, y_train, y_test)

    path = tmp_path / "best_model.joblib"
    save_model(clf.best_model_, path)
    loaded = load_model(path)

    assert loaded.predict(X_test).shape[0] == len(X_test)
