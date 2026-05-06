"""Minimal classification example for all_predict 0.3.0."""

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from all_predict import AllClassifier


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = AllClassifier(
        verbose=True,
        random_state=42,
        n_jobs=-1,
        sort_by="roc_auc",
        predictions=True,
        include_models=[
            "LogisticRegression",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "GradientBoostingClassifier",
            "DummyClassifier",
        ],
        output_dir=Path("runs") / "breast_cancer_example",
        save_best=True,
    )
    results, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(results.head())
    print(predictions.head())


if __name__ == "__main__":
    main()
