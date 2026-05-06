# Quickstart

`all_predict` compares many tabular classifiers and regressors with a small amount of code.

## Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from all_predict import AllClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

clf = AllClassifier(predictions=True, sort_by="roc_auc")
results, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(results.head())
```

## Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from all_predict import AllRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = AllRegressor(predictions=True, sort_by="r2")
results, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(results.head())
```
