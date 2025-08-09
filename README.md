# README.md
# all_predict

**Author:** Santu Chall  
**Email:** santuchal@gmail.com  

## Overview
`all_predict` is an advanced, production-ready alternative to [LazyPredict](https://github.com/shankarpandala/lazypredict), providing:
- **30+ regression models** and **30+ classification models** from scikit-learn, XGBoost, LightGBM, CatBoost, and more.
- Automatic evaluation and ranking of all models.
- **Top-3 model hyperparameter tuning** using `GridSearchCV` with **robust, extended parameter grids** for deeper optimization.
- Performance metrics, training/prediction time logging.
- Model saving/loading utilities.
- Comparison visualizations.

## Installation
```bash
pip install all_predict
```

## Quick Start
### Regression Example
```python
from sklearn.datasets import make_regression
from all_predict.regression import LazyRegressorPlus

X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)

reg = LazyRegressorPlus(verbose=True)
results, tuned = reg.fit(X, y)
print(results.head())
print(tuned)
```

### Classification Example
```python
from sklearn.datasets import make_classification
from all_predict.classification import LazyClassifierPlus

X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

clf = LazyClassifierPlus(verbose=True)
results, tuned = clf.fit(X, y)
print(results.head())
print(tuned)
```

## Robust Parameter Grids
The GridSearchCV now uses **expanded hyperparameter grids** tailored for each model type:
- **Tree-based models**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, learning rate (if applicable)
- **Linear models**: `alpha`, `l1_ratio`, regularization type, solver variations
- **Boosting models**: `n_estimators`, `learning_rate`, `max_depth`, `colsample_bytree`, `subsample`, regularization terms
- **SVM**: `C`, `kernel`, `gamma`, `degree`
- **KNN**: `n_neighbors`, `weights`, `metric`

These grids allow the tuner to explore broader, more robust parameter spaces for significantly better model performance.

## Features
- **Extended Model List:** More models than LazyPredict.
- **Top-3 GridSearch Tuning:** Automated hyperparameter optimization with deep parameter grids.
- **Persistence:** Save and load best models.
- **Visualization:** Compare performance and timings.

## Output Example
**Regression**
| Model                  | R2   | RMSE  | MAE  | Train Time | Predict Time |
|------------------------|------|-------|------|------------|--------------|
| RandomForestRegressor  | 0.95 | 2.10  | 1.50 | 0.12       | 0.02         |

**Tuned Models**
| Model                  | Best Params                                     | Best CV Score | Test Score |
|------------------------|-------------------------------------------------|---------------|------------|
| RandomForestRegressor  | {"n_estimators":200, "max_depth":10}            | 0.96          | 0.95       |

## Comparison with LazyPredict
| Feature                        | LazyPredict | all_predict |
|--------------------------------|-------------|-------------|
| Model Count                    | ~20         | 60+         |
| Auto GridSearch Tuning         | ❌           | ✅           |
| Robust Parameter Grids         | ❌           | ✅           |
| Model Saving/Loading           | ❌           | ✅           |
| Visualization                  | ❌           | ✅           |
| Parallel Processing            | Limited     | Full        |

## License
MIT License.
