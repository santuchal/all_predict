# model_registry.py
"""Model registry for all_predict.
This module contains lists of regression and classification models with sensible parameter grids.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier

try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = None
    XGBClassifier = None

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    LGBMRegressor = None
    LGBMClassifier = None

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:
    CatBoostRegressor = None
    CatBoostClassifier = None

# Parameter grids for GridSearchCV
REGRESSOR_PARAM_GRIDS = {
    "RandomForestRegressor": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "GradientBoostingRegressor": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "XGBRegressor": {"n_estimators": [100, 200], "max_depth": [3, 5]},
    "LGBMRegressor": {"n_estimators": [100, 200], "num_leaves": [31, 50]},
    "CatBoostRegressor": {"iterations": [200, 500], "depth": [6, 8]},
}

CLASSIFIER_PARAM_GRIDS = {
    "RandomForestClassifier": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "GradientBoostingClassifier": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "XGBClassifier": {"n_estimators": [100, 200], "max_depth": [3, 5]},
    "LGBMClassifier": {"n_estimators": [100, 200], "num_leaves": [31, 50]},
    "CatBoostClassifier": {"iterations": [200, 500], "depth": [6, 8]},
}

# Model lists (instances will be created in runtime with default params)
REGRESSORS = [
    LinearRegression(), Ridge(), Lasso(), ElasticNet(), SGDRegressor(),
    RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor(), ExtraTreesRegressor(),
    SVR(), DecisionTreeRegressor(), KNeighborsRegressor(),
    MLPRegressor(), GaussianProcessRegressor(),
]

CLASSIFIERS = [
    LogisticRegression(max_iter=1000), RandomForestClassifier(), GradientBoostingClassifier(),
    AdaBoostClassifier(), ExtraTreesClassifier(), SVC(probability=True),
    LinearSVC(), DecisionTreeClassifier(), KNeighborsClassifier(),
    GaussianNB(), MLPClassifier(max_iter=1000), GaussianProcessClassifier(),
]
