# model_registry.py

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression, SGDRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

# Parameter grids for GridSearchCV
CLASSIFIER_PARAM_GRIDS = {
    cls.__name__: params for cls, params in {
        RandomForestClassifier: {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 5, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        GradientBoostingClassifier: {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [3, 5, 10],
            'subsample': [0.6, 0.8, 1.0],
            'max_features': ['sqrt', 'log2', None]
        },
        AdaBoostClassifier: {
            'n_estimators': [50, 100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 1.0]
        },
        ExtraTreesClassifier: {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2', None]
        },
        LogisticRegression: {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': [100, 500, 1000]
        },
        SVC: {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4, 5]
        },
        KNeighborsClassifier: {
            'n_neighbors': [3, 5, 7, 9, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        DecisionTreeClassifier: {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        MLPClassifier: {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        },
        XGBClassifier: {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [3, 5, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        } if XGBClassifier else {},
        LGBMClassifier: {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [-1, 5, 10],
            'num_leaves': [31, 50, 100],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        } if LGBMClassifier else {},
        CatBoostClassifier: {
            'iterations': [200, 500, 1000],
            'depth': [4, 6, 10],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5]
        } if CatBoostClassifier else {}
    }.items() if params
}

REGRESSOR_PARAM_GRIDS = {
    cls.__name__: params for cls, params in {
        RandomForestRegressor: {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 5, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        GradientBoostingRegressor: {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [3, 5, 10],
            'subsample': [0.6, 0.8, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
    }.items()
}

CLASSIFIERS = [
    LogisticRegression(max_iter=1000), RandomForestClassifier(), GradientBoostingClassifier(),
    AdaBoostClassifier(), ExtraTreesClassifier(), SVC(probability=True),
    LinearSVC(), DecisionTreeClassifier(), KNeighborsClassifier(),
    GaussianNB(), MLPClassifier(max_iter=1000), GaussianProcessClassifier()
]

REGRESSORS = [
    Ridge(), Lasso(), ElasticNet(), SGDRegressor(), RandomForestRegressor(),
    GradientBoostingRegressor(), AdaBoostRegressor(), ExtraTreesRegressor(),
    SVR(), DecisionTreeRegressor(), KNeighborsRegressor(),
    MLPRegressor(), GaussianProcessRegressor()
]

__all__ = ["CLASSIFIERS", "REGRESSORS", "CLASSIFIER_PARAM_GRIDS", "REGRESSOR_PARAM_GRIDS"]
