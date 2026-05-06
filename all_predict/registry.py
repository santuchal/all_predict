"""Model registry with metadata for supported estimators."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    Lasso,
    LassoCV,
    LassoLars,
    LinearRegression,
    LogisticRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

Factory = Callable[[int | None, int | None, bool], BaseEstimator]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a supported model entry."""

    name: str
    estimator_factory: Factory
    task: str
    needs_scaling: bool = False
    supports_proba: bool = False
    supports_decision_function: bool = False
    optional_dependency: str | None = None
    include_by_default: bool = True
    param_grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)


def is_dependency_available(dependency_name: str | None) -> bool:
    """Return True when an optional dependency can be imported."""

    if dependency_name is None:
        return True
    return importlib.util.find_spec(dependency_name) is not None


def _set_shared_params(
    estimator: BaseEstimator, random_state: int | None, n_jobs: int | None
) -> BaseEstimator:
    params = estimator.get_params(deep=True)
    updates: dict[str, Any] = {}
    if random_state is not None and "random_state" in params:
        updates["random_state"] = random_state
    if n_jobs is not None and "n_jobs" in params:
        updates["n_jobs"] = n_jobs
    if updates:
        estimator.set_params(**updates)
    return estimator


def _factory(constructor: Callable[[], BaseEstimator]) -> Factory:
    return lambda random_state, n_jobs, use_gpu: _set_shared_params(
        constructor(), random_state, n_jobs
    )


def _optional_xgb_classifier(
    random_state: int | None, n_jobs: int | None, use_gpu: bool
) -> BaseEstimator:
    from xgboost import XGBClassifier

    kwargs = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    if use_gpu:
        kwargs["tree_method"] = "hist"
        kwargs["device"] = "cuda"
    return XGBClassifier(**kwargs)


def _optional_xgb_regressor(
    random_state: int | None, n_jobs: int | None, use_gpu: bool
) -> BaseEstimator:
    from xgboost import XGBRegressor

    kwargs = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    if use_gpu:
        kwargs["tree_method"] = "hist"
        kwargs["device"] = "cuda"
    return XGBRegressor(**kwargs)


def _optional_lgbm_classifier(
    random_state: int | None, n_jobs: int | None, use_gpu: bool
) -> BaseEstimator:
    from lightgbm import LGBMClassifier

    kwargs = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }
    if use_gpu:
        kwargs["device_type"] = "gpu"
    return LGBMClassifier(**kwargs)


def _optional_lgbm_regressor(
    random_state: int | None, n_jobs: int | None, use_gpu: bool
) -> BaseEstimator:
    from lightgbm import LGBMRegressor

    kwargs = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }
    if use_gpu:
        kwargs["device_type"] = "gpu"
    return LGBMRegressor(**kwargs)


def _optional_catboost_classifier(
    random_state: int | None, n_jobs: int | None, use_gpu: bool
) -> BaseEstimator:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        random_state=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=n_jobs,
        task_type="GPU" if use_gpu else "CPU",
    )


def _optional_catboost_regressor(
    random_state: int | None, n_jobs: int | None, use_gpu: bool
) -> BaseEstimator:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        random_state=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=n_jobs,
        task_type="GPU" if use_gpu else "CPU",
    )


CLASSIFICATION_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        "LogisticRegression",
        _factory(lambda: LogisticRegression(max_iter=2000)),
        "classification",
        needs_scaling=True,
        supports_proba=True,
        supports_decision_function=True,
        param_grid={"C": [0.1, 1.0, 10.0]},
    ),
    ModelSpec(
        "RidgeClassifier",
        _factory(RidgeClassifier),
        "classification",
        needs_scaling=True,
        supports_decision_function=True,
        param_grid={"alpha": [0.1, 1.0, 10.0]},
    ),
    ModelSpec(
        "RidgeClassifierCV",
        _factory(lambda: RidgeClassifierCV(alphas=(0.1, 1.0, 10.0))),
        "classification",
        needs_scaling=True,
        supports_decision_function=True,
    ),
    ModelSpec(
        "SGDClassifier",
        _factory(lambda: SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-3)),
        "classification",
        needs_scaling=True,
        supports_proba=True,
        supports_decision_function=True,
        param_grid={"alpha": [0.0001, 0.001, 0.01]},
    ),
    ModelSpec(
        "PassiveAggressiveClassifier",
        _factory(lambda: PassiveAggressiveClassifier(max_iter=2000, tol=1e-3)),
        "classification",
        needs_scaling=True,
        supports_decision_function=True,
        param_grid={"C": [0.1, 1.0, 10.0]},
    ),
    ModelSpec(
        "KNeighborsClassifier",
        _factory(KNeighborsClassifier),
        "classification",
        needs_scaling=True,
        supports_proba=True,
        param_grid={"n_neighbors": [3, 5, 9], "weights": ["uniform", "distance"]},
    ),
    ModelSpec(
        "SVC",
        _factory(lambda: SVC(probability=True)),
        "classification",
        needs_scaling=True,
        supports_proba=True,
        supports_decision_function=True,
        param_grid={"C": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"]},
    ),
    ModelSpec(
        "LinearSVC",
        _factory(lambda: LinearSVC(dual="auto")),
        "classification",
        needs_scaling=True,
        supports_decision_function=True,
        param_grid={"C": [0.1, 1.0, 10.0]},
    ),
    ModelSpec(
        "NuSVC",
        _factory(lambda: NuSVC(probability=True)),
        "classification",
        needs_scaling=True,
        supports_proba=True,
        supports_decision_function=True,
        param_grid={"nu": [0.25, 0.5, 0.75], "gamma": ["scale", "auto"]},
    ),
    ModelSpec(
        "GaussianNB",
        _factory(GaussianNB),
        "classification",
        supports_proba=True,
        param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
    ModelSpec(
        "BernoulliNB",
        _factory(BernoulliNB),
        "classification",
        supports_proba=True,
        param_grid={"alpha": [0.1, 1.0, 10.0]},
    ),
    ModelSpec(
        "DecisionTreeClassifier",
        _factory(DecisionTreeClassifier),
        "classification",
        supports_proba=True,
        param_grid={"max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4]},
    ),
    ModelSpec(
        "ExtraTreeClassifier",
        _factory(ExtraTreeClassifier),
        "classification",
        supports_proba=True,
        param_grid={"max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4]},
    ),
    ModelSpec(
        "RandomForestClassifier",
        _factory(lambda: RandomForestClassifier(n_estimators=150)),
        "classification",
        supports_proba=True,
        param_grid={"max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 4]},
    ),
    ModelSpec(
        "ExtraTreesClassifier",
        _factory(lambda: ExtraTreesClassifier(n_estimators=150)),
        "classification",
        supports_proba=True,
        param_grid={"max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 4]},
    ),
    ModelSpec(
        "GradientBoostingClassifier",
        _factory(GradientBoostingClassifier),
        "classification",
        supports_proba=True,
        param_grid={"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    ),
    ModelSpec(
        "HistGradientBoostingClassifier",
        _factory(HistGradientBoostingClassifier),
        "classification",
        supports_proba=True,
        param_grid={"max_iter": [100, 200], "learning_rate": [0.05, 0.1]},
    ),
    ModelSpec(
        "AdaBoostClassifier",
        _factory(AdaBoostClassifier),
        "classification",
        supports_proba=True,
        param_grid={"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
    ),
    ModelSpec(
        "BaggingClassifier",
        _factory(lambda: BaggingClassifier(n_estimators=25)),
        "classification",
        supports_proba=True,
        param_grid={"n_estimators": [10, 25, 50]},
    ),
    ModelSpec(
        "LinearDiscriminantAnalysis",
        _factory(LinearDiscriminantAnalysis),
        "classification",
        supports_proba=True,
        supports_decision_function=True,
    ),
    ModelSpec(
        "QuadraticDiscriminantAnalysis",
        _factory(QuadraticDiscriminantAnalysis),
        "classification",
        supports_proba=True,
    ),
    ModelSpec(
        "MLPClassifier",
        _factory(
            lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True)
        ),
        "classification",
        needs_scaling=True,
        supports_proba=True,
        param_grid={"hidden_layer_sizes": [(64,), (100,), (100, 50)], "alpha": [0.0001, 0.001]},
    ),
    ModelSpec(
        "DummyClassifier",
        _factory(lambda: DummyClassifier(strategy="prior")),
        "classification",
        supports_proba=True,
    ),
    ModelSpec(
        "XGBClassifier",
        _optional_xgb_classifier,
        "classification",
        supports_proba=True,
        optional_dependency="xgboost",
        param_grid={"max_depth": [4, 6], "learning_rate": [0.03, 0.1]},
    ),
    ModelSpec(
        "LGBMClassifier",
        _optional_lgbm_classifier,
        "classification",
        supports_proba=True,
        optional_dependency="lightgbm",
        param_grid={"num_leaves": [31, 63], "learning_rate": [0.03, 0.1]},
    ),
    ModelSpec(
        "CatBoostClassifier",
        _optional_catboost_classifier,
        "classification",
        supports_proba=True,
        optional_dependency="catboost",
        param_grid={"depth": [4, 6, 8], "learning_rate": [0.03, 0.1]},
    ),
]

REGRESSION_REGISTRY: list[ModelSpec] = [
    ModelSpec("LinearRegression", _factory(LinearRegression), "regression"),
    ModelSpec(
        "Ridge",
        _factory(Ridge),
        "regression",
        needs_scaling=True,
        param_grid={"alpha": [0.1, 1.0, 10.0]},
    ),
    ModelSpec("RidgeCV", _factory(lambda: RidgeCV(alphas=(0.1, 1.0, 10.0))), "regression"),
    ModelSpec(
        "Lasso",
        _factory(Lasso),
        "regression",
        needs_scaling=True,
        param_grid={"alpha": [0.001, 0.01, 0.1, 1.0]},
    ),
    ModelSpec("LassoCV", _factory(lambda: LassoCV(cv=3)), "regression", needs_scaling=True),
    ModelSpec(
        "ElasticNet",
        _factory(ElasticNet),
        "regression",
        needs_scaling=True,
        param_grid={"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
    ),
    ModelSpec(
        "ElasticNetCV", _factory(lambda: ElasticNetCV(cv=3)), "regression", needs_scaling=True
    ),
    ModelSpec("BayesianRidge", _factory(BayesianRidge), "regression", needs_scaling=True),
    ModelSpec(
        "HuberRegressor",
        _factory(HuberRegressor),
        "regression",
        needs_scaling=True,
        param_grid={"epsilon": [1.2, 1.35, 1.5]},
    ),
    ModelSpec("Lars", _factory(Lars), "regression", needs_scaling=True),
    ModelSpec("LassoLars", _factory(LassoLars), "regression", needs_scaling=True),
    ModelSpec(
        "OrthogonalMatchingPursuit",
        _factory(OrthogonalMatchingPursuit),
        "regression",
        needs_scaling=True,
    ),
    ModelSpec(
        "SGDRegressor",
        _factory(lambda: SGDRegressor(max_iter=2000, tol=1e-3)),
        "regression",
        needs_scaling=True,
        param_grid={"alpha": [0.0001, 0.001, 0.01]},
    ),
    ModelSpec(
        "PassiveAggressiveRegressor",
        _factory(lambda: PassiveAggressiveRegressor(max_iter=2000, tol=1e-3)),
        "regression",
        needs_scaling=True,
        param_grid={"C": [0.1, 1.0, 10.0]},
    ),
    ModelSpec(
        "KNeighborsRegressor",
        _factory(KNeighborsRegressor),
        "regression",
        needs_scaling=True,
        param_grid={"n_neighbors": [3, 5, 9], "weights": ["uniform", "distance"]},
    ),
    ModelSpec(
        "SVR",
        _factory(SVR),
        "regression",
        needs_scaling=True,
        param_grid={"C": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"]},
    ),
    ModelSpec(
        "LinearSVR",
        _factory(lambda: LinearSVR(dual="auto")),
        "regression",
        needs_scaling=True,
        param_grid={"C": [0.1, 1.0, 10.0]},
    ),
    ModelSpec(
        "NuSVR",
        _factory(NuSVR),
        "regression",
        needs_scaling=True,
        param_grid={"C": [0.5, 1.0, 2.0], "nu": [0.25, 0.5, 0.75]},
    ),
    ModelSpec(
        "DecisionTreeRegressor",
        _factory(DecisionTreeRegressor),
        "regression",
        param_grid={"max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4]},
    ),
    ModelSpec(
        "ExtraTreeRegressor",
        _factory(ExtraTreeRegressor),
        "regression",
        param_grid={"max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4]},
    ),
    ModelSpec(
        "RandomForestRegressor",
        _factory(lambda: RandomForestRegressor(n_estimators=150)),
        "regression",
        param_grid={"max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 4]},
    ),
    ModelSpec(
        "ExtraTreesRegressor",
        _factory(lambda: ExtraTreesRegressor(n_estimators=150)),
        "regression",
        param_grid={"max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 4]},
    ),
    ModelSpec(
        "GradientBoostingRegressor",
        _factory(GradientBoostingRegressor),
        "regression",
        param_grid={"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    ),
    ModelSpec(
        "HistGradientBoostingRegressor",
        _factory(HistGradientBoostingRegressor),
        "regression",
        param_grid={"max_iter": [100, 200], "learning_rate": [0.05, 0.1]},
    ),
    ModelSpec(
        "AdaBoostRegressor",
        _factory(AdaBoostRegressor),
        "regression",
        param_grid={"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
    ),
    ModelSpec(
        "BaggingRegressor",
        _factory(lambda: BaggingRegressor(n_estimators=25)),
        "regression",
        param_grid={"n_estimators": [10, 25, 50]},
    ),
    ModelSpec(
        "MLPRegressor",
        _factory(
            lambda: MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True)
        ),
        "regression",
        needs_scaling=True,
        param_grid={"hidden_layer_sizes": [(64,), (100,), (100, 50)], "alpha": [0.0001, 0.001]},
    ),
    ModelSpec("DummyRegressor", _factory(lambda: DummyRegressor(strategy="mean")), "regression"),
    ModelSpec(
        "XGBRegressor",
        _optional_xgb_regressor,
        "regression",
        optional_dependency="xgboost",
        param_grid={"max_depth": [4, 6], "learning_rate": [0.03, 0.1]},
    ),
    ModelSpec(
        "LGBMRegressor",
        _optional_lgbm_regressor,
        "regression",
        optional_dependency="lightgbm",
        param_grid={"num_leaves": [31, 63], "learning_rate": [0.03, 0.1]},
    ),
    ModelSpec(
        "CatBoostRegressor",
        _optional_catboost_regressor,
        "regression",
        optional_dependency="catboost",
        param_grid={"depth": [4, 6, 8], "learning_rate": [0.03, 0.1]},
    ),
]


def get_model_registry(task: str) -> list[ModelSpec]:
    """Return the registry for a task."""

    return CLASSIFICATION_REGISTRY if task == "classification" else REGRESSION_REGISTRY


def get_model_spec(task: str, model_name: str) -> ModelSpec:
    """Return a single model spec by name."""

    registry = get_model_registry(task)
    for spec in registry:
        if spec.name.lower() == model_name.lower():
            return spec
    raise KeyError(model_name)


def make_estimator(
    spec: ModelSpec, random_state: int | None, n_jobs: int | None, use_gpu: bool = False
) -> BaseEstimator:
    """Build a fresh estimator instance from a model spec."""

    return spec.estimator_factory(random_state, n_jobs, use_gpu)
