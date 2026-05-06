"""Preprocessing helpers for tabular classification and regression."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_preprocessor(
    X_train: pd.DataFrame,
    *,
    needs_scaling: bool,
    categorical_encoder: str = "onehot",
    scale_numeric: str | bool = "auto",
) -> ColumnTransformer:
    """Create a leakage-safe tabular preprocessor."""

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    boolean_columns: list[str] = []

    for column in X_train.columns:
        dtype = X_train[column].dtype
        if is_bool_dtype(dtype):
            boolean_columns.append(column)
        elif is_numeric_dtype(dtype):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    use_scaler = scale_numeric is True or (scale_numeric == "auto" and needs_scaling)
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        numeric_steps.append(("scaler", StandardScaler()))

    if categorical_encoder == "ordinal":
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
    else:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(steps=numeric_steps), numeric_columns))
    if categorical_columns or boolean_columns:
        transformers.append(
            ("categorical", categorical_transformer, categorical_columns + boolean_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_model_pipeline(
    *,
    estimator: Any,
    X_train: pd.DataFrame,
    preprocess: bool,
    needs_scaling: bool,
    categorical_encoder: str,
    scale_numeric: str | bool,
) -> Any:
    """Wrap an estimator in a Pipeline when preprocessing is enabled."""

    if not preprocess:
        return estimator
    preprocessor = build_preprocessor(
        X_train,
        needs_scaling=needs_scaling,
        categorical_encoder=categorical_encoder,
        scale_numeric=scale_numeric,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def get_effective_feature_count(model: Any, X_reference: pd.DataFrame) -> int:
    """Best-effort feature count after preprocessing for adjusted R2."""

    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        try:
            return int(len(preprocessor.get_feature_names_out()))
        except Exception:
            try:
                transformed = preprocessor.transform(X_reference.iloc[: min(5, len(X_reference))])
                return int(transformed.shape[1])
            except Exception:
                pass
    return int(X_reference.shape[1])
