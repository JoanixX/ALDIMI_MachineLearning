"""Preprocesamiento comun: imputacion, one-hot y divisiones train/test."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import RANDOM_STATE, TEST_SIZE


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """ColumnTransformer unico del proyecto: mediana + one-hot.

    Las categoricas no llevan imputer dentro del pipeline: FeatureSet ya
    rellena sus nulos con "desconocido" (skl2onnx no convierte SimpleImputer
    sobre strings, y este pipeline debe ser exportable a ONNX).
    """
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def stratified_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Division estratificada reproducible (misma semilla en todo el proyecto)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def temporal_split(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Division temporal para series: el ultimo tramo de fechas es test.

    Evita evaluar prediciendo el pasado con datos del futuro.
    """
    if len(X) != len(dates):
        raise ValueError("X y dates deben tener la misma longitud")
    cutoff = dates.sort_values().iloc[int(len(dates) * (1 - test_size))]
    train_mask = dates < cutoff
    if train_mask.all() or not train_mask.any():
        raise ValueError("La division temporal dejo un conjunto vacio; revisa las fechas")
    return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]
