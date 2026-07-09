"""Exportacion de pipelines scikit-learn/XGBoost a ONNX.

El backend Rust (crate ``ort``) consume estos .onnx directamente, por lo que
la inferencia en produccion no necesita Python. Cada columna del FeatureSet
se declara como input ONNX independiente con su nombre original: el backend
arma el feed por nombre de columna.
"""

from __future__ import annotations

from pathlib import Path

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)
from sklearn.pipeline import Pipeline
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from xgboost import XGBClassifier, XGBRegressor

from .features import FeatureSet


class OnnxExportError(Exception):
    """La conversion a ONNX fallo para un pipeline entrenado."""


def _register_xgboost_converters() -> None:
    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )
    update_registered_converter(
        XGBRegressor,
        "XGBoostXGBRegressor",
        calculate_linear_regressor_output_shapes,
        convert_xgboost,
    )


def export_pipeline(pipeline: Pipeline, feature_set: FeatureSet, path: Path) -> None:
    """Convierte el pipeline entrenado a ONNX y lo escribe en ``path``."""
    _register_xgboost_converters()

    initial_types = [
        (col, FloatTensorType([None, 1])) for col in feature_set.numeric_features
    ] + [
        (col, StringTensorType([None, 1])) for col in feature_set.categorical_features
    ]

    final_estimator = pipeline.steps[-1][1]
    options = None
    if hasattr(final_estimator, "predict_proba"):
        # zipmap=False devuelve las probabilidades como tensor plano,
        # mucho mas simple de leer desde Rust que una secuencia de mapas.
        options = {id(final_estimator): {"zipmap": False}}

    try:
        # ai.onnx.ml se fija en 3: el conversor de XGBoost (onnxmltools) aun
        # no soporta la version 5 de ese dominio.
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_types,
            options=options,
            target_opset={"": 17, "ai.onnx.ml": 3},
        )
    except Exception as exc:  # skl2onnx lanza tipos heterogeneos
        raise OnnxExportError(f"No se pudo convertir el pipeline a ONNX: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_model.SerializeToString())
