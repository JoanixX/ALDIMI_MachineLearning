"""Registro central de los seis frentes de modelado de ALDIMI Predict."""

from __future__ import annotations

from .spec import ModelSpec
from . import demanda, donaciones, estadia, prioridad, stock_critico

MODEL_SPECS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        prioridad.SPEC,
        stock_critico.SPEC_7D,
        stock_critico.SPEC_14D,
        demanda.SPEC,
        estadia.SPEC,
        donaciones.SPEC,
    )
}

__all__ = ["MODEL_SPECS", "ModelSpec"]
