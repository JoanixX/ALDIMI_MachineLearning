"""(b)/(c) Clasificacion de stock critico a 7 y 14 dias, sin fuga de informacion."""

from __future__ import annotations

from ..data_loading import load_consumo
from ..features import FeatureSet, stock_critico_features
from .spec import ModelSpec


def _build_7d() -> FeatureSet:
    return stock_critico_features(load_consumo(), horizon_days=7)


def _build_14d() -> FeatureSet:
    return stock_critico_features(load_consumo(), horizon_days=14)


SPEC_7D = ModelSpec(
    name="stock_critico_7d",
    task="classification",
    description="Alerta de stock critico a 7 dias por item de inventario",
    build_features=_build_7d,
    split="temporal",
)

SPEC_14D = ModelSpec(
    name="stock_critico_14d",
    task="classification",
    description="Alerta de stock critico a 14 dias por item de inventario",
    build_features=_build_14d,
    split="temporal",
)
