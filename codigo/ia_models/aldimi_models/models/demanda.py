"""(d) Regresion de la demanda diaria de consumo por item (features de lags)."""

from __future__ import annotations

from ..data_loading import load_consumo
from ..features import FeatureSet, demanda_features
from .spec import ModelSpec


def _build() -> FeatureSet:
    return demanda_features(load_consumo())


SPEC = ModelSpec(
    name="demanda_consumo",
    task="regression",
    description="Consumo diario esperado por item (serie con lags de 1/7/14 dias)",
    build_features=_build,
    split="temporal",
)
