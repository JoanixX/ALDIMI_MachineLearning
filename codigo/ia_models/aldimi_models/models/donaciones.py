"""(f) Proyeccion mensual del valor de donaciones por categoria."""

from __future__ import annotations

from ..data_loading import load_donaciones
from ..features import FeatureSet, donaciones_features
from .spec import ModelSpec


def _build() -> FeatureSet:
    return donaciones_features(load_donaciones())


SPEC = ModelSpec(
    name="proyeccion_donaciones",
    task="regression",
    description="Valor mensual proyectado de donaciones (S/) por categoria",
    build_features=_build,
    split="temporal",
)
