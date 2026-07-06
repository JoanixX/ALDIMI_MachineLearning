"""(e) Regresion de la duracion de hospedaje del paciente."""

from __future__ import annotations

from ..data_loading import load_pacientes
from ..features import FeatureSet, estadia_features
from .spec import ModelSpec


def _build() -> FeatureSet:
    return estadia_features(load_pacientes())


SPEC = ModelSpec(
    name="duracion_estadia",
    task="regression",
    description="Dias de hospedaje estimados del paciente al momento del ingreso",
    build_features=_build,
    split="stratified",
)
