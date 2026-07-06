"""(a) Clasificacion multiclase del nivel de prioridad de atencion."""

from __future__ import annotations

from ..data_loading import load_pacientes
from ..features import FeatureSet, prioridad_features
from .spec import ModelSpec


def _build() -> FeatureSet:
    return prioridad_features(load_pacientes())


SPEC = ModelSpec(
    name="prioridad_atencion",
    task="classification",
    description="Nivel de prioridad de atencion del paciente (Bajo/Medio/Alto)",
    build_features=_build,
    split="stratified",
)
