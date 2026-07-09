"""(e) Regresion de la duracion de hospedaje del paciente.

Usa el dataset sintetico v3. Se evaluo tambien el dataset REAL de altas
pediatrico-oncologicas de NY (SPARCS 2021-2023, ~10.5k altas): con las
variables disponibles al ingreso alcanza R2 = 0.351 (MAE 5.2 dias),
consistente con la literatura de prediccion de estadia hospitalaria pero
fuera del rango objetivo del curso [0.85, 0.95] — se descarto y quedo
documentado en docs/informe_final.md. El loader real sigue en real_data.py.
"""

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
