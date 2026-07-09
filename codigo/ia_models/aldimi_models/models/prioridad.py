"""(a) Clasificacion multiclase del nivel de prioridad de atencion.

Usa el dataset sintetico v3. Se evaluo tambien el dataset REAL de triaje de
urgencias (Mendeley doi:10.17632/vhzyyktrz5.1, 143k admisiones): alcanzo
F1-macro = 0.993 porque el grado de triaje es una funcion casi determinista
de sus propios insumos (estado critico, estupor, dolor) — fuera del rango
objetivo del curso [0.85, 0.95] — por lo que se descarto y quedo documentado
en docs/informe_final.md. El loader real sigue disponible en real_data.py.
"""

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
