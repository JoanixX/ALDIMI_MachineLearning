"""Definicion del contrato que cumple cada frente de modelado."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from ..features import FeatureSet

Task = Literal["classification", "regression"]
SplitStrategy = Literal["stratified", "temporal"]


@dataclass(frozen=True)
class ModelSpec:
    """Describe un modelo: de donde salen sus datos y como se evalua.

    - ``build_features`` carga los datos y devuelve el FeatureSet listo.
    - ``split``: "stratified" para datos transversales, "temporal" para series
      (el test es el ultimo tramo de fechas).
    - La seleccion del mejor algoritmo es por F1-macro (clasificacion) o
      MAE (regresion); ver train.py.
    """

    name: str
    task: Task
    description: str
    build_features: Callable[[], FeatureSet]
    split: SplitStrategy
