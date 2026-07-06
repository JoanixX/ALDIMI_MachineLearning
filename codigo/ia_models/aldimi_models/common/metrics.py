"""Calculo de metricas y escritura de resultados. Modulo compartido unico."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from ..config import CSV_ENCODING


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
    """Metricas estandar de clasificacion; F1-macro es el criterio de seleccion."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Metricas estandar de regresion; MAE es el criterio de seleccion."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_report_frame(y_true: Sequence[Any], y_pred: Sequence[Any]) -> pd.DataFrame:
    """Reporte por clase de sklearn como DataFrame."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def confusion_matrix_frame(
    y_true: Sequence[Any], y_pred: Sequence[Any], labels: list[Any]
) -> pd.DataFrame:
    """Matriz de confusion etiquetada Real_x / Pred_x."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"Real_{c}" for c in labels],
        columns=[f"Pred_{c}" for c in labels],
    )


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Guarda un DataFrame como CSV creando la carpeta si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding=CSV_ENCODING)


def save_json(data: dict[str, Any] | list[Any], path: Path) -> None:
    """Guarda un objeto serializable como JSON legible (utf-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Lee un JSON generado por este paquete."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
