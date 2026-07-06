"""Inferencia puntual desde CLI usando los artefactos ONNX.

Valida que los .onnx exportados producen las mismas predicciones que serviria
el backend Rust. Entrada: JSON con un objeto o lista de objetos
feature -> valor.

Uso:
    python -m aldimi_models.predict --model prioridad_atencion --input caso.json
    echo {...} | python -m aldimi_models.predict --model duracion_estadia
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .common.metrics import load_json
from .config import ARTIFACTS_DIR
from .train import REGISTRY_PATH


class PredictionInputError(Exception):
    """La entrada no cumple el esquema de features del modelo."""


def _build_feeds(
    records: list[dict[str, Any]],
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for record in records:
        missing = [
            col for col in numeric_features + categorical_features if col not in record
        ]
        if missing:
            raise PredictionInputError(f"Faltan features en la entrada: {missing}")
    for col in numeric_features:
        try:
            values = [float(r[col]) for r in records]
        except (TypeError, ValueError) as exc:
            raise PredictionInputError(f"'{col}' debe ser numerico: {exc}") from exc
        feeds[col] = np.array(values, dtype=np.float32).reshape(-1, 1)
    for col in categorical_features:
        feeds[col] = np.array([str(r[col]) for r in records], dtype=object).reshape(-1, 1)
    return feeds


def predict(model_name: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ejecuta el .onnx del modelo y devuelve una prediccion por registro."""
    registry = {entry["name"]: entry for entry in load_json(REGISTRY_PATH)}
    if model_name not in registry:
        raise PredictionInputError(
            f"Modelo desconocido '{model_name}'. Disponibles: {list(registry)}"
        )
    entry = registry[model_name]
    if not entry.get("onnx_file"):
        raise PredictionInputError(f"El modelo {model_name} no tiene ONNX exportado")

    session = ort.InferenceSession(
        str(ARTIFACTS_DIR / entry["onnx_file"]), providers=["CPUExecutionProvider"]
    )
    feeds = _build_feeds(records, entry["numeric_features"], entry["categorical_features"])
    outputs = session.run(None, feeds)

    results: list[dict[str, Any]] = []
    if entry["task"] == "classification":
        labels, probabilities = outputs[0], outputs[1]
        class_labels: list[str] = entry["class_labels"]
        for i in range(len(records)):
            probs = [float(p) for p in np.asarray(probabilities[i]).ravel()]
            idx = int(labels[i])
            results.append(
                {
                    "prediccion": class_labels[idx],
                    "clase_indice": idx,
                    "probabilidades": dict(zip(class_labels, probs)),
                }
            )
    else:
        values = np.asarray(outputs[0]).ravel()
        results = [{"prediccion": float(v)} for v in values]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Nombre del modelo en el registro")
    parser.add_argument("--input", type=Path, default=None, help="JSON de entrada (default: stdin)")
    args = parser.parse_args()

    raw = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Entrada JSON invalida: {exc}") from exc
    records = payload if isinstance(payload, list) else [payload]

    try:
        results = predict(args.model, records)
    except PredictionInputError as exc:
        raise SystemExit(f"Error de entrada: {exc}") from exc
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
