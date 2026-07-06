"""Entrenamiento de los seis modelos de ALDIMI Predict.

Para cada frente compara Decision Tree (baseline) vs Random Forest vs XGBoost
y selecciona el mejor por F1-macro (clasificacion) o MAE (regresion). Guarda:

- ``outputs/entrenamiento/<modelo>_comparacion_algoritmos.csv``
- ``artifacts/<modelo>.pkl``  (pipeline sklearn del mejor algoritmo)
- ``artifacts/<modelo>.onnx`` (mismo pipeline exportado para el backend Rust)
- ``artifacts/model_registry.json`` (metricas reales + esquema de entrada)

Uso:
    python -m aldimi_models.train [--models prioridad_atencion stock_critico_7d ...]
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from .common.metrics import (
    classification_metrics,
    regression_metrics,
    save_dataframe,
    save_json,
)
from .config import ARTIFACTS_DIR, OUTPUTS_DIR, RANDOM_STATE, TEST_SIZE
from .features import FeatureSet
from .models import MODEL_SPECS, ModelSpec
from .onnx_export import OnnxExportError, export_pipeline
from .preprocessing import build_preprocessor, stratified_split, temporal_split

TRAIN_OUTPUT_DIR = OUTPUTS_DIR / "entrenamiento"
REGISTRY_PATH = ARTIFACTS_DIR / "model_registry.json"


def candidate_estimators(task: str) -> dict[str, object]:
    """Los tres algoritmos comparados en cada frente."""
    if task == "classification":
        return {
            "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "random_forest": RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
            ),
            "xgboost": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss",
            ),
        }
    if task == "regression":
        return {
            "decision_tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "random_forest": RandomForestRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
            ),
            "xgboost": XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }
    raise ValueError(f"Tarea desconocida: {task}")


def split_features(spec: ModelSpec, fs: FeatureSet) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Division reproducible segun la estrategia declarada en el spec."""
    if spec.split == "temporal":
        if fs.dates is None:
            raise ValueError(f"{spec.name}: split temporal requiere fechas en el FeatureSet")
        return temporal_split(fs.X, fs.y, fs.dates)
    if spec.task == "classification":
        return stratified_split(fs.X, fs.y)
    return train_test_split(fs.X, fs.y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_model(spec: ModelSpec) -> dict[str, object]:
    """Entrena y compara los tres algoritmos de un frente; devuelve la entrada del registro."""
    print(f"[{spec.name}] construyendo features...")
    fs = spec.build_features()
    X_train, X_test, y_train, y_test = split_features(spec, fs)
    print(f"[{spec.name}] train={len(X_train)} test={len(X_test)}")

    selection_metric = "f1_macro" if spec.task == "classification" else "mae"
    comparison_rows: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}

    for algo_name, estimator in candidate_estimators(spec.task).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(fs.numeric_features, fs.categorical_features)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if spec.task == "classification":
            metrics = classification_metrics(y_test, y_pred)
        else:
            metrics = regression_metrics(y_test, y_pred)
        comparison_rows.append({"algoritmo": algo_name, **metrics})
        fitted[algo_name] = pipeline
        print(f"[{spec.name}] {algo_name}: {selection_metric}={metrics[selection_metric]:.4f}")

    comparison = pd.DataFrame(comparison_rows)
    ascending = spec.task == "regression"  # MAE: menor es mejor
    comparison = comparison.sort_values(selection_metric, ascending=ascending).reset_index(drop=True)
    save_dataframe(comparison, TRAIN_OUTPUT_DIR / f"{spec.name}_comparacion_algoritmos.csv")

    best_algo = str(comparison.iloc[0]["algoritmo"])
    best_pipeline = fitted[best_algo]
    best_metrics = {
        k: float(v) for k, v in comparison.iloc[0].items() if k != "algoritmo"
    }
    print(f"[{spec.name}] mejor algoritmo: {best_algo}")

    pkl_path = ARTIFACTS_DIR / f"{spec.name}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(best_pipeline, fh)

    onnx_path = ARTIFACTS_DIR / f"{spec.name}.onnx"
    try:
        export_pipeline(best_pipeline, fs, onnx_path)
        onnx_file: str | None = onnx_path.name
        print(f"[{spec.name}] ONNX exportado: {onnx_path.name}")
    except OnnxExportError as exc:
        onnx_file = None
        print(f"[{spec.name}] ADVERTENCIA: export ONNX fallo: {exc}")

    # Valores vistos de cada categorica: el frontend construye sus selects
    # con esto y el backend los expone en /models.
    categorical_values: dict[str, list[str]] = {}
    if fs.categorical_features:
        ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        categorical_values = {
            col: [str(v) for v in values]
            for col, values in zip(fs.categorical_features, ohe.categories_)
        }

    return {
        "name": spec.name,
        "task": spec.task,
        "description": spec.description,
        "split": spec.split,
        "selection_metric": selection_metric,
        "selected_algorithm": best_algo,
        "metrics": best_metrics,
        "all_algorithms": comparison_rows,
        "numeric_features": fs.numeric_features,
        "categorical_features": fs.categorical_features,
        "categorical_values": categorical_values,
        "class_labels": fs.class_labels,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "onnx_file": onnx_file,
        "pickle_file": pkl_path.name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def main(model_names: list[str] | None = None) -> None:
    names = model_names or list(MODEL_SPECS)
    unknown = [n for n in names if n not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {list(MODEL_SPECS)}")

    # Actualiza el registro de forma incremental para poder reentrenar un solo modelo.
    registry: dict[str, dict[str, object]] = {}
    if REGISTRY_PATH.exists():
        from .common.metrics import load_json

        registry = {entry["name"]: entry for entry in load_json(REGISTRY_PATH)}

    for name in names:
        registry[name] = train_model(MODEL_SPECS[name])

    save_json(list(registry.values()), REGISTRY_PATH)
    print(f"Registro de modelos actualizado: {REGISTRY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None, help="Subconjunto de modelos a entrenar")
    args = parser.parse_args()
    main(args.models)
