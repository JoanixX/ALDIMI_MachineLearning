"""Entrenamiento de los seis modelos de ALDIMI Predict.

Para cada frente compara Decision Tree (baseline) vs Random Forest vs XGBoost,
afinando hiperparametros con RandomizedSearchCV (validacion cruzada k-fold;
TimeSeriesSplit para los frentes temporales) y tratando el desbalance de
clases via class_weight / scale_pos_weight dentro de la busqueda. La seleccion
final es por F1-macro (clasificacion) o MAE (regresion) sobre un test set
intacto que nunca participa en la busqueda.

Guarda:
- ``outputs/entrenamiento/<modelo>_comparacion_algoritmos.csv``
- ``artifacts/<modelo>.pkl``  (pipeline sklearn del mejor algoritmo)
- ``artifacts/<modelo>.onnx`` (mismo pipeline exportado para el backend Rust)
- ``artifacts/model_registry.json`` (metricas reales + esquema de entrada)

Uso:
    python -m aldimi_models.train [--models ...] [--sin-tuning]
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
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


def candidate_estimators(task: str, pos_ratio: float | None = None) -> dict[str, tuple[object, dict]]:
    """(estimador base, espacio de busqueda) por algoritmo.

    ``pos_ratio``: proporcion de la clase positiva en clasificacion binaria,
    usada para el rango de ``scale_pos_weight`` de XGBoost.
    """
    if task == "classification":
        spw = [1.0] if not pos_ratio else [1.0, (1 - pos_ratio) / max(pos_ratio, 1e-6)]
        return {
            "decision_tree": (
                DecisionTreeClassifier(random_state=RANDOM_STATE),
                {
                    "model__max_depth": [4, 6, 8, 12, 16, 24, None],
                    "model__min_samples_leaf": randint(1, 40),
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "random_forest": (
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                {
                    "model__n_estimators": [200, 300, 500],
                    "model__max_depth": [None, 12, 18, 26],
                    "model__min_samples_leaf": randint(1, 10),
                    "model__max_features": ["sqrt", 0.5],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "xgboost": (
                XGBClassifier(
                    random_state=RANDOM_STATE, n_jobs=1, eval_metric="mlogloss",
                    tree_method="hist",
                ),
                {
                    "model__n_estimators": [200, 400, 600],
                    "model__max_depth": randint(3, 10),
                    "model__learning_rate": uniform(0.03, 0.17),
                    "model__subsample": uniform(0.7, 0.3),
                    "model__colsample_bytree": uniform(0.7, 0.3),
                    "model__min_child_weight": randint(1, 10),
                    "model__scale_pos_weight": spw,
                },
            ),
        }
    if task == "regression":
        return {
            "decision_tree": (
                DecisionTreeRegressor(random_state=RANDOM_STATE),
                {
                    "model__max_depth": [4, 6, 8, 12, 16, None],
                    "model__min_samples_leaf": randint(1, 40),
                },
            ),
            "random_forest": (
                RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
                {
                    "model__n_estimators": [200, 300, 500],
                    "model__max_depth": [None, 12, 18, 26],
                    "model__min_samples_leaf": randint(1, 10),
                    "model__max_features": ["sqrt", 0.5, 1.0],
                },
            ),
            "xgboost": (
                XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"),
                {
                    "model__n_estimators": [200, 400, 600],
                    "model__max_depth": randint(3, 10),
                    "model__learning_rate": uniform(0.03, 0.17),
                    "model__subsample": uniform(0.7, 0.3),
                    "model__colsample_bytree": uniform(0.7, 0.3),
                    "model__min_child_weight": randint(1, 10),
                },
            ),
        }
    raise ValueError(f"Tarea desconocida: {task}")


def split_features(
    spec: ModelSpec, fs: FeatureSet
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series | None]:
    """Division reproducible; devuelve tambien las fechas del train (para CV temporal)."""
    if spec.split == "temporal":
        if fs.dates is None:
            raise ValueError(f"{spec.name}: split temporal requiere fechas en el FeatureSet")
        X_train, X_test, y_train, y_test = temporal_split(fs.X, fs.y, fs.dates)
        dates_train = fs.dates.loc[X_train.index]
        return X_train, X_test, y_train, y_test, dates_train
    if spec.task == "classification":
        X_train, X_test, y_train, y_test = stratified_split(fs.X, fs.y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            fs.X, fs.y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    return X_train, X_test, y_train, y_test, None


def _cv_and_order(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates_train: pd.Series | None,
):
    """CV honesta: TimeSeriesSplit (datos ordenados por fecha) para series,
    StratifiedKFold para clasificacion transversal, KFold implicito si no."""
    if spec.split == "temporal" and dates_train is not None:
        order = dates_train.sort_values(kind="stable").index
        return TimeSeriesSplit(n_splits=3), X_train.loc[order], y_train.loc[order]
    if spec.task == "classification":
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE), X_train, y_train
    return 3, X_train, y_train


def train_model(spec: ModelSpec, tuning: bool = True, n_iter: int = 15) -> dict[str, object]:
    """Entrena y compara los tres algoritmos de un frente; devuelve la entrada del registro."""
    print(f"[{spec.name}] construyendo features...")
    fs = spec.build_features()
    X_train, X_test, y_train, y_test, dates_train = split_features(spec, fs)
    print(f"[{spec.name}] train={len(X_train)} test={len(X_test)} "
          f"features={len(fs.numeric_features) + len(fs.categorical_features)}")

    selection_metric = "f1_macro" if spec.task == "classification" else "mae"
    scoring = "f1_macro" if spec.task == "classification" else "neg_mean_absolute_error"
    pos_ratio = None
    if spec.task == "classification" and y_train.nunique() == 2:
        pos_ratio = float(y_train.mean())

    cv, X_search, y_search = _cv_and_order(spec, X_train, y_train, dates_train)

    comparison_rows: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}
    best_params_by_algo: dict[str, dict] = {}

    for algo_name, (estimator, param_space) in candidate_estimators(spec.task, pos_ratio).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(fs.numeric_features, fs.categorical_features)),
                ("model", estimator),
            ]
        )
        if tuning:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_space,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_search, y_search)
            fitted_pipeline = search.best_estimator_
            best_params = {
                k.removeprefix("model__"): v for k, v in search.best_params_.items()
            }
            cv_score = float(search.best_score_)
        else:
            fitted_pipeline = pipeline.fit(X_train, y_train)
            best_params, cv_score = {}, None  # None (no NaN): el registry es JSON estricto

        y_pred = fitted_pipeline.predict(X_test)
        if spec.task == "classification":
            metrics = classification_metrics(y_test, y_pred)
        else:
            metrics = regression_metrics(y_test, y_pred)
        comparison_rows.append(
            {"algoritmo": algo_name, "cv_score": cv_score, **metrics,
             "mejores_hiperparametros": str(best_params)}
        )
        fitted[algo_name] = fitted_pipeline
        best_params_by_algo[algo_name] = best_params
        cv_txt = f"{cv_score:.4f}" if cv_score is not None else "sin tuning"
        print(f"[{spec.name}] {algo_name}: {selection_metric}={metrics[selection_metric]:.4f} "
              f"(cv={cv_txt})")

    comparison = pd.DataFrame(comparison_rows)
    ascending = spec.task == "regression"  # MAE: menor es mejor
    comparison = comparison.sort_values(selection_metric, ascending=ascending).reset_index(drop=True)
    save_dataframe(comparison, TRAIN_OUTPUT_DIR / f"{spec.name}_comparacion_algoritmos.csv")

    best_algo = str(comparison.iloc[0]["algoritmo"])
    best_pipeline = fitted[best_algo]
    metric_cols = [c for c in comparison.columns if c not in ("algoritmo", "mejores_hiperparametros")]
    best_metrics = {k: float(comparison.iloc[0][k]) for k in metric_cols if pd.notna(comparison.iloc[0][k])}
    print(f"[{spec.name}] mejor algoritmo: {best_algo} ({best_params_by_algo[best_algo]})")

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
        "best_hyperparameters": {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in best_params_by_algo[best_algo].items()
        },
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


def main(model_names: list[str] | None = None, tuning: bool = True, n_iter: int = 15) -> None:
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
        registry[name] = train_model(MODEL_SPECS[name], tuning=tuning, n_iter=n_iter)

    save_json(list(registry.values()), REGISTRY_PATH)
    print(f"Registro de modelos actualizado: {REGISTRY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None, help="Subconjunto de modelos a entrenar")
    parser.add_argument("--sin-tuning", action="store_true", help="Entrena con hiperparametros por defecto")
    parser.add_argument("--n-iter", type=int, default=15, help="Iteraciones de RandomizedSearchCV")
    args = parser.parse_args()
    main(args.models, tuning=not args.sin_tuning, n_iter=args.n_iter)
