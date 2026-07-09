"""Evaluacion de los modelos entrenados sobre su conjunto de test.

Reconstruye la misma division train/test que train.py (misma semilla y
estrategia), carga el pipeline ganador desde artifacts/ y genera en
``outputs/evaluacion/``:

- ``metrics_summary.csv`` global con las metricas reales de los 6 modelos
- por clasificador: classification report, matriz de confusion (csv + png)
  e importancia de variables (csv + png)
- por regresor: dispersion prediccion vs real e importancia de variables

Uso:
    python -m aldimi_models.evaluate [--models ...]
"""

from __future__ import annotations

import argparse
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline

from .common import plots
from .common.metrics import (
    classification_metrics,
    classification_report_frame,
    confusion_matrix_frame,
    load_json,
    regression_metrics,
    save_dataframe,
)
from .config import ARTIFACTS_DIR, OUTPUTS_DIR
from .models import MODEL_SPECS
from .train import REGISTRY_PATH, split_features

EVAL_OUTPUT_DIR = OUTPUTS_DIR / "evaluacion"


def load_pipeline(name: str) -> Pipeline:
    """Carga el pipeline ganador serializado por train.py."""
    path = ARTIFACTS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero: python -m aldimi_models.train"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def feature_importance_frame(pipeline: Pipeline) -> pd.DataFrame | None:
    """Importancia de variables del modelo final, con nombres legibles."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    frame = (
        pd.DataFrame({"variable": names, "importancia": model.feature_importances_})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    frame["variable"] = (
        frame["variable"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    return frame


def evaluate_model(name: str, registry_entry: dict[str, object]) -> dict[str, object]:
    """Evalua un modelo sobre su test set y escribe sus reportes."""
    spec = MODEL_SPECS[name]
    fs = spec.build_features()
    _, X_test, _, y_test, _ = split_features(spec, fs)
    pipeline = load_pipeline(name)
    y_pred = pipeline.predict(X_test)

    row: dict[str, object] = {
        "modelo": name,
        "tarea": spec.task,
        "algoritmo": registry_entry["selected_algorithm"],
        "n_test": len(X_test),
    }

    if spec.task == "classification":
        row.update(classification_metrics(y_test, y_pred))
        labels = list(range(len(fs.class_labels or []))) or sorted(set(y_test))
        display = fs.class_labels or [str(c) for c in labels]
        report = classification_report_frame(y_test, y_pred)
        save_dataframe(report, EVAL_OUTPUT_DIR / f"{name}_classification_report.csv", index=True)
        cm = confusion_matrix_frame(y_test, y_pred, labels)
        cm.index = [f"Real_{c}" for c in display]
        cm.columns = [f"Pred_{c}" for c in display]
        save_dataframe(cm, EVAL_OUTPUT_DIR / f"{name}_confusion_matrix.csv", index=True)
        plots.confusion_matrix_heatmap(
            cm, f"Matriz de confusion - {name}", EVAL_OUTPUT_DIR / f"{name}_confusion_matrix.png"
        )
    else:
        row.update(regression_metrics(y_test, y_pred))
        plots.predictions_vs_actual(
            list(y_test), list(y_pred),
            f"Prediccion vs real - {name}",
            EVAL_OUTPUT_DIR / f"{name}_pred_vs_real.png",
        )

    importance = feature_importance_frame(pipeline)
    if importance is not None:
        save_dataframe(importance, EVAL_OUTPUT_DIR / f"{name}_feature_importance.csv")
        plots.feature_importance_barh(
            importance, f"Importancia de variables - {name}",
            EVAL_OUTPUT_DIR / f"{name}_feature_importance.png",
        )
    return row


def main(model_names: list[str] | None = None) -> None:
    registry = {entry["name"]: entry for entry in load_json(REGISTRY_PATH)}
    names = model_names or list(registry)
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(f"Modelos sin entrenar o desconocidos: {unknown}")

    summary_rows = []
    for name in names:
        print(f"[evaluate] {name}...")
        summary_rows.append(evaluate_model(name, registry[name]))

    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, EVAL_OUTPUT_DIR / "metrics_summary.csv")
    print(f"Resumen de metricas: {EVAL_OUTPUT_DIR / 'metrics_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    main(args.models)
