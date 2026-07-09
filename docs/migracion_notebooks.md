# Migración de notebooks a módulos — diff conceptual

Los dos notebooks originales fueron eliminados y su contenido migrado íntegramente a
módulos Python del paquete `codigo/ia_models/aldimi_models/`. Esta tabla documenta
dónde quedó cada bloque.

## `EDA_ALDIMI_Predict_v2_ampliado.ipynb` → `eda.py` + módulos comunes

| Sección del notebook | Destino |
|---|---|
| 1. Importación de librerías | Desaparece (imports por módulo, sin estado global) |
| 2. Configuración de rutas y extracción del ZIP | `config.py` (rutas fijas a `/datos`; ya no hay ZIP) |
| 3. Lectura de CSVs y conversión de fechas | `data_loading.py` (`load_pacientes`, `load_consumo`, `load_items`, `load_donaciones`, `load_capturas_ia`, `load_all`) con validación de esquema |
| 3-4. Resumen general, nulos por dataset | `eda.py::resumen_general`, `eda.py::resumen_nulos` |
| 5. EDA de pacientes (distribución del target, crosstabs, histogramas, boxplots) | `eda.py::_eda_pacientes` usando `common/plots.py` |
| 6. Correlaciones de pacientes | `eda.py::_eda_pacientes` + `common/plots.py::correlation_heatmap` |
| 7. Outliers IQR | `eda.py::detectar_outliers_iqr` |
| 8-9. EDA de inventario/consumo y correlaciones | `eda.py::_eda_consumo` |
| 10. EDA de donaciones | `eda.py::_eda_donaciones` |
| (implícito) capturas OCR | `eda.py::_eda_capturas_ia` (contrato con el curso de IA) |

Salidas: antes `outputs_eda_v2/` en la raíz; ahora `codigo/ia_models/outputs/eda/`.

## `baseline_hito2_aldimi.ipynb` → `train.py` / `evaluate.py` / módulos

| Sección del notebook | Destino |
|---|---|
| 1-2. Imports, rutas y descompresión | `config.py` + `data_loading.py` |
| 3-5. Lectura de pacientes, revisión y distribución del target | `data_loading.py` + `eda.py` |
| 6. Selección de features del modelo de prioridad | `features.py::prioridad_features` (constantes `PRIORIDAD_FEATURES_NUM/CAT`) |
| 7. Pipeline de preprocesamiento (imputer + one-hot) | `preprocessing.py::build_preprocessor` (único en el proyecto) |
| 7. División train/test estratificada | `preprocessing.py::stratified_split` |
| 8. Entrenamiento del Decision Tree baseline | `train.py::candidate_estimators` — el DT sigue siendo el baseline, ahora comparado contra RF y XGBoost |
| 9-10. Métricas, classification report, matriz de confusión | `common/metrics.py` + `evaluate.py::evaluate_model` |
| 11. Importancia de variables | `evaluate.py::feature_importance_frame` + `common/plots.py` |
| 12. Guardado de resultados | `common/metrics.py::save_dataframe` (salidas en `outputs/`) |
| 13.3. Corrección de fuga de información del modelo de stock | `config.py::STOCK_LEAKAGE_FEATURES` + `features.py::stock_critico_features` (ahora con verificación activa y tests en `tests/test_no_leakage.py`) |
| 13.x. Baseline de stock crítico 7d completo | `models/stock_critico.py` (specs 7d **y** 14d) + `train.py` |

## Qué se añadió respecto a los notebooks

- Frentes nuevos: `stock_critico_14d`, `demanda_consumo`, `duracion_estadia`,
  `proyeccion_donaciones` (`models/*.py`, features en `features.py`).
- Comparación automática DT vs RF vs XGBoost con selección por F1-macro/MAE.
- Split temporal para las series (el notebook usaba solo estratificado).
- Exportación a ONNX (`onnx_export.py`) y registro de modelos (`model_registry.json`).
- Inferencia CLI (`predict.py`), backend Rust y dashboard React.
- Tests (`pytest`: 20 tests; `cargo test`: 5 tests).

## Duplicaciones eliminadas

En el notebook, el pipeline de preprocesamiento, el cálculo de métricas, la matriz
de confusión y la importancia de variables estaban **copiadas dos veces** (una por
baseline). Ahora existen una sola vez en `preprocessing.py`, `common/metrics.py`,
`common/plots.py` y `evaluate.py`, parametrizadas por modelo.
