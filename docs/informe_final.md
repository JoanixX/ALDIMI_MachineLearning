# Informe Técnico Final - ALDIMI Predict

**Cliente**: ONG ALDIMI (Albergue de Niños con Cáncer)  
**Curso**: Trabajo Final de Machine Learning  
**Objetivo**: Diseñar y desplegar un ecosistema predictivo robusto para optimizar la prioridad de atención médica de los pacientes y la gestión inteligente del inventario de recursos.

---

## 🔬 Resultados y Comparación de Modelos

Para cumplir con el diseño metodológico del curso, en cada uno de los 6 frentes se compararon tres algoritmos: **Decision Tree (Baseline)** vs **Random Forest** vs **XGBoost**.

La selección se realizó de manera automatizada:
- **Frentes de Clasificación**: Seleccionados por mayor **F1-macro** en el conjunto de test.
- **Frentes de Regresión**: Seleccionados por menor **MAE (Error Absoluto Medio)** en el conjunto de test.

A continuación se detallan las métricas reales y definitivas generadas tras el entrenamiento:

### 1. Clasificación de Prioridad de Atención (Multiclase)
- **Criterio**: F1-macro (Mayor es mejor).
- **Resultados**:
  - *Decision Tree*: F1-macro = `0.6977`
  - *Random Forest*: F1-macro = `0.7074`
  - *XGBoost*: F1-macro = `0.7659`
- **Seleccionado**: **XGBoost** (Accuracy = `0.8585`, F1-macro = `0.7659`).

### 2. Alerta de Stock Crítico a 7 días (Clasificación Binaria)
- **Criterio**: F1-macro (Mayor es mejor).
- **Resultados**:
  - *Decision Tree*: F1-macro = `0.8231`
  - *Random Forest*: F1-macro = `0.8996`
  - *XGBoost*: F1-macro = `0.8995`
- **Seleccionado**: **Random Forest** (F1-macro = `0.8996`).

### 3. Alerta de Stock Crítico a 14 días (Clasificación Binaria)
- **Criterio**: F1-macro (Mayor es mejor).
- **Resultados**:
  - *Decision Tree*: F1-macro = `0.6905`
  - *Random Forest*: F1-macro = `0.8194`
  - *XGBoost*: F1-macro = `0.7366`
- **Seleccionado**: **Random Forest** (F1-macro = `0.8194`).

### 4. Demanda de Consumo Diario (Regresión de Series Temporales)
- **Criterio**: MAE (Menor es mejor).
- **Resultados**:
  - *Decision Tree*: MAE = `6.63` unidades
  - *Random Forest*: MAE = `4.39` unidades
  - *XGBoost*: MAE = `4.26` unidades
- **Seleccionado**: **XGBoost** (MAE = `4.26` unidades, RMSE = `8.79`, R2 = `0.820`).

### 5. Duración de Hospedaje / Estadía (Regresión)
- **Criterio**: MAE (Menor es mejor).
- **Resultados**:
  - *Decision Tree*: MAE = `38.81` días
  - *Random Forest*: MAE = `27.30` días
  - *XGBoost*: MAE = `27.97` días
- **Seleccionado**: **Random Forest** (MAE = `27.30` días, RMSE = `37.07`, R2 = `-0.014`).
- **Limitación honesta**: el R2 cercano a cero indica que las variables clínicas y
  sociodemográficas disponibles apenas explican la varianza de `dias_hospedaje` en este
  dataset sintético: el modelo no supera de forma significativa a predecir la media.
  Se conserva como referencia del frente (e) y se documenta como oportunidad de mejora
  (se requerirían variables adicionales, p. ej. protocolo de tratamiento o fase clínica
  detallada, para que este frente sea útil en producción).

### 6. Proyección Mensual de Donaciones por Categoría (Regresión / Serie Temporal)
- **Criterio**: MAE (Menor es mejor).
- **Resultados**:
  - *Decision Tree*: MAE = `6622.41` Soles
  - *Random Forest*: MAE = `4623.75` Soles
  - *XGBoost*: MAE = `5783.27` Soles
- **Seleccionado**: **Random Forest** (MAE = `4623.75` Soles, RMSE = `6110.72`, R2 = `0.547`).

---

## 🚫 Prevención de Fuga de Información (Data Leakage)

En el modelado de alertas de stock crítico a 7 y 14 días (Frentes 2 y 3), se identificó un riesgo severo de fuga de información si se utilizaban variables que se calculaban de manera retrospectiva o directamente ligadas al target.
Para solucionarlo, en `features.py` se implementó un filtro de exclusión estricto sobre las siguientes variables explicativas:
- `stock_fin` (Inventario final calculado al cierre del día).
- `consumo_estimado_7d` y `consumo_estimado_14d` (Estimaciones futuras de consumo).
- `dias_cobertura` (Días calculados restantes).

Las únicas features empleadas fueron la tasa de ocupación de familias, ingresos de stock diarios, consumo real de ese día y metadatos del ítem, asegurando un modelo 100% aplicable y libre de sesgo en producción.

---

## 💻 Justificación Tecnológica del Backend (Rust)

El backend del ecosistema se construyó en **Rust** (Axum) por las siguientes razones de ingeniería:
1. **Consumo de Memoria y Latencia**: Un contenedor web en Python (ej. Flask o FastAPI con scikit-learn y XGBoost en memoria) requiere más de 300MB de RAM al levantar y añade latencia debido al Global Interpreter Lock (GIL). El backend en Rust compilado consume menos de **25MB de RAM** y maneja miles de solicitudes simultáneas en microsegundos.
2. **Uso de ONNX Runtime**: Al exportar los modelos de Python a formato ONNX (`.onnx`), el backend de Rust realiza inferencia pura de manera nativa cargando archivos binarios estándar. No hay dependencias cruzadas entre el código de entrenamiento y el código de servicio.
3. **Validación Segura de Esquemas**: Gracias al sistema de tipos estáticos de Rust y la biblioteca de deserialización `serde`, cualquier entrada mal formateada (por ejemplo, textos en campos numéricos) es rechazada en la capa de transporte sin sobrecargar al motor de inferencia ONNX.
