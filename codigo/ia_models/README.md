# ALDIMI Predict - Pipeline de Machine Learning

Este subproyecto contiene el código puramente matemático, estadístico y de entrenamiento de modelos para la ONG ALDIMI. La refactorización ha eliminado por completo las celdas y el estado oculto de los cuadernos de Jupyter (`.ipynb`), transformándolos en un paquete de Python estructurado y modular.

## 📁 Estructura del Módulo `aldimi_models`

- [data_loading.py](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/codigo/ia_models/aldimi_models/data_loading.py): Único módulo con permiso para leer archivos CSV de la carpeta `/datos`. Valida la presencia de esquemas y realiza el parseo de fechas.
- [preprocessing.py](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/codigo/ia_models/aldimi_models/preprocessing.py): Define el preprocesador compartido (SimpleImputer sobre numéricas + OneHotEncoder sobre categóricas) y los algoritmos de partición de datos.
- [features.py](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/codigo/ia_models/aldimi_models/features.py): Construcción y tratamiento de variables explicativas por frente. Contiene reglas de exclusión para evitar fugas de información.
- [train.py](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/codigo/ia_models/aldimi_models/train.py): Compara explícitamente Decision Tree vs Random Forest vs XGBoost en cada frente de modelado. Selecciona el mejor y lo exporta.
- [evaluate.py](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/codigo/ia_models/aldimi_models/evaluate.py): Evaluación del modelo ganador contra el conjunto de test. Genera reportes de clasificación, matrices de confusión y gráficos de importancia de variables.
- [predict.py](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/codigo/ia_models/aldimi_models/predict.py): Utilidad CLI para realizar predicciones sobre casos puntuales en formato JSON usando la sesión local de ONNX.

---

## 🛠️ Comandos de Uso CLI

1. **Entrenar todos los modelos** (comparando clasificadores y regresores):
   ```bash
   python -m aldimi_models.train
   ```
   *Esto regenerará los archivos `.onnx` en `artifacts/` y el archivo `artifacts/model_registry.json`.*

2. **Entrenar solo modelos específicos**:
   ```bash
   python -m aldimi_models.train --models prioridad_atencion stock_critico_7d
   ```

3. **Evaluar los modelos ganadores**:
   ```bash
   python -m aldimi_models.evaluate
   ```
   *Genera los archivos descriptivos en la carpeta `outputs/evaluacion/`.*

4. **Realizar una predicción desde consola**:
   Puedes pasar datos en formato JSON mediante la entrada estándar (stdin) o archivos:
   ```bash
   # Inferencia de prioridad de atención para un paciente
   echo {"edad": 15, "sexo": "M", "region_origen": "Lima", "diagnostico_general": "Linfoma", "estado_tratamiento": "Inducción", "dias_hospedaje": 12, "num_controles_mes": 3, "num_quimios_mes": 2, "hemoglobina_g_dl": 11.5, "neutrofilos": 1500, "plaquetas": 200000, "temperatura_c": 37.2, "peso_kg": 54.0, "imc": 21.0, "distancia_origen_km": 15.0, "ingreso_familiar_mensual": 1200.0, "acompanante_presente": 1, "seguro_salud": "SIS", "alfabetizacion_digital": "Básica", "requiere_apoyo_psicosocial": 0} | python -m aldimi_models.predict --model prioridad_atencion
   ```

---

## 🧪 Pruebas Automatizadas

El proyecto incluye 20 tests unitarios implementados en `tests/` que validan:
1. La carga e integridad de los datasets en formato CSV.
2. Que no existan fugas de información en los modelos de stock crítico.
3. La lógica de cálculo de métricas de clasificación y regresión.

Para ejecutarlos, simplemente corre:
```bash
pytest
```
