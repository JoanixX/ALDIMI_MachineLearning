# ALDIMI Predict - Sistema de Producción de Modelos de ML

Este proyecto es la solución de Machine Learning (ML) para la ONG **ALDIMI**, diseñado para optimizar la prioridad de atención de pacientes y gestionar eficientemente el inventario de recursos en sus albergues. 

El sistema está diseñado bajo un ecosistema de producción gratuito, robusto y desacoplado, estructurado en tres componentes principales:
1. **Pipeline de Machine Learning (Python)**: Entrenamiento de 6 modelos con selección de algoritmo optimizada y exportación a ONNX.
2. **Servicio Backend de Inferencia (Rust - Axum)**: Inferencia de alto rendimiento y validación de tipos consumiendo directamente los modelos ONNX.
3. **Frontend Dashboard (React - Vite)**: Interfaz gráfica interactiva y amigable para el personal de la ONG.

---

## 📁 Estructura del Repositorio

Alineado al Anexo 10 del curso:

```
├── codigo/
│   ├── ia_models/              # Pipelines de ML (Python)
│   │   ├── aldimi_models/      # Módulo empaquetado (cargadores, preprocesamiento, train, evaluate)
│   │   ├── artifacts/          # Modelos exportados (.onnx, .pkl) y model_registry.json
│   │   └── outputs/            # Reportes de evaluación, gráficos y métricas (EDA y test)
│   │
│   ├── backend/                # API de inferencia en producción (Rust + ONNX Runtime)
│   │   ├── src/                # Endpoints, validación de esquemas y sesión ONNX
│   │   └── Dockerfile          # Empaquetado para despliegue en Render
│   │
│   └── frontend/               # Panel de control de usuario (React + Vite)
│       └── src/                # Dashboard de métricas y formulario dinámico de predicción
│
├── datos/                      # Carpeta de datasets sintéticos y diccionario de datos
│   └── DICCIONARIO_DATOS.md    # Definición de variables y targets
│
└── docs/                       # Documentación técnica, manual de usuario y diagramas
```

---

## 🧠 Los 6 Frentes de Modelado

| Frente | Target | Tipo de Tarea | Algoritmo Ganador | Métrica de Selección (Test) |
|---|---|---|---|---|
| **(a) Prioridad de Atención** | `nivel_prioridad_atencion` | Clasificación Multiclase | **XGBoost** | F1-macro = `0.7659` |
| **(b) Stock Crítico a 7 días** | `stock_critico_7d` | Clasificación Binaria | **Random Forest** | F1-macro = `0.8996` |
| **(c) Stock Crítico a 14 días** | `stock_critico_14d` | Clasificación Binaria | **Random Forest** | F1-macro = `0.8194` |
| **(d) Demanda de Consumo** | `consumo_real` | Regresión (Serie Temporal) | **XGBoost** | MAE = `4.26` unidades |
| **(e) Duración de Estadía** | `dias_hospedaje` | Regresión | **Random Forest** | MAE = `27.30` días |
| **(f) Proyección de Donaciones** | `valor_total_soles` | Regresión (Serie Temporal) | **Random Forest** | MAE = `4623.75` soles |

*Nota: Para prevenir fuga de información, en las variables explicativas de los frentes de stock crítico (b/c) se han excluido estrictamente las columnas `stock_fin`, `consumo_estimado_7d`, `consumo_estimado_14d` y `dias_cobertura`.*

---

## 🚀 Cómo Correr el Proyecto Localmente

### Paso 1: Clonar e instalar dependencias del Pipeline de ML
1. Accede a la carpeta del pipeline:
   ```bash
   cd codigo/ia_models
   ```
2. Crea un entorno virtual y actívalo:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instala los paquetes requeridos:
   ```bash
   pip install -r requirements.txt
   ```
4. Corre los tests para verificar que todo esté correcto:
   ```bash
   pytest
   ```

### Paso 2: Entrenar y Evaluar Modelos
1. Entrena todos los modelos comparando algoritmos (Decision Tree, Random Forest, XGBoost):
   ```bash
   python -m aldimi_models.train
   ```
2. Ejecuta la evaluación final y genera los reportes/gráficos en `outputs/evaluacion/`:
   ```bash
   python -m aldimi_models.evaluate
   ```

### Paso 3: Correr el Backend de Rust
1. No necesitas instalar ONNX Runtime a mano: la crate `ort` descarga sus binarios
   automáticamente durante `cargo build`.
2. Ve a la carpeta del backend:
   ```bash
   cd ../backend
   ```
3. Ejecuta la aplicación:
   ```bash
   cargo run
   ```
   La API estará lista en `http://localhost:8000`.

### Paso 4: Levantar el Frontend React
1. Ve a la carpeta del frontend:
   ```bash
   cd ../frontend
   ```
2. Instala los módulos de Node:
   ```bash
   npm install
   ```
3. Arranca el servidor de desarrollo de Vite:
   ```bash
   npm run dev
   ```
   Abre `http://localhost:5173` en tu navegador.

---

## ☁️ Guía de Despliegue en Producción (Gratuito)

Consulte los README correspondientes para conocer detalles minuciosos del despliegue en producción:
- **Backend Rust** en [Render (Web Service vía Docker)](./codigo/backend/README.md)
- **Frontend React** en [Vercel (Hobby plan)](./codigo/frontend/README.md)
- **Base de Datos** en [Supabase (Plan Free Postgres)](./codigo/backend/README.md) para el contrato del curso hermano de IA.
