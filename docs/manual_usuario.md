# Manual de Usuario - ALDIMI Predict

Bienvenido al manual de usuario de **ALDIMI Predict**. Este documento guiará al personal del albergue y a los administradores de la ONG en el uso del dashboard para estimar necesidades y optimizar la atención de pacientes.

---

## 🖥️ Interfaz del Dashboard

El dashboard se compone de dos áreas principales pensadas para facilitar la toma de decisiones rápidas:

### 1. Panel de Modelos Desplegados (Resumen y Métricas)
- **Tabla de Modelos**: Muestra los algoritmos seleccionados en producción para cada uno de los 6 frentes.
- **Gráficos de Barras Horizontales**: 
  - Para los frentes de **Clasificación** (Prioridad y alertas de stock), visualiza el valor de **F1-macro** (escala de 0 a 1, donde más cercano a 1 representa un modelo más robusto).
  - Para los frentes de **Regresión** (Demanda diaria, estadía, valor de donaciones), visualiza el **Error Absoluto Medio (MAE)**. Esto ayuda a comprender la precisión del modelo en unidades físicas (ej. unidades de alimentos, días u soles).

### 2. Formulario de Predicción Interactiva
Este formulario le permite ingresar información en tiempo real para generar predicciones instantáneas sin requerir conocimientos técnicos de programación.

---

## 🔮 Cómo Realizar una Predicción Paso a Paso

1. **Seleccionar el Modelo**:
   En la lista desplegable superior del bloque de predicciones, elija el modelo que desea evaluar:
   - *Prioridad de Atención del Paciente*: Para clasificar si un paciente ingresante requiere prioridad "Baja", "Media" o "Alta".
   - *Alerta de Stock Crítico (7 o 14 días)*: Para evaluar si un ítem del almacén corre riesgo de desabastecimiento.
   - *Consumo Diario de un Ítem (Demanda)*: Para estimar cuántas unidades físicas se consumirán el día de mañana.
   - *Duración de Hospedaje (Estadía)*: Para prever cuántos días de albergue requerirá un paciente a su llegada.
   - *Proyección de Donaciones por Categoría*: Para prever el valor total en Soles de las donaciones de una categoría específica en el mes en curso.

2. **Ingresar las Variables**:
   El formulario cambiará automáticamente mostrando únicamente las variables que requiere el modelo seleccionado.
   - **Campos Numéricos**: Ingrese valores numéricos válidos (ej. Edad: `10`, Neutrófilos: `1500`).
   - **Campos Categóricos**: Seleccione una opción de las listas desplegables (ej. Diagnóstico: `Linfoma`, Seguro: `SIS`). Estas opciones son enviadas por el backend y se actualizan solas.

3. **Ejecutar Inferencia**:
   Haga clic en el botón **Predecir**. El backend en Rust resolverá la predicción sobre el modelo ONNX en milisegundos.

4. **Interpretar Resultados**:
   - **En Clasificación**: Muestra la clase ganadora en texto grande (ej. `Medio`) y un gráfico de barras que detalla las probabilidades calculadas para cada categoría (ej. Bajo: 12%, Medio: 78%, Alto: 10%).
   - **En Regresión**: Muestra un valor numérico estimado directamente en su unidad física original (ej. `12.50` días de hospedaje o `4500.00` Soles proyectados).

---

## ⚠️ Diagnóstico de Errores Frecuentes

- **"Cargando modelos..." o "No se pudo conectar con el backend"**:
  Esto ocurre si la API en Rust no está iniciada o si la URL de conexión (`VITE_API_URL`) está mal configurada. Verifique que el servicio web de Render esté activo.
- **"El campo X debe ser un número"**:
  Asegúrese de no ingresar letras en los campos de tipo numérico.
