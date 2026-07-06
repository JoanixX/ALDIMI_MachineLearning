# ALDIMI Predict - Backend de Inferencia (Rust)

Servicio de alto rendimiento construido en **Rust** usando **Axum** para exponer las predicciones de los modelos de Machine Learning entrenados en Python. Al usar **ONNX Runtime (crate `ort`)**, el servicio es capaz de realizar predicciones rápidas y concurrentes sin necesidad de levantar un intérprete de Python en producción.

---

## 🛠️ Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Chequeo de salud del servicio (retorna `{"status": "ok"}`). |
| `GET` | `/models` | Lista los esquemas de entrada (features) y métricas reales de los 6 modelos. |
| `POST` | `/predict/{model_name}` | Realiza la predicción puntual o batch para el modelo especificado. |

### Contrato de Inferencia (`POST /predict/{model_name}`)

El body debe ser un objeto JSON (caso individual) o una lista de objetos JSON (lote) que provea las variables declaradas por el modelo en `/models`.

**Ejemplo de llamada a `/predict/prioridad_atencion`**:
```bash
curl -X POST http://localhost:8000/predict/prioridad_atencion \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 12,
    "sexo": "F",
    "region_origen": "Arequipa",
    "diagnostico_general": "Leucemia linfoblástica aguda",
    "estado_tratamiento": "Inducción",
    "dias_hospedaje": 5,
    "num_controles_mes": 2,
    "num_quimios_mes": 1,
    "hemoglobina_g_dl": 12.1,
    "neutrofilos": 1200,
    "plaquetas": 150000,
    "temperatura_c": 36.8,
    "peso_kg": 40.0,
    "imc": 18.0,
    "distancia_origen_km": 200,
    "ingreso_familiar_mensual": 800,
    "acompanante_presente": 1,
    "seguro_salud": "SIS",
    "alfabetizacion_digital": "Media",
    "requiere_apoyo_psicosocial": 1
  }'
```

**Respuesta**:
```json
{
  "model": "prioridad_atencion",
  "predictions": [
    {
      "prediccion": "Bajo",
      "clase_indice": 0,
      "probabilidades": {
        "Alto": 0.052,
        "Bajo": 0.812,
        "Medio": 0.136
      }
    }
  ]
}
```

---

## 🗄️ Contrato de Datos (Supabase Postgres)

Para mantener la interoperabilidad con el curso hermano de Inteligencia Artificial (procesamiento de imágenes médicas y OCR), se recomienda centralizar los datos en una base de datos Postgres gratuita en **Supabase**:

1. **Tabla `capturas_ia`**: Mapea el contrato de datos representado en [capturas_ia_sinteticas.csv](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/datos/capturas_ia_sinteticas.csv).
2. **Esquema de SQL sugerido**:
   ```sql
   CREATE TABLE capturas_ia (
       captura_id VARCHAR(50) PRIMARY KEY,
       fecha_captura TIMESTAMP NOT NULL,
       paciente_id VARCHAR(50) NOT NULL,
       tipo_documento VARCHAR(50) NOT NULL,
       calidad_imagen FLOAT NOT NULL,
       confianza_ocr FLOAT NOT NULL,
       campos_extraidos JSONB,
       requiere_revision_manual BOOLEAN NOT NULL,
       origen_captura VARCHAR(50)
   );
   ```
Esta base de datos Supabase actúa como el "contrato de datos real" donde el curso de IA escribe los resultados de OCR y el curso de ML lee para alimentar los modelos predictivos o mostrar el historial en el dashboard.

---

## ☁️ Guía de Despliegue en Render (Web Service Gratuito)

Render ofrece planes de servicio web gratuitos siempre que el despliegue se realice vía **Docker**:

1. Crea una cuenta gratuita en [Render.com](https://render.com/).
2. Conecta tu repositorio de GitHub.
3. Crea un nuevo **Web Service**.
4. Configura los siguientes campos:
   - **Repository URL**: URL de tu repositorio de GitHub.
   - **Branch**: `main` o `develop`.
   - **Region**: Selecciona la más cercana (por ejemplo, `Oregon` o `Ohio`).
   - **Runtime**: `Docker`.
   - **Docker Command**: Dejar en blanco (se usa el `CMD` del Dockerfile).
   - **Plan**: `Free`.
5. En la sección **Environment Variables**, añade:
   - `PORT`: `8000` (Render expone la app externamente, pero Axum debe escuchar en este puerto).
   - `ALDIMI_ARTIFACTS_DIR`: `/app/artifacts` (ubicación interna de los modelos ONNX dentro de la imagen Docker).

Render compilará el código de Rust en la nube y levantará el contenedor de Axum automáticamente de forma gratuita.
