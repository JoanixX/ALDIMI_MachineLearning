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

**Ejemplo de llamada a `/predict/prioridad_atencion`** (ejecutado realmente contra el servicio):
```bash
curl -X POST http://localhost:8000/predict/prioridad_atencion \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 8,
    "sexo": "F",
    "region_origen": "Loreto",
    "diagnostico_general": "Leucemia linfoblástica aguda",
    "estado_tratamiento": "En tratamiento",
    "dias_hospedaje": 30,
    "num_controles_mes": 4,
    "num_quimios_mes": 2,
    "hemoglobina_g_dl": 9.1,
    "neutrofilos": 800,
    "plaquetas": 150000,
    "temperatura_c": 38.2,
    "peso_kg": 25.0,
    "imc": 15.5,
    "distancia_origen_km": 400,
    "ingreso_familiar_mensual": 900,
    "acompanante_presente": 1,
    "seguro_salud": "SIS",
    "alfabetizacion_digital": "Baja",
    "requiere_apoyo_psicosocial": 1
  }'
```

**Respuesta real del servicio**:
```json
{
  "model": "prioridad_atencion",
  "predictions": [
    {
      "clase_indice": 2,
      "prediccion": "Alto",
      "probabilidades": {
        "Alto": 0.6346927881240845,
        "Bajo": 0.0003548664681147784,
        "Medio": 0.3649523854255676
      }
    }
  ]
}
```

Un error de esquema responde `422` con detalle, por ejemplo:
`{"error":"entrada invalida: 'edad' debe ser numerico, se recibio: \"mal\""}`.

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
       calidad_imagen VARCHAR(20) NOT NULL, -- 'Alta' | 'Media' | 'Baja' segun el CSV de ejemplo
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
