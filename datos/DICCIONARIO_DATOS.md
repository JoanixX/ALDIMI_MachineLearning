# Diccionario de Datos - ALDIMI Predict (v2_ampliado)

Este documento detalla exhaustivamente las variables contenidas en los cinco datasets del proyecto, sus tipos de datos y su relevancia en el flujo de modelado predictivo.

---

## 1. Pacientes Riesgo Sintético (`pacientes_riesgo_sintetico.csv`)
*Contiene información sociodemográfica y clínica de 10,000 registros para determinar la prioridad de atención y estimar la estadía.*

| Variable | Tipo de Dato | Descripción | Rol en ML |
|---|---|---|---|
| `paciente_id` | Texto | Identificador único del paciente (ej. `PAC-0001`). | Excluido (ID) |
| `fecha_registro` | Fecha | Fecha en la que ingresa el paciente al albergue. | Variable temporal |
| `edad` | Entero | Edad del paciente en años. | Feature |
| `sexo` | Texto | Sexo biológico (`F` o `M`). | Feature |
| `region_origen` | Texto | Región de procedencia del paciente (ej. `Lima`, `Cusco`). | Feature |
| `diagnostico_general` | Texto | Categoría de la enfermedad (ej. `Leucemia`, `Linfoma`). | Feature |
| `estado_tratamiento` | Texto | Fase clínica actual (ej. `Inducción`, `Consolidación`, `Control`). | Feature |
| `num_controles_mes` | Entero | Cantidad de controles programados al mes. | Feature |
| `num_quimios_mes` | Entero | Cantidad de sesiones de quimioterapia al mes. | Feature |
| `hemoglobina_g_dl` | Real | Concentración de hemoglobina en sangre. | Feature |
| `neutrofilos` | Entero | Recuento absoluto de neutrófilos (indicador de inmunidad). | Feature |
| `plaquetas` | Entero | Recuento absoluto de plaquetas. | Feature |
| `temperatura_c` | Real | Temperatura corporal medida en grados Celsius. | Feature |
| `peso_kg` | Real | Peso corporal del paciente. | Feature |
| `imc` | Real | Índice de Masa Corporal. | Feature |
| `distancia_origen_km` | Real | Distancia desde el hogar del paciente al albergue. | Feature |
| `ingreso_familiar_mensual` | Real | Ingresos del hogar en Soles. | Feature |
| `acompanante_presente` | Binario | Indica si el menor cuenta con un cuidador (0: No, 1: Sí). | Feature |
| `seguro_salud` | Texto | Tipo de seguro médico (ej. `SIS`, `EsSalud`, `Particular`). | Feature |
| `alfabetizacion_digital` | Texto | Nivel de destreza tecnológica familiar (`Baja`, `Media`, `Alta`). | Feature |
| `requiere_apoyo_psicosocial` | Binario | Determinado al ingreso (0: No, 1: Sí). | Feature |
| `nivel_prioridad_atencion` | Categórico | Prioridad de asistencia (Bajo / Medio / Alto). | **Target (a)** |
| `dias_hospedaje` | Entero | Duración total de la estadía en el albergue. | **Target (e)** / Feature |

---

## 2. Consumo e Inventario Diario (`consumo_inventario_diario_sintetico.csv`)
*Registro temporal de consumo y stock de recursos a nivel diario (18,000 registros).*

| Variable | Tipo de Dato | Descripción | Rol en ML |
|---|---|---|---|
| `fecha` | Fecha | Día de registro (ej. `2024-01-01`). | Temporal |
| `item_id` | Texto | Identificador único del recurso (ej. `ITEM-001`). | Feature |
| `nombre_item` | Texto | Nombre legible del producto (ej. `Leche entera 1L`). | Excluido |
| `categoria` | Texto | Categoría del ítem (ej. `Alimentos`, `Higiene`, `Medicamentos`). | Feature |
| `unidad_medida` | Texto | Unidad del recurso (ej. `Litro`, `Unidad`, `Caja`). | Feature |
| `ocupacion_familias` | Entero | Cantidad de familias alojadas ese día en el albergue. | Feature |
| `stock_inicio` | Entero | Inventario inicial del día. | Feature |
| `ingreso_stock` | Entero | Nuevas unidades recibidas o compradas ese día. | Feature |
| `consumo_real` | Entero | Unidades consumidas ese día. | **Target (d)** / Feature |
| `stock_fin` | Entero | Inventario final del día. | Fuga de Stock (Excluido) |
| `consumo_estimado_7d` | Real | Estimación del consumo de los siguientes 7 días. | Fuga de Stock (Excluido) |
| `consumo_estimado_14d` | Real | Estimación del consumo de los siguientes 14 días. | Fuga de Stock (Excluido) |
| `dias_cobertura` | Real | Días restantes de stock antes del desabastecimiento. | Fuga de Stock (Excluido) |
| `stock_critico_7d` | Binario | Alerta de desabastecimiento a 7 días (0: Normal, 1: Crítico). | **Target (b)** |
| `stock_critico_14d` | Binario | Alerta de desabastecimiento a 14 días (0: Normal, 1: Crítico). | **Target (c)** |

---

## 3. Catálogo de Productos (`inventario_items.csv`)
*Datos maestros de los recursos (30 registros).*

| Variable | Tipo de Dato | Descripción |
|---|---|---|
| `item_id` | Texto | Identificador único del recurso. |
| `nombre_item` | Texto | Nombre comercial/legible. |
| `categoria` | Texto | Categoría del recurso. |
| `unidad_medida` | Texto | Unidad física. |
| `consumo_base_por_familia` | Real | Tasa estimada de consumo diario por familia alojada. |
| `stock_seguridad_7d` | Entero | Umbral mínimo para evitar desabastecimiento. |
| `stock_maximo_referencial` | Entero | Capacidad física máxima en el almacén. |
| `es_critico` | Binario | Clasificación estática del producto (0: No, 1: Sí). |

---

## 4. Donaciones Históricas (`donaciones_sinteticas.csv`)
*Registro mensual e individual de ingresos por donaciones (1,500 registros).*

| Variable | Tipo de Dato | Descripción | Rol en ML |
|---|---|---|---|
| `donacion_id` | Texto | Identificador único de la donación (ej. `DON-0001`). | Excluido |
| `fecha` | Fecha | Fecha de recepción de la donación. | Temporal |
| `tipo_donante` | Texto | Tipo de entidad (`Persona Natural`, `Corporativo`, `Recurrente`). | Excluido |
| `item_id` | Texto | Identificador del producto donado. | Excluido |
| `nombre_item` | Texto | Nombre del producto. | Excluido |
| `categoria` | Texto | Categoría del producto (usado para agrupar). | Feature |
| `unidad_medida` | Texto | Unidad física. | Excluido |
| `cantidad_donada` | Entero | Cantidad de unidades físicas donadas. | Excluido |
| `valor_estimado_soles` | Real | Valorización monetaria de la donación en Soles. | **Target (f)** / Feature |
| `campania` | Texto | Campaña asociada (ej. `Navidad`, `Friaje`, `Ninguna`). | Excluido |

---

## 5. Capturas IA / OCR (`capturas_ia_sinteticas.csv`)
*Contrato de datos de interoperabilidad con el curso hermano de Inteligencia Artificial (5,000 registros).*

| Variable | Tipo de Dato | Descripción |
|---|---|---|
| `captura_id` | Texto | Identificador único del análisis OCR (ej. `OCR-0001`). |
| `fecha_captura` | Fecha | Fecha y hora del análisis. |
| `paciente_id` | Texto | Enlace hacia el paciente correspondiente en la base de datos de ALDIMI. |
| `tipo_documento` | Texto | Tipo de archivo analizado (ej. `Receta Médica`, `DNI`, `Ficha Ingreso`). |
| `calidad_imagen` | Real | Métrica de calidad óptica de la captura (escala 0 a 1). |
| `confianza_ocr` | Real | Confianza del modelo de extracción de texto (escala 0 a 1). |
| `campos_extraidos` | JSON | Estructura JSON con los campos leídos (ej. medicinas, dosis, nombres). |
| `requiere_revision_manual`| Binario | Bandera para validación humana (0: No, 1: Sí). |
| `origen_captura` | Texto | Canal de subida (`Móvil`, `Escáner`, `Web`). |

---

## Nota de uso en el modelado (refactorización 2026)

- Estos CSV viven en `/datos` y son leídos únicamente por
  `codigo/ia_models/aldimi_models/data_loading.py`, que valida el esquema
  descrito en este diccionario.
- **Anti-fuga de información**: para los targets `stock_critico_7d` y
  `stock_critico_14d`, las columnas `stock_fin`, `consumo_estimado_7d`,
  `consumo_estimado_14d` y `dias_cobertura` derivan del propio target y están
  **prohibidas como features** (ver `config.STOCK_LEAKAGE_FEATURES` y los tests
  de `tests/test_no_leakage.py`).
- `capturas_ia_sinteticas.csv` es el ejemplo del contrato de datos real con el
  curso hermano de IA (tabla `capturas_ia` en Supabase); su esquema SQL sugerido
  está en `codigo/backend/README.md`.
