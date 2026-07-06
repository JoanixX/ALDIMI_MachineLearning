# Diccionario de Datos - ALDIMI Predict Dataset Sintético v2 Ampliado

## Nota ética
Estos datos son completamente sintéticos y fueron creados solo para fines académicos. No contienen información real de pacientes ni de ALDIMI.

## Archivos incluidos
- pacientes_riesgo_sintetico.csv: datos clínicos y sociales para clasificación de prioridad.
- consumo_inventario_diario_sintetico.csv: consumo, stock, ocupación y alertas a 7/14 días.
- inventario_items.csv: catálogo de productos.
- donaciones_sinteticas.csv: donaciones por fecha, tipo de donante e ítem.
- capturas_ia_sinteticas.csv: simulación de datos capturados por módulo IA/OCR.

## Targets sugeridos
- nivel_prioridad_atencion: clasificación multiclase Bajo/Medio/Alto.
- consumo_real: regresión o series temporales.
- stock_critico_7d: clasificación binaria.
- stock_critico_14d: clasificación binaria.

## Variables útiles para EDA
Pacientes:
edad, dias_hospedaje, num_controles_mes, num_quimios_mes, hemoglobina_g_dl, neutrofilos, plaquetas, temperatura_c, distancia_origen_km, ingreso_familiar_mensual, nivel_prioridad_atencion.

Inventario:
fecha, item_id, categoria, ocupacion_familias, stock_inicio, ingreso_stock, consumo_real, stock_fin, consumo_estimado_7d, consumo_estimado_14d, dias_cobertura, stock_critico_7d, stock_critico_14d.

OCR:
tipo_documento, calidad_imagen, confianza_ocr, requiere_revision_manual.
