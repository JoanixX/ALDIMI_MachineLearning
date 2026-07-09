# Datos reales

Estos CSV se descargan con `python -m aldimi_models.real_data` (no editar a mano).

| Archivo | Fuente real | Uso |
|---|---|---|
| `ed_triage_iran.csv` | [Mendeley Data doi:10.17632/vhzyyktrz5.1](https://data.mendeley.com/datasets/vhzyyktrz5/1) — 143 582 admisiones reales de urgencias de un hospital universitario (CC-BY). Edad, motivo de consulta (ICD), dolor, estados crítico/estupor, signos vitales y **grado de triaje 1-5 asignado por enfermería**. | Evaluado para el modelo (a) `prioridad_atencion` (ESI 1-2 → Alto, 3 → Medio, 4-5 → Bajo). **Descartado**: F1-macro 0.993 > 0.95 (el triaje es casi determinista de sus insumos). Ver informe. |
| `sparcs_pediatrico_oncologico.csv` | [NY SPARCS De-Identified](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/tg3i-cinn) 2021-2023 vía API Socrata — 10 512 altas hospitalarias reales de pacientes de 0-17 años con diagnóstico oncológico (CCSR `NEO*` o MDC 17). | Evaluado para el modelo (e) `duracion_estadia` (target `length_of_stay` real, censurado en 120+). **Descartado**: R² 0.351 < 0.85. Ver informe. |

## Por qué el resto sigue siendo sintético

Se buscaron datasets públicos para los demás frentes sin éxito (julio 2026):

- **Inventario/stock crítico y demanda por ítem**: no existe ningún dataset público
  de inventario diario de un albergue/banco de alimentos con stock por ítem
  (Feeding America publica solo agregados anuales; Mesa AZ publica eventos puntuales
  COVID; los datasets de retail requieren cuenta de Kaggle y no representan el dominio).
- **Donaciones fechadas por categoría**: el dataset abierto más cercano
  (NYC "Donations Received by City Agencies") solo tiene granularidad **anual**;
  DonorsChoose requiere autenticación de Kaggle.
- **Capturas OCR**: es el contrato interno con el curso hermano de IA; no existe
  equivalente público.

Para esos frentes se usa el generador sintético documentado
(`python -m aldimi_models.datagen`), como última instancia.
