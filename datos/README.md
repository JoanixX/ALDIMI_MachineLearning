# ALDIMI Predict - Carpeta de Datos

Esta carpeta contiene los archivos en formato CSV que representan el "contrato de datos" y los insumos históricos para entrenar los modelos predictivos de la ONG ALDIMI.

## 📋 Contenido de la carpeta

1. [DICCIONARIO_DATOS.md](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/datos/DICCIONARIO_DATOS.md): Detalle técnico exhaustivo de cada columna y su significado.
2. [pacientes_riesgo_sintetico.csv](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/datos/pacientes_riesgo_sintetico.csv): Registro clínico de 12,000 pacientes ficticios para clasificación de prioridad y estadía.
3. [consumo_inventario_diario_sintetico.csv](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/datos/consumo_inventario_diario_sintetico.csv): Registro diario de 21,900 movimientos (30 items x 730 dias) de stock y consumo de items para series temporales y alertas.
4. [inventario_items.csv](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/datos/inventario_items.csv): Catálogo maestro de 30 productos con su consumo base y stock de seguridad.
5. [donaciones_sinteticas.csv](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/datos/donaciones_sinteticas.csv): Registro de ~4,300 donaciones (36 meses) recibidas para proyecciones financieras mensuales.
6. [capturas_ia_sinteticas.csv](file:///c:/Users/practicante.coe03/Desktop/Clases/Machine%20Learning/TF/ALDIMI_MachineLearning/datos/capturas_ia_sinteticas.csv): 6,000 registros del OCR del curso hermano de Inteligencia Artificial.

*Nota ética: Todos estos datos son sintéticos, diseñados con fines estrictamente pedagógicos e ilustrativos para el Trabajo Final del curso de Machine Learning.*

> Estos CSV son la **v3**, generada de forma reproducible con `python -m aldimi_models.datagen --seed 42`; ver la seccion "Generacion del dataset v3" del diccionario.
