"""Analisis exploratorio de datos (EDA) del dataset sintetico de ALDIMI.

Migracion integra del notebook EDA_ALDIMI_Predict_v2_ampliado.ipynb a modulo
ejecutable. Genera tablas CSV y graficos PNG en ``outputs/eda/``.

Uso:
    python -m aldimi_models.eda
"""

from __future__ import annotations

import pandas as pd

from .common import plots
from .common.metrics import save_dataframe
from .config import OUTPUTS_DIR, PRIORIDAD_CLASSES
from .data_loading import load_all

EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"

_VARIABLES_CLINICAS = [
    "edad", "dias_hospedaje", "num_controles_mes", "num_quimios_mes",
    "hemoglobina_g_dl", "neutrofilos", "plaquetas", "temperatura_c",
    "peso_kg", "imc",
]
_BOXPLOT_VARIABLES = [
    "hemoglobina_g_dl", "neutrofilos", "plaquetas", "temperatura_c",
    "dias_hospedaje", "ingreso_familiar_mensual",
]


def resumen_general(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dimensiones, duplicados y nulos de cada dataset."""
    filas = [
        {
            "dataset": nombre,
            "filas": df.shape[0],
            "columnas": df.shape[1],
            "duplicados": int(df.duplicated().sum()),
            "valores_nulos_totales": int(df.isna().sum().sum()),
        }
        for nombre, df in datasets.items()
    ]
    return pd.DataFrame(filas)


def resumen_nulos(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Nulos por columna de todos los datasets, ordenados por porcentaje."""
    tablas = []
    for nombre, df in datasets.items():
        tablas.append(
            pd.DataFrame(
                {
                    "dataset": nombre,
                    "columna": df.columns,
                    "nulos": df.isna().sum().values,
                    "porcentaje_nulos": (df.isna().mean().values * 100).round(2),
                }
            )
        )
    return pd.concat(tablas, ignore_index=True).sort_values(
        "porcentaje_nulos", ascending=False
    )


def detectar_outliers_iqr(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Conteo de outliers por el metodo IQR (1.5 * rango intercuartil)."""
    resultados = []
    for col in columnas:
        serie = df[col].dropna()
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        limite_inf, limite_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        resultados.append(
            {
                "variable": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "limite_inferior": limite_inf,
                "limite_superior": limite_sup,
                "cantidad_outliers": int(((df[col] < limite_inf) | (df[col] > limite_sup)).sum()),
            }
        )
    return pd.DataFrame(resultados)


def _eda_pacientes(pacientes: pd.DataFrame) -> None:
    out = EDA_OUTPUT_DIR
    save_dataframe(
        pacientes.describe(include="all").T, out / "estadisticas_descriptivas_pacientes.csv", index=True
    )

    dist = pacientes["nivel_prioridad_atencion"].value_counts().reset_index()
    dist.columns = ["nivel_prioridad_atencion", "cantidad"]
    dist["porcentaje"] = (dist["cantidad"] / len(pacientes) * 100).round(2)
    save_dataframe(dist, out / "distribucion_prioridad_pacientes.csv")
    plots.bar_chart(
        dist["nivel_prioridad_atencion"], dist["cantidad"],
        "Distribucion del nivel de prioridad de atencion",
        "Nivel de prioridad", "Cantidad de registros",
        out / "distribucion_prioridad_atencion.png", rotate_xticks=False,
    )

    for group_col in ("estado_tratamiento", "diagnostico_general"):
        tabla = pd.crosstab(pacientes[group_col], pacientes["nivel_prioridad_atencion"])
        save_dataframe(tabla, out / f"prioridad_por_{group_col}.csv", index=True)

    for col in _VARIABLES_CLINICAS:
        plots.histogram(pacientes[col], f"Distribucion de {col}", col, out / f"hist_{col}.png")
    for col in _BOXPLOT_VARIABLES:
        plots.boxplot_by_group(
            pacientes, col, "nivel_prioridad_atencion", PRIORIDAD_CLASSES,
            f"{col} por nivel de prioridad", out / f"boxplot_{col}_por_prioridad.png",
        )

    corr = pacientes.select_dtypes(include=["number"]).corr()
    save_dataframe(corr, out / "correlacion_pacientes.csv", index=True)
    plots.correlation_heatmap(corr, "Matriz de correlacion - Pacientes", out / "matriz_correlacion_pacientes.png")

    outliers = detectar_outliers_iqr(pacientes, _VARIABLES_CLINICAS)
    save_dataframe(outliers, out / "outliers_iqr_pacientes.csv")


def _eda_consumo(consumo: pd.DataFrame) -> None:
    out = EDA_OUTPUT_DIR
    save_dataframe(
        consumo.describe(include="all").T, out / "estadisticas_descriptivas_consumo.csv", index=True
    )

    stock_resumen = pd.DataFrame(
        {
            "target": ["stock_critico_7d", "stock_critico_14d"],
            "cantidad_alertas": [
                int(consumo["stock_critico_7d"].sum()),
                int(consumo["stock_critico_14d"].sum()),
            ],
            "porcentaje_alertas": [
                round(consumo["stock_critico_7d"].mean() * 100, 2),
                round(consumo["stock_critico_14d"].mean() * 100, 2),
            ],
        }
    )
    save_dataframe(stock_resumen, out / "resumen_stock_critico.csv")
    plots.bar_chart(
        stock_resumen["target"], stock_resumen["porcentaje_alertas"],
        "Porcentaje de alertas de stock critico", "Target", "% de alertas",
        out / "porcentaje_alertas_stock_critico.png", rotate_xticks=False,
    )

    por_categoria = consumo.groupby("categoria").agg(
        consumo_promedio=("consumo_real", "mean"),
        consumo_total=("consumo_real", "sum"),
        stock_promedio=("stock_fin", "mean"),
        dias_cobertura_promedio=("dias_cobertura", "mean"),
        alerta_7d_rate=("stock_critico_7d", "mean"),
        alerta_14d_rate=("stock_critico_14d", "mean"),
    ).reset_index()
    por_categoria["alerta_7d_rate"] = (por_categoria["alerta_7d_rate"] * 100).round(2)
    por_categoria["alerta_14d_rate"] = (por_categoria["alerta_14d_rate"] * 100).round(2)
    save_dataframe(por_categoria, out / "consumo_por_categoria.csv")
    plots.bar_chart(
        por_categoria["categoria"], por_categoria["consumo_total"],
        "Consumo total por categoria", "Categoria", "Consumo total",
        out / "consumo_total_por_categoria.png",
    )

    top_items = (
        consumo.groupby(["item_id", "nombre_item", "categoria"])
        .agg(
            consumo_total=("consumo_real", "sum"),
            consumo_promedio=("consumo_real", "mean"),
            alertas_7d=("stock_critico_7d", "sum"),
            alertas_14d=("stock_critico_14d", "sum"),
        )
        .reset_index()
        .sort_values("consumo_total", ascending=False)
        .head(10)
    )
    save_dataframe(top_items, out / "top_10_items_consumo.csv")
    plots.bar_chart(
        top_items["nombre_item"], top_items["consumo_total"],
        "Top 10 productos con mayor consumo total", "Producto", "Consumo total",
        out / "top_10_productos_consumo.png",
    )

    diario = consumo.groupby("fecha").agg(
        ocupacion_familias=("ocupacion_familias", "mean"),
        consumo_total_diario=("consumo_real", "sum"),
        stock_total_fin=("stock_fin", "sum"),
        alertas_7d=("stock_critico_7d", "sum"),
        alertas_14d=("stock_critico_14d", "sum"),
    ).reset_index()
    save_dataframe(diario, out / "consumo_diario_agregado.csv")
    plots.line_chart(
        diario["fecha"], diario["ocupacion_familias"],
        "Evolucion de la ocupacion de familias", "Fecha", "Ocupacion promedio",
        out / "evolucion_ocupacion_familias.png",
    )
    plots.line_chart(
        diario["fecha"], diario["consumo_total_diario"],
        "Evolucion del consumo total diario", "Fecha", "Consumo total diario",
        out / "evolucion_consumo_total_diario.png",
    )
    plots.scatter_chart(
        diario["ocupacion_familias"], diario["consumo_total_diario"],
        "Relacion entre ocupacion y consumo total diario",
        "Ocupacion de familias", "Consumo total diario",
        out / "scatter_ocupacion_vs_consumo.png",
    )

    corr = consumo.select_dtypes(include=["number"]).corr()
    save_dataframe(corr, out / "correlacion_consumo.csv", index=True)
    plots.correlation_heatmap(
        corr, "Matriz de correlacion - Inventario y consumo", out / "matriz_correlacion_consumo.png"
    )


def _eda_donaciones(donaciones: pd.DataFrame) -> None:
    out = EDA_OUTPUT_DIR
    save_dataframe(
        donaciones.describe(include="all").T, out / "estadisticas_descriptivas_donaciones.csv", index=True
    )
    por_tipo = (
        donaciones.groupby("tipo_donante")
        .agg(
            cantidad_donaciones=("donacion_id", "count"),
            cantidad_total_donada=("cantidad_donada", "sum"),
            valor_total_soles=("valor_estimado_soles", "sum"),
            valor_promedio_soles=("valor_estimado_soles", "mean"),
        )
        .reset_index()
        .sort_values("valor_total_soles", ascending=False)
    )
    save_dataframe(por_tipo, out / "donaciones_por_tipo_donante.csv")
    plots.bar_chart(
        por_tipo["tipo_donante"], por_tipo["valor_total_soles"],
        "Valor total estimado de donaciones por tipo de donante",
        "Tipo de donante", "Valor estimado total S/",
        out / "valor_donaciones_por_tipo_donante.png",
    )

    por_categoria = (
        donaciones.groupby("categoria")
        .agg(
            cantidad_donaciones=("donacion_id", "count"),
            cantidad_total_donada=("cantidad_donada", "sum"),
            valor_total_soles=("valor_estimado_soles", "sum"),
        )
        .reset_index()
        .sort_values("valor_total_soles", ascending=False)
    )
    save_dataframe(por_categoria, out / "donaciones_por_categoria.csv")
    plots.bar_chart(
        por_categoria["categoria"], por_categoria["valor_total_soles"],
        "Valor total de donaciones por categoria", "Categoria", "Valor estimado total S/",
        out / "valor_donaciones_por_categoria.png",
    )


def _eda_capturas_ia(capturas: pd.DataFrame) -> None:
    """Contrato de datos con el curso hermano de IA (capturas OCR)."""
    out = EDA_OUTPUT_DIR
    save_dataframe(
        capturas.describe(include="all").T, out / "estadisticas_descriptivas_capturas_ia.csv", index=True
    )
    confianza = (
        capturas.groupby("tipo_documento")["confianza_ocr"].mean().sort_values(ascending=False)
    )
    plots.bar_chart(
        confianza.index, confianza.values,
        "Confianza OCR promedio por tipo de documento",
        "Tipo de documento", "Confianza OCR promedio",
        out / "confianza_ocr_por_tipo_documento.png",
    )


def main() -> None:
    datasets = load_all()
    print(f"Datasets cargados: {list(datasets)}")

    save_dataframe(resumen_general(datasets), EDA_OUTPUT_DIR / "resumen_general_datasets.csv")
    save_dataframe(resumen_nulos(datasets), EDA_OUTPUT_DIR / "resumen_nulos.csv")

    _eda_pacientes(datasets["pacientes"])
    _eda_consumo(datasets["consumo"])
    _eda_donaciones(datasets["donaciones"])
    _eda_capturas_ia(datasets["capturas_ia"])
    print(f"EDA completo en: {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
