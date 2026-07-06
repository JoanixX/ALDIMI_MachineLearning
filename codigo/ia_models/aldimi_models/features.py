"""Construccion de features por frente de modelado.

Cada funcion devuelve un ``FeatureSet`` autocontenido: matriz X, target y,
listas de columnas numericas/categoricas y (si aplica) la fecha de cada fila
para divisiones temporales.

Regla anti-fuga: para los targets ``stock_critico_7d``/``stock_critico_14d``
se excluyen explicitamente las variables derivadas del propio target
(``stock_fin``, ``consumo_estimado_7d``, ``consumo_estimado_14d``,
``dias_cobertura``) definidas en ``config.STOCK_LEAKAGE_FEATURES``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import STOCK_LEAKAGE_FEATURES


@dataclass
class FeatureSet:
    """Datos listos para entrenar un modelo."""

    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    dates: pd.Series | None = None
    class_labels: list[str] | None = None

    def __post_init__(self) -> None:
        declared = set(self.numeric_features) | set(self.categorical_features)
        actual = set(self.X.columns)
        if declared != actual:
            raise ValueError(
                f"Features declaradas y columnas de X no coinciden: "
                f"faltan {sorted(actual - declared)}, sobran {sorted(declared - actual)}"
            )
        # Los nulos categoricos se imputan aqui (y no con SimpleImputer en el
        # pipeline) porque skl2onnx no convierte imputers sobre strings.
        for col in self.categorical_features:
            self.X[col] = self.X[col].fillna("desconocido").astype(str)


PRIORIDAD_FEATURES_NUM = [
    "edad", "dias_hospedaje", "num_controles_mes", "num_quimios_mes",
    "hemoglobina_g_dl", "neutrofilos", "plaquetas", "temperatura_c",
    "peso_kg", "imc", "distancia_origen_km", "ingreso_familiar_mensual",
    "acompanante_presente", "requiere_apoyo_psicosocial",
]
PRIORIDAD_FEATURES_CAT = [
    "sexo", "region_origen", "diagnostico_general", "estado_tratamiento",
    "seguro_salud", "alfabetizacion_digital",
]


def prioridad_features(pacientes: pd.DataFrame) -> FeatureSet:
    """(a) Clasificacion multiclase del nivel de prioridad de atencion."""
    from .config import PRIORIDAD_CLASSES

    target = "nivel_prioridad_atencion"
    df = pacientes.dropna(subset=[target])
    y = df[target].map({label: i for i, label in enumerate(PRIORIDAD_CLASSES)})
    if y.isna().any():
        raise ValueError(f"Valores de {target} fuera de {PRIORIDAD_CLASSES}")
    X = df[PRIORIDAD_FEATURES_NUM + PRIORIDAD_FEATURES_CAT].copy()
    return FeatureSet(
        X=X,
        y=y.astype(int),
        numeric_features=PRIORIDAD_FEATURES_NUM,
        categorical_features=PRIORIDAD_FEATURES_CAT,
        class_labels=PRIORIDAD_CLASSES,
    )


STOCK_FEATURES_NUM = ["ocupacion_familias", "stock_inicio", "ingreso_stock", "consumo_real"]
STOCK_FEATURES_CAT = ["categoria", "unidad_medida", "item_id"]


def stock_critico_features(consumo: pd.DataFrame, horizon_days: int) -> FeatureSet:
    """(b)/(c) Clasificacion binaria de stock critico a 7 o 14 dias."""
    if horizon_days not in (7, 14):
        raise ValueError("horizon_days debe ser 7 o 14")
    target = f"stock_critico_{horizon_days}d"
    features = STOCK_FEATURES_NUM + STOCK_FEATURES_CAT
    leaked = [col for col in features if col in STOCK_LEAKAGE_FEATURES]
    if leaked:
        raise ValueError(f"Fuga de informacion: {leaked} no pueden ser features de {target}")
    df = consumo.dropna(subset=[target])
    return FeatureSet(
        X=df[features].copy(),
        y=df[target].astype(int),
        numeric_features=STOCK_FEATURES_NUM,
        categorical_features=STOCK_FEATURES_CAT,
        dates=df["fecha"],
        class_labels=["normal", "critico"],
    )


DEMANDA_LAGS = [1, 7, 14]
DEMANDA_FEATURES_CAT = ["categoria", "item_id"]


def demanda_features(consumo: pd.DataFrame) -> FeatureSet:
    """(d) Regresion de demanda: consumo_real diario por item con lags.

    Solo usa informacion disponible antes del dia predicho (lags y calendario),
    nunca columnas del mismo dia derivadas del stock.
    """
    df = consumo.sort_values(["item_id", "fecha"]).copy()
    grouped = df.groupby("item_id")["consumo_real"]
    for lag in DEMANDA_LAGS:
        df[f"consumo_lag_{lag}"] = grouped.shift(lag)
    df["consumo_media_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    df["ocupacion_lag_1"] = df.groupby("item_id")["ocupacion_familias"].shift(1)

    numeric = [f"consumo_lag_{lag}" for lag in DEMANDA_LAGS] + [
        "consumo_media_7", "dia_semana", "mes", "ocupacion_lag_1",
    ]
    df = df.dropna(subset=numeric + ["consumo_real"])
    return FeatureSet(
        X=df[numeric + DEMANDA_FEATURES_CAT].copy(),
        y=df["consumo_real"].astype(float),
        numeric_features=numeric,
        categorical_features=DEMANDA_FEATURES_CAT,
        dates=df["fecha"],
    )


ESTADIA_FEATURES_NUM = [
    "edad", "num_controles_mes", "num_quimios_mes", "hemoglobina_g_dl",
    "neutrofilos", "plaquetas", "temperatura_c", "peso_kg", "imc",
    "distancia_origen_km", "ingreso_familiar_mensual",
    "acompanante_presente", "requiere_apoyo_psicosocial",
]
ESTADIA_FEATURES_CAT = [
    "sexo", "region_origen", "diagnostico_general", "estado_tratamiento",
    "seguro_salud", "alfabetizacion_digital",
]


def estadia_features(pacientes: pd.DataFrame) -> FeatureSet:
    """(e) Regresion de la duracion de hospedaje (dias_hospedaje).

    No usa nivel_prioridad_atencion como feature: en produccion la estadia se
    estima al ingreso, antes de conocer la clasificacion de prioridad.
    """
    target = "dias_hospedaje"
    df = pacientes.dropna(subset=[target])
    return FeatureSet(
        X=df[ESTADIA_FEATURES_NUM + ESTADIA_FEATURES_CAT].copy(),
        y=df[target].astype(float),
        numeric_features=ESTADIA_FEATURES_NUM,
        categorical_features=ESTADIA_FEATURES_CAT,
    )


DONACIONES_LAGS = [1, 2, 3]


def donaciones_features(donaciones: pd.DataFrame) -> FeatureSet:
    """(f) Proyeccion mensual del valor donado (S/) por categoria.

    Serie mensual por categoria con lags de 1-3 meses y mes calendario.
    """
    df = donaciones.copy()
    df["mes_periodo"] = df["fecha"].dt.to_period("M")
    mensual = (
        df.groupby(["mes_periodo", "categoria"], observed=True)
        .agg(
            valor_total_soles=("valor_estimado_soles", "sum"),
            num_donaciones=("donacion_id", "count"),
        )
        .reset_index()
        .sort_values(["categoria", "mes_periodo"])
    )
    grouped = mensual.groupby("categoria")["valor_total_soles"]
    for lag in DONACIONES_LAGS:
        mensual[f"valor_lag_{lag}"] = grouped.shift(lag)
    mensual["donaciones_lag_1"] = mensual.groupby("categoria")["num_donaciones"].shift(1)
    mensual["mes"] = mensual["mes_periodo"].dt.month

    numeric = [f"valor_lag_{lag}" for lag in DONACIONES_LAGS] + ["donaciones_lag_1", "mes"]
    mensual = mensual.dropna(subset=numeric)
    if mensual.empty:
        raise ValueError("No hay suficientes meses de historia de donaciones para los lags")
    return FeatureSet(
        X=mensual[numeric + ["categoria"]].copy(),
        y=mensual["valor_total_soles"].astype(float),
        numeric_features=numeric,
        categorical_features=["categoria"],
        dates=mensual["mes_periodo"].dt.to_timestamp(),
    )
