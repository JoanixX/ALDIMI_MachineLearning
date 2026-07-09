"""Construccion de features por frente de modelado, con feature engineering.

Cada funcion devuelve un ``FeatureSet`` autocontenido: matriz X, target y,
listas de columnas numericas/categoricas y (si aplica) la fecha de cada fila
para divisiones temporales.

Regla anti-fuga: para los targets ``stock_critico_7d``/``stock_critico_14d``
se excluyen explicitamente las variables derivadas del propio target
(``stock_fin``, ``consumo_estimado_7d``, ``consumo_estimado_14d``,
``dias_cobertura``) definidas en ``config.STOCK_LEAKAGE_FEATURES``. Las
features derivadas (rolling, ratios) usan solo informacion disponible en el
momento de la prediccion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import STOCK_LEAKAGE_FEATURES
from .data_loading import load_items


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


def _flag(series: pd.Series, condition: pd.Series) -> pd.Series:
    """Indicador 0/1 que conserva NaN cuando el insumo es NaN (lo imputa el pipeline)."""
    return pd.Series(
        np.where(series.isna(), np.nan, condition.astype(float)), index=series.index
    )


# ----------------------------------------------------------------------------
# Features clinicas compartidas por prioridad (a) y estadia (e)
# ----------------------------------------------------------------------------

_CLINICAS_BASE_NUM = [
    "edad", "num_controles_mes", "num_quimios_mes", "hemoglobina_g_dl",
    "neutrofilos", "plaquetas", "temperatura_c", "peso_kg", "imc",
    "distancia_origen_km", "ingreso_familiar_mensual",
    "acompanante_presente", "requiere_apoyo_psicosocial",
]
_CLINICAS_ENGINEERED = [
    "neutropenia", "neutropenia_severa", "fiebre", "neutropenia_febril",
    "anemia_severa", "plaquetopenia", "carga_clinica", "vulnerabilidad_social",
]
_CLINICAS_CAT = [
    "sexo", "region_origen", "diagnostico_general", "estado_tratamiento",
    "seguro_salud", "alfabetizacion_digital",
]


def _features_clinicas(df: pd.DataFrame) -> pd.DataFrame:
    """Flags e interacciones con umbral clinico estandar (oncologia pediatrica).

    - neutropenia (<500) y neutropenia severa (<200): riesgo de infeccion
    - fiebre (>=38 C) y su interaccion con neutropenia (urgencia oncologica)
    - anemia severa (Hb < 8 g/dL) y plaquetopenia (<50k): soporte transfusional
    - carga_clinica: controles + quimios del mes (intensidad de tratamiento)
    - vulnerabilidad_social: pobreza + lejania + sin seguro + sin acompanante
    """
    out = df.copy()
    out["neutropenia"] = _flag(df["neutrofilos"], df["neutrofilos"] < 500)
    out["neutropenia_severa"] = _flag(df["neutrofilos"], df["neutrofilos"] < 200)
    out["fiebre"] = _flag(df["temperatura_c"], df["temperatura_c"] >= 38.0)
    out["neutropenia_febril"] = out["neutropenia"] * out["fiebre"]
    out["anemia_severa"] = _flag(df["hemoglobina_g_dl"], df["hemoglobina_g_dl"] < 8.0)
    out["plaquetopenia"] = _flag(df["plaquetas"], df["plaquetas"] < 50_000)
    out["carga_clinica"] = df["num_controles_mes"] + df["num_quimios_mes"]
    out["vulnerabilidad_social"] = (
        _flag(df["ingreso_familiar_mensual"], df["ingreso_familiar_mensual"] < 600).fillna(0.0)
        + (df["seguro_salud"] == "Ninguno").astype(float)
        + (df["acompanante_presente"] == 0).astype(float)
        + (df["distancia_origen_km"] > 400).astype(float)
    )
    return out


PRIORIDAD_FEATURES_NUM = _CLINICAS_BASE_NUM + ["dias_hospedaje"] + _CLINICAS_ENGINEERED
PRIORIDAD_FEATURES_CAT = _CLINICAS_CAT


def prioridad_features(pacientes: pd.DataFrame) -> FeatureSet:
    """(a) Clasificacion multiclase del nivel de prioridad de atencion."""
    from .config import PRIORIDAD_CLASSES

    target = "nivel_prioridad_atencion"
    df = _features_clinicas(pacientes.dropna(subset=[target]))
    y = df[target].map({label: i for i, label in enumerate(PRIORIDAD_CLASSES)})
    if y.isna().any():
        raise ValueError(f"Valores de {target} fuera de {PRIORIDAD_CLASSES}")
    return FeatureSet(
        X=df[PRIORIDAD_FEATURES_NUM + PRIORIDAD_FEATURES_CAT].copy(),
        y=y.astype(int),
        numeric_features=PRIORIDAD_FEATURES_NUM,
        categorical_features=PRIORIDAD_FEATURES_CAT,
        class_labels=PRIORIDAD_CLASSES,
    )


ESTADIA_FEATURES_NUM = _CLINICAS_BASE_NUM + _CLINICAS_ENGINEERED
ESTADIA_FEATURES_CAT = _CLINICAS_CAT


def estadia_features(pacientes: pd.DataFrame) -> FeatureSet:
    """(e) Regresion de la duracion de hospedaje (dias_hospedaje).

    No usa nivel_prioridad_atencion ni dias_hospedaje como features: en
    produccion la estadia se estima al ingreso del paciente.
    """
    target = "dias_hospedaje"
    df = _features_clinicas(pacientes.dropna(subset=[target]))
    return FeatureSet(
        X=df[ESTADIA_FEATURES_NUM + ESTADIA_FEATURES_CAT].copy(),
        y=df[target].astype(float),
        numeric_features=ESTADIA_FEATURES_NUM,
        categorical_features=ESTADIA_FEATURES_CAT,
    )


# ----------------------------------------------------------------------------
# Inventario: historial rolling comun a stock critico (b, c) y demanda (d)
# ----------------------------------------------------------------------------

def _con_historial_inventario(consumo: pd.DataFrame) -> pd.DataFrame:
    """Agrega rolling stats por item usando solo dias anteriores (shift 1)
    y atributos del catalogo maestro (inventario_items.csv)."""
    df = consumo.sort_values(["item_id", "fecha"]).copy()
    grupo = df.groupby("item_id")
    consumo_prev = grupo["consumo_real"].transform(lambda s: s.shift(1))
    df["consumo_roll7"] = grupo["consumo_real"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
    df["consumo_roll14"] = grupo["consumo_real"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=5).mean()
    )
    df["consumo_std7"] = grupo["consumo_real"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).std()
    )
    df["ingreso_roll7"] = grupo["ingreso_stock"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).sum()
    )
    df["consumo_lag_1"] = consumo_prev
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    # Dias transcurridos desde el ultimo reabastecimiento (>0): permite al
    # modelo inferir en que punto del ciclo de reposicion esta el item.
    con_ingreso = df["ingreso_stock"] > 0
    ultimo_ingreso = df["fecha"].where(con_ingreso).groupby(df["item_id"]).ffill()
    df["dias_sin_ingreso"] = (df["fecha"] - ultimo_ingreso).dt.days.clip(upper=60)

    items = load_items()[
        ["item_id", "consumo_base_por_familia", "stock_seguridad_7d", "es_critico"]
    ]
    df = df.merge(items, on="item_id", how="left", validate="many_to_one")
    return df


STOCK_FEATURES_NUM = [
    "ocupacion_familias", "stock_inicio", "ingreso_stock", "consumo_real",
    "consumo_roll7", "consumo_roll14", "consumo_std7", "ingreso_roll7",
    "cobertura_bruta", "ratio_stock_seguridad", "consumo_base_por_familia",
    "stock_seguridad_7d", "es_critico", "dia_semana", "mes", "dias_sin_ingreso",
]
STOCK_FEATURES_CAT = ["categoria", "unidad_medida", "item_id"]


def stock_critico_features(consumo: pd.DataFrame, horizon_days: int) -> FeatureSet:
    """(b)/(c) Clasificacion binaria de stock critico a 7 o 14 dias.

    ``cobertura_bruta`` = stock_inicio / consumo promedio reciente: es la
    version SIN fuga de ``dias_cobertura`` (que usa stock_fin y estimados
    derivados del target y esta prohibida).
    """
    if horizon_days not in (7, 14):
        raise ValueError("horizon_days debe ser 7 o 14")
    target = f"stock_critico_{horizon_days}d"

    df = _con_historial_inventario(consumo)
    consumo_ref = df["consumo_roll7"].replace(0, np.nan)
    df["cobertura_bruta"] = (df["stock_inicio"] / consumo_ref).clip(upper=120)
    df["ratio_stock_seguridad"] = (
        df["stock_inicio"] / df["stock_seguridad_7d"].replace(0, np.nan)
    ).clip(upper=60)

    features = STOCK_FEATURES_NUM + STOCK_FEATURES_CAT
    leaked = [col for col in features if col in STOCK_LEAKAGE_FEATURES]
    if leaked:
        raise ValueError(f"Fuga de informacion: {leaked} no pueden ser features de {target}")

    df = df.dropna(subset=[target, "consumo_roll7"])
    return FeatureSet(
        X=df[features].copy(),
        y=df[target].astype(int),
        numeric_features=STOCK_FEATURES_NUM,
        categorical_features=STOCK_FEATURES_CAT,
        dates=df["fecha"],
        class_labels=["normal", "critico"],
    )


DEMANDA_LAGS = [1, 2, 3, 7, 14]
DEMANDA_FEATURES_NUM = (
    [f"consumo_lag_{lag}" for lag in DEMANDA_LAGS]
    + [
        "consumo_roll7", "consumo_roll14", "consumo_std7",
        "ocupacion_familias", "consumo_base_por_familia",
        "base_x_ocupacion", "dia_semana", "mes",
    ]
)
DEMANDA_FEATURES_CAT = ["categoria", "item_id"]


def demanda_features(consumo: pd.DataFrame) -> FeatureSet:
    """(d) Regresion de demanda: consumo_real diario por item.

    Usa lags/rolling del propio consumo, la ocupacion del dia (conocida al
    inicio de la jornada) y el consumo base del catalogo; nunca columnas del
    mismo dia derivadas del stock.
    """
    df = _con_historial_inventario(consumo)
    grupo = df.groupby("item_id")["consumo_real"]
    for lag in DEMANDA_LAGS:
        df[f"consumo_lag_{lag}"] = grupo.shift(lag)
    df["base_x_ocupacion"] = df["consumo_base_por_familia"] * df["ocupacion_familias"]

    df = df.dropna(subset=DEMANDA_FEATURES_NUM + ["consumo_real"])
    return FeatureSet(
        X=df[DEMANDA_FEATURES_NUM + DEMANDA_FEATURES_CAT].copy(),
        y=df["consumo_real"].astype(float),
        numeric_features=DEMANDA_FEATURES_NUM,
        categorical_features=DEMANDA_FEATURES_CAT,
        dates=df["fecha"],
    )


# ----------------------------------------------------------------------------
# Donaciones (f)
# ----------------------------------------------------------------------------

DONACIONES_LAGS = [1, 2, 3]
DONACIONES_FEATURES_NUM = (
    [f"valor_lag_{lag}" for lag in DONACIONES_LAGS]
    + [
        "valor_roll3", "valor_roll6", "donaciones_lag_1", "tendencia",
        "mes", "mes_sin", "mes_cos", "mes_campania",
    ]
)


def donaciones_features(donaciones: pd.DataFrame) -> FeatureSet:
    """(f) Proyeccion mensual del valor donado (S/) por categoria.

    Serie mensual por categoria con lags, medias moviles, estacionalidad
    (mes cíclico + flag de campana: Navidad, Dia del Nino, Aniversario)
    y tendencia lineal.
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
    grupo = mensual.groupby("categoria")["valor_total_soles"]
    for lag in DONACIONES_LAGS:
        mensual[f"valor_lag_{lag}"] = grupo.shift(lag)
    mensual["valor_roll3"] = grupo.transform(lambda s: s.shift(1).rolling(3).mean())
    mensual["valor_roll6"] = grupo.transform(
        lambda s: s.shift(1).rolling(6, min_periods=3).mean()
    )
    mensual["donaciones_lag_1"] = mensual.groupby("categoria")["num_donaciones"].shift(1)
    mensual["mes"] = mensual["mes_periodo"].dt.month
    mensual["mes_sin"] = np.sin(2 * np.pi * mensual["mes"] / 12)
    mensual["mes_cos"] = np.cos(2 * np.pi * mensual["mes"] / 12)
    mensual["mes_campania"] = mensual["mes"].isin([5, 8, 12]).astype(float)
    primer_mes = mensual["mes_periodo"].min()
    mensual["tendencia"] = (mensual["mes_periodo"] - primer_mes).apply(lambda p: p.n)

    mensual = mensual.dropna(subset=DONACIONES_FEATURES_NUM)
    if mensual.empty:
        raise ValueError("No hay suficientes meses de historia de donaciones para los lags")
    return FeatureSet(
        X=mensual[DONACIONES_FEATURES_NUM + ["categoria"]].copy(),
        y=mensual["valor_total_soles"].astype(float),
        numeric_features=DONACIONES_FEATURES_NUM,
        categorical_features=["categoria"],
        dates=mensual["mes_periodo"].dt.to_timestamp(),
    )


# ----------------------------------------------------------------------------
# Frentes con DATOS REALES: (a) prioridad -> triaje de urgencias (Mendeley) y
# (e) estadia -> altas pediatrico-oncologicas NY SPARCS. Ver real_data.py.
# ----------------------------------------------------------------------------

# Escala ESI real 1-5 -> niveles operativos del albergue.
_TRIAGE_A_PRIORIDAD = {1: "Alto", 2: "Alto", 3: "Medio", 4: "Bajo", 5: "Bajo"}

PRIORIDAD_REAL_NUM = [
    "edad", "mes_admision", "dia_semana_admision", "hora_admision",
    "grado_dolor", "estado_critico", "estado_estupor",
    "distres_mental", "distres_material",
    "presion_sistolica", "presion_diastolica", "pulso",
    "frecuencia_respiratoria", "saturacion_o2",
]
PRIORIDAD_REAL_CAT = ["sexo", "motivo_consulta", "avpu"]
_TOP_MOTIVOS = 40
_MAX_MUESTRAS_TRIAGE = 45_000


def prioridad_triaje_features(triage: pd.DataFrame) -> FeatureSet:
    """(a) Prioridad real: grado de triaje asignado por enfermeria (ESI 1-5).

    Se excluyen columnas posteriores a la decision de triaje
    (``NeedFastExecute``, ``operational_patient``, ``ref_specialist``).
    La temperatura se descarta (<1% de cobertura); el resto de signos vitales
    se conserva y su ausencia la imputa el pipeline.
    """
    from .config import PRIORIDAD_CLASSES, RANDOM_STATE

    df = triage.dropna(subset=["TriageGrade"]).copy()
    df["prioridad"] = df["TriageGrade"].map(_TRIAGE_A_PRIORIDAD)
    df = df.dropna(subset=["prioridad"])

    top = df["ChiefComplaint"].value_counts().head(_TOP_MOTIVOS).index
    df["motivo_consulta"] = df["ChiefComplaint"].where(
        df["ChiefComplaint"].isin(top), "otros"
    )
    renombres = {
        "age": "edad", "gender": "sexo", "admission_month": "mes_admision",
        "admission_weekday": "dia_semana_admision", "admission_hour": "hora_admision",
        "PainGrade": "grado_dolor", "CriticalStatus": "estado_critico",
        "StuporStatus": "estado_estupor", "MentalDistress": "distres_mental",
        "MaterialDistress": "distres_material", "AVPU": "avpu",
        "BlooddpressurSystol": "presion_sistolica",
        "BlooddpressurDiastol": "presion_diastolica",
        "PulseRate": "pulso", "RespiratoryRate": "frecuencia_respiratoria",
        "O2Saturation": "saturacion_o2",
    }
    df = df.rename(columns=renombres)

    # Muestreo aleatorio reproducible para que el tuning con CV sea tratable
    # (conserva las proporciones de clase por ley de grandes numeros).
    if len(df) > _MAX_MUESTRAS_TRIAGE:
        df = df.sample(n=_MAX_MUESTRAS_TRIAGE, random_state=RANDOM_STATE)

    y = df["prioridad"].map({label: i for i, label in enumerate(PRIORIDAD_CLASSES)})
    return FeatureSet(
        X=df[PRIORIDAD_REAL_NUM + PRIORIDAD_REAL_CAT].copy(),
        y=y.astype(int),
        numeric_features=PRIORIDAD_REAL_NUM,
        categorical_features=PRIORIDAD_REAL_CAT,
        class_labels=PRIORIDAD_CLASSES,
    )


ESTADIA_REAL_NUM = ["anio_alta"]
ESTADIA_REAL_CAT = [
    "gender", "race", "ethnicity", "hospital_service_area", "type_of_admission",
    "ccsr_diagnosis_code", "apr_drg_code", "apr_medical_surgical",
    "payment_typology_1", "emergency_department_indicator",
]


def estadia_sparcs_features(sparcs: pd.DataFrame) -> FeatureSet:
    """(e) Estadia real: dias de hospitalizacion de altas pediatrico-oncologicas.

    Se excluyen resultados post-alta que fugarian informacion del target:
    severidad/mortalidad APR (abstraccion del alta), cargos y costos totales
    y disposicion del paciente.
    """
    df = sparcs.copy()
    # "120 +" es el tope censurado del dataset publico.
    df["length_of_stay"] = (
        df["length_of_stay"].astype(str).str.replace(" +", "", regex=False).str.strip()
    )
    df["dias_estadia"] = pd.to_numeric(df["length_of_stay"], errors="coerce")
    df = df.dropna(subset=["dias_estadia"])
    return FeatureSet(
        X=df[ESTADIA_REAL_NUM + ESTADIA_REAL_CAT].copy(),
        y=df["dias_estadia"].astype(float),
        numeric_features=ESTADIA_REAL_NUM,
        categorical_features=ESTADIA_REAL_CAT,
    )
