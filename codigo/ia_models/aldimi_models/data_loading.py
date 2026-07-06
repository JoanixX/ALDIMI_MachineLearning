"""Carga y validacion de los CSV del proyecto.

Unica fuente de verdad para leer datos: ningun otro modulo debe llamar a
``pd.read_csv`` directamente sobre los archivos de /datos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR


class DataValidationError(Exception):
    """El archivo existe pero no cumple el esquema esperado."""


_EXPECTED_COLUMNS: dict[str, list[str]] = {
    "pacientes_riesgo_sintetico.csv": [
        "paciente_id", "fecha_registro", "edad", "sexo", "region_origen",
        "diagnostico_general", "estado_tratamiento", "dias_hospedaje",
        "num_controles_mes", "num_quimios_mes", "hemoglobina_g_dl",
        "neutrofilos", "plaquetas", "temperatura_c", "peso_kg", "imc",
        "distancia_origen_km", "ingreso_familiar_mensual",
        "acompanante_presente", "seguro_salud", "alfabetizacion_digital",
        "requiere_apoyo_psicosocial", "nivel_prioridad_atencion",
    ],
    "consumo_inventario_diario_sintetico.csv": [
        "fecha", "item_id", "nombre_item", "categoria", "unidad_medida",
        "ocupacion_familias", "stock_inicio", "ingreso_stock", "consumo_real",
        "stock_fin", "consumo_estimado_7d", "consumo_estimado_14d",
        "dias_cobertura", "stock_critico_7d", "stock_critico_14d",
    ],
    "donaciones_sinteticas.csv": [
        "donacion_id", "fecha", "tipo_donante", "item_id", "nombre_item",
        "categoria", "unidad_medida", "cantidad_donada",
        "valor_estimado_soles", "campania",
    ],
    "inventario_items.csv": [
        "item_id", "nombre_item", "categoria", "unidad_medida",
        "consumo_base_por_familia", "stock_seguridad_7d",
        "stock_maximo_referencial", "es_critico",
    ],
    "capturas_ia_sinteticas.csv": [
        "captura_id", "fecha_captura", "paciente_id", "tipo_documento",
        "calidad_imagen", "confianza_ocr", "campos_extraidos",
        "requiere_revision_manual", "origen_captura",
    ],
}


def _read_validated(filename: str, date_columns: list[str], data_dir: Path) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = _EXPECTED_COLUMNS[filename]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise DataValidationError(f"{filename}: faltan columnas esperadas {missing}")
    if df.empty:
        raise DataValidationError(f"{filename}: el archivo no contiene filas")
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_pacientes(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Pacientes con indicadores clinicos y nivel de prioridad (target multiclase)."""
    return _read_validated("pacientes_riesgo_sintetico.csv", ["fecha_registro"], data_dir)


def load_consumo(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Consumo/stock diario por item (targets stock_critico_7d/14d y demanda)."""
    return _read_validated("consumo_inventario_diario_sintetico.csv", ["fecha"], data_dir)


def load_donaciones(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Donaciones historicas (target de proyeccion por categoria)."""
    return _read_validated("donaciones_sinteticas.csv", ["fecha"], data_dir)


def load_items(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Catalogo maestro de items de inventario."""
    return _read_validated("inventario_items.csv", [], data_dir)


def load_capturas_ia(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Capturas OCR: contrato de datos con el curso hermano de IA."""
    return _read_validated("capturas_ia_sinteticas.csv", ["fecha_captura"], data_dir)


def load_all(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Carga los cinco datasets en un diccionario nombre -> DataFrame."""
    return {
        "pacientes": load_pacientes(data_dir),
        "consumo": load_consumo(data_dir),
        "items": load_items(data_dir),
        "donaciones": load_donaciones(data_dir),
        "capturas_ia": load_capturas_ia(data_dir),
    }
