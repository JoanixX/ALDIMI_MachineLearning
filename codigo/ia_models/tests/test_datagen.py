"""Tests del generador de datos sinteticos v3."""

import numpy as np
import pandas as pd

from aldimi_models.data_loading import _EXPECTED_COLUMNS
from aldimi_models.datagen import generar_capturas, generar_pacientes


def test_pacientes_reproducible_con_misma_semilla() -> None:
    a = generar_pacientes(np.random.default_rng(7), n=500)
    b = generar_pacientes(np.random.default_rng(7), n=500)
    pd.testing.assert_frame_equal(a, b)


def test_pacientes_cumple_esquema_del_loader() -> None:
    df = generar_pacientes(np.random.default_rng(1), n=300)
    esperadas = _EXPECTED_COLUMNS["pacientes_riesgo_sintetico.csv"]
    assert list(df.columns) == esperadas


def test_pacientes_targets_validos() -> None:
    df = generar_pacientes(np.random.default_rng(1), n=2000)
    assert set(df["nivel_prioridad_atencion"].unique()) <= {"Bajo", "Medio", "Alto"}
    assert (df["dias_hospedaje"] >= 1).all()
    assert df["edad"].between(0, 17).all()


def test_capturas_cumple_esquema_y_rangos() -> None:
    pacientes = generar_pacientes(np.random.default_rng(2), n=200)
    df = generar_capturas(np.random.default_rng(2), pacientes, n=400)
    esperadas = _EXPECTED_COLUMNS["capturas_ia_sinteticas.csv"]
    assert list(df.columns) == esperadas
    assert df["confianza_ocr"].between(0, 1).all()
    assert set(df["requiere_revision_manual"].unique()) <= {0, 1}
