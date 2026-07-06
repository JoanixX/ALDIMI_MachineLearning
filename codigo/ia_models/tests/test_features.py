"""Tests de construccion de features de los seis frentes."""

import pytest

from aldimi_models.data_loading import load_consumo, load_donaciones, load_pacientes
from aldimi_models.features import (
    demanda_features,
    donaciones_features,
    estadia_features,
    prioridad_features,
)


@pytest.fixture(scope="module")
def pacientes():
    return load_pacientes()


def test_prioridad_features_encodes_target(pacientes) -> None:
    fs = prioridad_features(pacientes)
    assert set(fs.y.unique()) <= {0, 1, 2}
    assert fs.class_labels == ["Bajo", "Medio", "Alto"]
    assert "nivel_prioridad_atencion" not in fs.X.columns
    assert len(fs.X) == len(fs.y)


def test_estadia_features_excludes_target_and_prioridad(pacientes) -> None:
    fs = estadia_features(pacientes)
    assert "dias_hospedaje" not in fs.X.columns
    assert "nivel_prioridad_atencion" not in fs.X.columns
    assert fs.y.dtype == float


def test_demanda_features_lags_have_no_nans() -> None:
    fs = demanda_features(load_consumo())
    assert not fs.X[fs.numeric_features].isna().any().any()
    assert fs.dates is not None
    assert len(fs.X) == len(fs.y) == len(fs.dates)


def test_donaciones_features_monthly_aggregation() -> None:
    fs = donaciones_features(load_donaciones())
    assert "categoria" in fs.X.columns
    assert (fs.y >= 0).all()
    assert fs.dates is not None
