"""Tests de los datasets reales (se omiten si aun no fueron descargados)."""

import pytest

from aldimi_models.real_data import (
    REAL_DATA_DIR,
    SPARCS_CSV,
    TRIAGE_CSV,
    load_sparcs,
    load_triage,
)

triage_disponible = (REAL_DATA_DIR / TRIAGE_CSV).exists()
sparcs_disponible = (REAL_DATA_DIR / SPARCS_CSV).exists()


@pytest.mark.skipif(not triage_disponible, reason="ejecutar: python -m aldimi_models.real_data")
def test_triage_features_reales() -> None:
    from aldimi_models.features import prioridad_triaje_features

    fs = prioridad_triaje_features(load_triage())
    assert fs.class_labels == ["Bajo", "Medio", "Alto"]
    assert set(fs.y.unique()) <= {0, 1, 2}
    # Columnas posteriores a la decision de triaje no pueden ser features.
    assert not {"NeedFastExecute", "operational_patient", "ref_specialist"} & set(fs.X.columns)
    assert len(fs.X) == len(fs.y) > 10_000


@pytest.mark.skipif(not sparcs_disponible, reason="ejecutar: python -m aldimi_models.real_data")
def test_sparcs_features_reales() -> None:
    from aldimi_models.features import estadia_sparcs_features

    fs = estadia_sparcs_features(load_sparcs())
    assert (fs.y >= 0).all() and fs.y.max() <= 120
    # Abstracciones del alta (fuga) excluidas de las features.
    fuga = {"apr_severity_of_illness", "apr_risk_of_mortality", "total_charges",
            "total_costs", "patient_disposition", "length_of_stay"}
    assert not fuga & set(fs.X.columns)
    assert len(fs.X) > 5_000
