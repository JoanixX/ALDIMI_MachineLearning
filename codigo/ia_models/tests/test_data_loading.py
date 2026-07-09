"""Tests de carga y validacion de los datasets."""

from pathlib import Path

import pytest

from aldimi_models import data_loading
from aldimi_models.config import DATA_DIR


def test_data_dir_exists() -> None:
    assert DATA_DIR.exists(), f"No existe la carpeta de datos: {DATA_DIR}"


def test_load_all_returns_five_datasets() -> None:
    datasets = data_loading.load_all()
    assert set(datasets) == {"pacientes", "consumo", "items", "donaciones", "capturas_ia"}
    for nombre, df in datasets.items():
        assert not df.empty, f"{nombre} cargo vacio"


def test_pacientes_has_target_and_parsed_dates() -> None:
    pacientes = data_loading.load_pacientes()
    assert "nivel_prioridad_atencion" in pacientes.columns
    assert str(pacientes["fecha_registro"].dtype).startswith("datetime64")


def test_consumo_has_stock_targets() -> None:
    consumo = data_loading.load_consumo()
    assert {"stock_critico_7d", "stock_critico_14d"} <= set(consumo.columns)
    assert set(consumo["stock_critico_7d"].dropna().unique()) <= {0, 1}


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        data_loading.load_pacientes(data_dir=tmp_path)


def test_wrong_schema_raises_validation_error(tmp_path: Path) -> None:
    (tmp_path / "pacientes_riesgo_sintetico.csv").write_text(
        "columna_incorrecta\n1\n", encoding="utf-8"
    )
    with pytest.raises(data_loading.DataValidationError):
        data_loading.load_pacientes(data_dir=tmp_path)
