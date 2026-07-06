"""Tests anti-fuga: las features de stock critico no derivan del target."""

import pytest

from aldimi_models.config import STOCK_LEAKAGE_FEATURES
from aldimi_models.data_loading import load_consumo
from aldimi_models.features import stock_critico_features


@pytest.fixture(scope="module")
def consumo():
    return load_consumo()


@pytest.mark.parametrize("horizon", [7, 14])
def test_stock_features_exclude_leakage_columns(consumo, horizon: int) -> None:
    fs = stock_critico_features(consumo, horizon_days=horizon)
    used = set(fs.X.columns)
    leaked = used & set(STOCK_LEAKAGE_FEATURES)
    assert not leaked, f"Fuga de informacion en stock_critico_{horizon}d: {sorted(leaked)}"


@pytest.mark.parametrize("horizon", [7, 14])
def test_stock_features_exclude_both_targets(consumo, horizon: int) -> None:
    fs = stock_critico_features(consumo, horizon_days=horizon)
    assert "stock_critico_7d" not in fs.X.columns
    assert "stock_critico_14d" not in fs.X.columns


def test_invalid_horizon_raises() -> None:
    with pytest.raises(ValueError):
        stock_critico_features(load_consumo(), horizon_days=10)
