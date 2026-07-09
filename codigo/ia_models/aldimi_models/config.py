"""Configuracion central del paquete: rutas, semilla y constantes de dominio."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parent
IA_MODELS_DIR: Path = PACKAGE_DIR.parent
REPO_ROOT: Path = IA_MODELS_DIR.parent.parent
DATA_DIR: Path = REPO_ROOT / "datos"
OUTPUTS_DIR: Path = IA_MODELS_DIR / "outputs"
ARTIFACTS_DIR: Path = IA_MODELS_DIR / "artifacts"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# Variables derivadas directamente de los targets de stock critico:
# usarlas como features seria fuga de informacion.
STOCK_LEAKAGE_FEATURES: list[str] = [
    "stock_fin",
    "consumo_estimado_7d",
    "consumo_estimado_14d",
    "dias_cobertura",
]

# Orden canonico de clases del modelo de prioridad (se codifica 0/1/2).
PRIORIDAD_CLASSES: list[str] = ["Bajo", "Medio", "Alto"]

CSV_ENCODING: str = "utf-8-sig"
