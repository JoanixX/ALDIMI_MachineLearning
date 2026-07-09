"""Descarga y carga de los datasets REALES del proyecto.

Tras la revision del dataset sintetico, el proyecto usa datos reales en los
frentes donde existe un dataset publico compatible:

- (a) Prioridad de atencion -> **ED triage, hospital universitario de Iran**
  (Mendeley Data, doi:10.17632/vhzyyktrz5.1; 143,582 admisiones reales de
  urgencias con edad, motivo de consulta, dolor, estados critico/estupor,
  signos vitales y el grado de triaje asignado por enfermeria, escala 1-5).
- (e) Duracion de estadia -> **NY SPARCS Hospital Inpatient Discharges
  (De-Identified)**, altas hospitalarias reales de pacientes pediatricos
  (0-17) con diagnostico oncologico, 2021-2023 (~10,500 altas con dias de
  estadia, diagnostico CCSR, tipo de admision, aseguradora, region).

Para (b)(c) stock critico, (d) demanda y (f) donaciones no existe un dataset
publico con la granularidad requerida (inventario diario por item de un
albergue; donaciones fechadas por categoria) — se documento la busqueda en
docs/informe_final.md — por lo que esos frentes usan el dataset sintetico v3
como ultima instancia.

Uso:
    python -m aldimi_models.real_data   # descarga ambos datasets a datos/reales/
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

REAL_DATA_DIR = DATA_DIR / "reales"
TRIAGE_CSV = "ed_triage_iran.csv"
SPARCS_CSV = "sparcs_pediatrico_oncologico.csv"

_MENDELEY_TRIAGE_URL = (
    "https://data.mendeley.com/public-files/datasets/vhzyyktrz5/files/"
    "67ebb6a5-33e5-4fcf-a1ac-e566e8ee2265/file_downloaded"
)
_SPARCS_DATASETS: dict[int, str] = {2021: "tg3i-cinn", 2022: "5dtw-tffi", 2023: "46xm-urtu"}
_SPARCS_WHERE = (
    "age_group='0 to 17' AND (starts_with(ccsr_diagnosis_code,'NEO') OR apr_mdc_code='17')"
)


class RealDataMissingError(FileNotFoundError):
    """El dataset real no esta descargado todavia."""


def descargar_triage(data_dir: Path = REAL_DATA_DIR) -> Path:
    """Descarga el CSV de triaje de Mendeley Data (publico, ~10 MB)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    destino = data_dir / TRIAGE_CSV
    print(f"Descargando triaje (Mendeley) -> {destino} ...")
    urllib.request.urlretrieve(_MENDELEY_TRIAGE_URL, destino)
    return destino


def descargar_sparcs(data_dir: Path = REAL_DATA_DIR) -> Path:
    """Descarga las altas pediatrico-oncologicas de SPARCS via API Socrata."""
    data_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year, dataset_id in _SPARCS_DATASETS.items():
        url = (
            f"https://health.data.ny.gov/resource/{dataset_id}.json?"
            + urllib.parse.urlencode({"$where": _SPARCS_WHERE, "$limit": 50000})
        )
        print(f"Descargando SPARCS {year} ...")
        with urllib.request.urlopen(url, timeout=120) as response:
            rows = json.load(response)
        df = pd.DataFrame(rows)
        df["anio_alta"] = year
        frames.append(df)
    destino = data_dir / SPARCS_CSV
    pd.concat(frames, ignore_index=True).to_csv(destino, index=False, encoding="utf-8-sig")
    return destino


def _read_real(filename: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise RealDataMissingError(
            f"No existe {path}. Descarga los datos reales con: python -m aldimi_models.real_data"
        )
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    if df.empty:
        raise RealDataMissingError(f"{path} esta vacio; vuelve a descargarlo")
    return df


def load_triage(data_dir: Path = REAL_DATA_DIR) -> pd.DataFrame:
    """Triaje real de urgencias (143k filas). Target: TriageGrade 1-5."""
    df = _read_real(TRIAGE_CSV, data_dir)
    esperadas = {"age", "gender", "ChiefComplaint", "PainGrade", "TriageGrade"}
    faltantes = esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"{TRIAGE_CSV}: faltan columnas {sorted(faltantes)}")
    return df


def load_sparcs(data_dir: Path = REAL_DATA_DIR) -> pd.DataFrame:
    """Altas pediatrico-oncologicas reales de NY. Target: length_of_stay."""
    df = _read_real(SPARCS_CSV, data_dir)
    esperadas = {"length_of_stay", "ccsr_diagnosis_code", "type_of_admission", "gender"}
    faltantes = esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"{SPARCS_CSV}: faltan columnas {sorted(faltantes)}")
    return df


def main() -> None:
    triage = descargar_triage()
    sparcs = descargar_sparcs()
    print(f"triaje: {sum(1 for _ in open(triage, encoding='utf-8', errors='ignore')) - 1} filas")
    print(f"sparcs: {sum(1 for _ in open(sparcs, encoding='utf-8', errors='ignore')) - 1} filas")


if __name__ == "__main__":
    main()
