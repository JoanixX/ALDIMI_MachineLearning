"""Generador del dataset sintetico v3 de ALDIMI Predict.

Regenera los CSV de /datos con el MISMO esquema de columnas que la v2 pero con
una estructura causal realista: cada target depende de variables observables
mas ruido controlado, de modo que un buen modelo pueda aprender la relacion
sin que sea trivial ni memorizable (anti-overfitting: el ruido es irreducible).

Relaciones generadas (documentadas tambien en datos/DICCIONARIO_DATOS.md):

- ``nivel_prioridad_atencion``: severidad clinica latente (labs, fiebre,
  neutropenia, estado de tratamiento) + vulnerabilidad social.
- ``dias_hospedaje``: diagnostico x fase de tratamiento + quimios/mes +
  distancia de origen + severidad + ruido gamma.
- ``consumo_real``: consumo base del item x ocupacion x dia de semana x
  estacionalidad x ruido lognormal.
- ``stock_critico_7d/14d``: contabilidad real de inventario con politica de
  reposicion con lead time (los quiebres ocurren mientras llega el pedido).
- ``valor_estimado_soles`` (donaciones): estacionalidad mensual (Navidad,
  Dia del Nino) + tendencia + mezcla de tipos de donante.

Uso:
    python -m aldimi_models.datagen [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

# ----------------------------------------------------------------------------
# Vocabularios (identicos a la v2 para mantener el contrato de datos)
# ----------------------------------------------------------------------------

REGIONES = {
    # region: (probabilidad, distancia_km media, distancia_km sigma lognormal)
    "Lima": (0.28, 25, 0.8),
    "Callao_no": (0.0, 15, 0.5),  # placeholder no usado
    "Piura": (0.07, 980, 0.15),
    "La Libertad": (0.08, 560, 0.15),
    "Cajamarca": (0.06, 860, 0.18),
    "Junín": (0.09, 300, 0.25),
    "Huánuco": (0.06, 410, 0.2),
    "Ayacucho": (0.05, 570, 0.2),
    "Cusco": (0.08, 1100, 0.15),
    "Puno": (0.05, 1300, 0.12),
    "Arequipa": (0.07, 1010, 0.12),
    "Loreto": (0.06, 1030, 0.2),
    "San Martín": (0.05, 870, 0.18),
}
REGIONES = {k: v for k, v in REGIONES.items() if v[0] > 0}

DIAGNOSTICOS = {
    # diagnostico: (prob, dias_hospedaje_base, severidad_extra)
    "Leucemia linfoblástica aguda": (0.34, 46, 0.10),
    "Leucemia mieloide": (0.12, 58, 0.35),
    "Linfoma": (0.15, 30, 0.00),
    "Tumor sólido": (0.17, 26, -0.05),
    "Sarcoma": (0.11, 34, 0.10),
    "Neuroblastoma": (0.11, 40, 0.20),
}

ESTADOS = {
    # estado: (prob, factor_estadia, severidad_extra, quimios_lambda)
    "Inicio": (0.16, 1.30, 0.45, 2.2),
    "En tratamiento": (0.48, 1.00, 0.20, 2.8),
    "Recaída": (0.12, 1.55, 0.95, 3.4),
    "Seguimiento": (0.24, 0.25, -0.85, 0.3),
}

SEGUROS = ["SIS", "EsSalud", "Privado", "Ninguno"]
ALFABETIZACION = ["Baja", "Media", "Alta"]
TIPOS_DONANTE = {
    "Persona natural": (0.38, 250),
    "Empresa": (0.20, 1400),
    "Iglesia/Comunidad": (0.14, 500),
    "ONG": (0.12, 1100),
    "Campaña digital": (0.10, 700),
    "Universidad": (0.06, 650),
}
TIPOS_DOCUMENTO = ["Receta médica", "DNI", "Hoja clínica", "Orden de laboratorio", "Informe social"]
CALIDADES = ["Alta", "Media", "Baja"]
ORIGENES_CAPTURA = ["Web", "Móvil", "Carga manual"]

N_PACIENTES = 12_000
CONSUMO_INICIO, CONSUMO_DIAS = "2025-01-01", 730
DONACIONES_MESES = 48  # 2022-09 .. 2026-08
N_CAPTURAS = 6_000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _choice(rng: np.random.Generator, table: dict, n: int) -> np.ndarray:
    keys = list(table)
    probs = np.array([table[k][0] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=n, p=probs)


# ----------------------------------------------------------------------------
# 1. Pacientes
# ----------------------------------------------------------------------------

def generar_pacientes(rng: np.random.Generator, n: int = N_PACIENTES) -> pd.DataFrame:
    edad = np.clip(rng.gamma(3.2, 2.4, n).round(), 0, 17).astype(int)
    sexo = rng.choice(["F", "M"], size=n, p=[0.47, 0.53])
    region = _choice(rng, REGIONES, n)
    dist_media = np.array([REGIONES[r][1] for r in region], dtype=float)
    dist_sigma = np.array([REGIONES[r][2] for r in region], dtype=float)
    distancia = np.round(dist_media * rng.lognormal(0.0, dist_sigma, n), 1)

    diagnostico = _choice(rng, DIAGNOSTICOS, n)
    estado = _choice(rng, ESTADOS, n)
    sev_diag = np.array([DIAGNOSTICOS[d][2] for d in diagnostico])
    sev_estado = np.array([ESTADOS[e][2] for e in estado])
    quimios_lambda = np.array([ESTADOS[e][3] for e in estado])

    num_quimios = rng.poisson(quimios_lambda)
    num_controles = 1 + rng.poisson(1.2 + 1.1 * (estado != "Seguimiento"))

    # Severidad clinica latente: motor comun de labs, prioridad y estadia.
    severidad = rng.normal(0.0, 1.0, n) * 0.9 + sev_diag + sev_estado

    hemoglobina = np.clip(12.4 - 1.5 * severidad - 0.25 * num_quimios + rng.normal(0, 0.55, n), 4.5, 16.5).round(1)
    neutrofilos = np.clip(np.exp(7.25 - 0.55 * severidad + rng.normal(0, 0.28, n)), 50, 9000).round(0)
    plaquetas = np.clip(np.exp(12.15 - 0.38 * severidad + rng.normal(0, 0.28, n)), 8_000, 650_000).round(0)
    temperatura = np.clip(36.7 + 0.38 * np.maximum(severidad, 0) + rng.normal(0, 0.22, n), 35.5, 40.8).round(1)

    peso = np.clip(9.5 + 3.1 * edad + rng.normal(0, 3.2, n) - 1.1 * np.maximum(severidad, 0), 3.5, 90).round(1)
    imc = np.clip(16.6 + rng.normal(0, 2.1, n) - 0.55 * np.maximum(severidad, 0), 11, 30).round(1)

    ingreso = np.round(rng.lognormal(6.95, 0.55, n), 0)
    p_seguro = np.column_stack([
        0.78 - 0.25 * (ingreso > 1500),                # SIS
        0.14 + 0.10 * (ingreso > 1500),                # EsSalud
        0.02 + 0.14 * (ingreso > 2500),                # Privado
        np.full(n, 0.06),                              # Ninguno
    ])
    p_seguro = p_seguro / p_seguro.sum(axis=1, keepdims=True)
    seguro = np.array([rng.choice(SEGUROS, p=p) for p in p_seguro])

    p_alf = _sigmoid((ingreso - 900) / 500) * 0.8 + 0.1 - 0.15 * (region != "Lima")
    alfabetizacion = np.where(
        rng.random(n) < np.clip(p_alf - 0.25, 0.02, 0.9), "Alta",
        np.where(rng.random(n) < 0.55, "Media", "Baja"),
    )

    acompanante = (rng.random(n) < (0.88 - 0.012 * edad)).astype(int)
    vulnerabilidad = (
        0.55 * (ingreso < 600).astype(float)
        + 0.35 * (distancia > 400).astype(float)
        + 0.30 * (seguro == "Ninguno").astype(float)
        + 0.25 * (acompanante == 0).astype(float)
    )
    apoyo_psico = (rng.random(n) < _sigmoid(0.9 * severidad + 1.2 * vulnerabilidad - 1.1)).astype(int)

    # Target (a): prioridad de atencion -----------------------------------
    neutropenia = (neutrofilos < 500).astype(float)
    fiebre = (temperatura >= 38.0).astype(float)
    anemia = (hemoglobina < 8.0).astype(float)
    plaquetopenia = (plaquetas < 50_000).astype(float)
    score = (
        0.70 * severidad
        + 1.25 * neutropenia
        + 1.10 * fiebre
        + 0.85 * anemia
        + 0.75 * plaquetopenia
        + 0.60 * (estado == "Recaída").astype(float)
        + 0.30 * (num_quimios >= 3).astype(float)
        + 0.45 * vulnerabilidad
        + rng.normal(0, 0.15, n)  # ruido irreducible
    )
    q_bajo, q_alto = np.quantile(score, [0.45, 0.80])
    prioridad = np.where(score < q_bajo, "Bajo", np.where(score < q_alto, "Medio", "Alto"))

    # Target (e): dias de hospedaje ----------------------------------------
    base_diag = np.array([DIAGNOSTICOS[d][1] for d in diagnostico], dtype=float)
    factor_estado = np.array([ESTADOS[e][1] for e in estado])
    dias = (
        base_diag * factor_estado
        + 5.5 * num_quimios
        + 0.035 * distancia
        + 5.0 * severidad
        + rng.gamma(2.2, 5.0, n)  # ruido irreducible (media 11, sd ~7.4)
    )
    dias_hospedaje = np.clip(np.round(dias), 1, 365).astype(int)

    fecha = pd.Timestamp("2024-09-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")

    df = pd.DataFrame({
        "paciente_id": [f"PAC{i:05d}" for i in range(1, n + 1)],
        "fecha_registro": fecha.strftime("%Y-%m-%d"),
        "edad": edad,
        "sexo": sexo,
        "region_origen": region,
        "diagnostico_general": diagnostico,
        "estado_tratamiento": estado,
        "dias_hospedaje": dias_hospedaje,
        "num_controles_mes": num_controles,
        "num_quimios_mes": num_quimios,
        "hemoglobina_g_dl": hemoglobina,
        "neutrofilos": neutrofilos,
        "plaquetas": plaquetas,
        "temperatura_c": temperatura,
        "peso_kg": peso,
        "imc": imc,
        "distancia_origen_km": distancia,
        "ingreso_familiar_mensual": ingreso,
        "acompanante_presente": acompanante,
        "seguro_salud": seguro,
        "alfabetizacion_digital": alfabetizacion,
        "requiere_apoyo_psicosocial": apoyo_psico,
        "nivel_prioridad_atencion": prioridad,
    })

    # Nulos realistas (2-4% MCAR) en columnas de laboratorio y sociales.
    for col, frac in [
        ("hemoglobina_g_dl", 0.025), ("neutrofilos", 0.02), ("plaquetas", 0.02),
        ("imc", 0.02), ("ingreso_familiar_mensual", 0.04), ("alfabetizacion_digital", 0.015),
    ]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan
    return df


# ----------------------------------------------------------------------------
# 2. Inventario diario (consumo + stock critico)
# ----------------------------------------------------------------------------

def generar_consumo(rng: np.random.Generator, items: pd.DataFrame) -> pd.DataFrame:
    fechas = pd.date_range(CONSUMO_INICIO, periods=CONSUMO_DIAS, freq="D")
    n_dias = len(fechas)

    # Ocupacion del albergue: estacional + autocorrelacionada (comun a items).
    doy = fechas.dayofyear.to_numpy()
    ocupacion = 45 + 7 * np.sin(2 * np.pi * (doy - 40) / 365)
    ar = np.zeros(n_dias)
    for t in range(1, n_dias):
        ar[t] = 0.85 * ar[t - 1] + rng.normal(0, 1.6)
    ocupacion = np.clip(np.round(ocupacion + ar), 22, 60).astype(int)

    dow_factor = np.array([1.0, 0.98, 0.99, 1.0, 1.05, 1.12, 1.08])  # lun..dom
    rows: list[dict] = []
    for item in items.itertuples(index=False):
        base = item.consumo_base_por_familia
        fase = rng.uniform(0, 2 * np.pi)
        estacional = 1 + 0.12 * np.sin(2 * np.pi * doy / 365 + fase)
        # Ruido de demanda diaria (sigma 0.15): incertidumbre irreducible que
        # acota el R2 alcanzable del modelo de demanda (~0.9, no ~1).
        demanda = (
            base * ocupacion * dow_factor[fechas.dayofweek] * estacional
            * rng.lognormal(0.0, 0.15, n_dias)
        )
        demanda = np.maximum(np.round(demanda, 1), 0.0)

        stock = float(item.stock_maximo_referencial) * rng.uniform(0.55, 0.85)
        pedido_pendiente = 0  # dias hasta que llegue (0 = sin pedido)
        pedido_qty = 0.0
        # Punto de pedido caracteristico de cada item (estable en el tiempo:
        # el modelo puede aprenderlo a traves de item_id y su historial).
        punto_pedido = float(rng.uniform(10, 14))
        historial: list[float] = []
        stock_fin_serie: list[float] = []
        item_rows: list[dict] = []
        for t in range(n_dias):
            ingreso = 0.0
            if pedido_pendiente > 0:
                pedido_pendiente -= 1
                if pedido_pendiente == 0:
                    ingreso = pedido_qty
                    pedido_qty = 0.0
            stock_inicio = stock + ingreso
            consumo = min(demanda[t], stock_inicio)
            stock_fin = stock_inicio - consumo

            historial.append(consumo)
            ventana = historial[-7:]
            est_diario = float(np.mean(ventana)) if ventana else base * float(ocupacion[t])
            est_7d = round(est_diario * 7, 2)
            est_14d = round(est_diario * 14, 2)
            cobertura = round(stock_fin / est_diario, 2) if est_diario > 0 else 99.0

            # Politica de reposicion: pedir cuando la cobertura proyectada baja
            # de ~13 dias; el pedido tarda 5 dias en llegar (alli se producen
            # la mayoria de los quiebres reales).
            if pedido_pendiente == 0 and pedido_qty == 0.0 and cobertura < punto_pedido:
                pedido_pendiente = 5
                pedido_qty = float(item.stock_maximo_referencial) - stock_fin

            item_rows.append({
                "fecha": fechas[t].strftime("%Y-%m-%d"),
                "item_id": item.item_id,
                "nombre_item": item.nombre_item,
                "categoria": item.categoria,
                "unidad_medida": item.unidad_medida,
                "ocupacion_familias": int(ocupacion[t]),
                "stock_inicio": round(stock_inicio, 1),
                "ingreso_stock": round(ingreso, 1),
                "consumo_real": round(consumo, 1),
                "stock_fin": round(stock_fin, 1),
                "consumo_estimado_7d": est_7d,
                "consumo_estimado_14d": est_14d,
                "dias_cobertura": cobertura,
            })
            stock_fin_serie.append(stock_fin)
            stock = stock_fin

        # Targets FORWARD-LOOKING: critico = el stock realmente caera por
        # debajo del minimo de seguridad en los proximos 7/14 dias. Depende
        # de la demanda futura y de cuando llegue el pedido (ambos inciertos
        # hoy), asi que ningun modelo puede acertar el 100%: la etiqueta tiene
        # incertidumbre irreducible genuina, como en la operacion real.
        serie = np.array(stock_fin_serie)
        # "Critico" = quedar con menos del stock de seguridad + ~5 dias de
        # demanda tipica en algun punto del horizonte.
        demanda_tipica = base * 45.0
        umbral = float(item.stock_seguridad_7d) + 5.0 * demanda_tipica
        for t, row in enumerate(item_rows):
            fut7 = serie[t + 1: t + 8]
            fut14 = serie[t + 1: t + 15]
            row["stock_critico_7d"] = (
                int(fut7.min() < umbral) if len(fut7) == 7 else np.nan
            )
            row["stock_critico_14d"] = (
                int(fut14.min() < umbral) if len(fut14) == 14 else np.nan
            )
        rows.extend(item_rows)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Donaciones
# ----------------------------------------------------------------------------

_SEASONAL_MES = {1: 0.9, 2: 0.85, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.0,
                 7: 1.15, 8: 1.25, 9: 0.95, 10: 1.0, 11: 1.1, 12: 1.85}
_CAMPANIA_MES = {12: "Navidad", 8: "Día del Niño", 5: "Aniversario ALDIMI"}
_BASE_DONACIONES_CAT = {
    "alimentos": 60, "higiene": 30, "insumos_medicos": 24, "medicamentos_soporte": 20,
}
_PRECIO_CAT = {  # soles por unidad donada (media lognormal)
    "alimentos": 9.0, "higiene": 7.5, "insumos_medicos": 18.0, "medicamentos_soporte": 32.0,
}


def generar_donaciones(rng: np.random.Generator, items: pd.DataFrame) -> pd.DataFrame:
    meses = pd.period_range("2022-09", periods=DONACIONES_MESES, freq="M")
    rows: list[dict] = []
    contador = 1
    for t, mes in enumerate(meses):
        seasonal = _SEASONAL_MES[mes.month]
        tendencia = 1.0 + 0.006 * t
        for categoria, base in _BASE_DONACIONES_CAT.items():
            lam = base * seasonal * tendencia * rng.lognormal(0, 0.05)
            n_don = max(int(rng.poisson(lam)), 1)
            cat_items = items[items["categoria"] == categoria]
            for _ in range(n_don):
                item = cat_items.iloc[int(rng.integers(0, len(cat_items)))]
                tipo = _choice(rng, TIPOS_DONANTE, 1)[0]
                escala_tipo = TIPOS_DONANTE[tipo][1] / 500.0
                cantidad = max(int(rng.lognormal(2.6, 0.35) * escala_tipo), 1)
                precio = _PRECIO_CAT[categoria] * rng.lognormal(0, 0.15)
                valor = round(cantidad * precio * seasonal, 2)
                dia = int(rng.integers(1, mes.days_in_month + 1))
                campania = _CAMPANIA_MES.get(mes.month, "Regular")
                if campania == "Regular" and rng.random() < 0.04:
                    campania = "Emergencia"
                rows.append({
                    "donacion_id": f"DON{contador:05d}",
                    "fecha": f"{mes.year}-{mes.month:02d}-{dia:02d}",
                    "tipo_donante": tipo,
                    "item_id": item["item_id"],
                    "nombre_item": item["nombre_item"],
                    "categoria": categoria,
                    "unidad_medida": item["unidad_medida"],
                    "cantidad_donada": cantidad,
                    "valor_estimado_soles": valor,
                    "campania": campania,
                })
                contador += 1
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Capturas OCR (contrato con el curso de IA)
# ----------------------------------------------------------------------------

_CAMPOS_POR_DOC = {
    "Receta médica": "medicamento|dosis|fecha|frecuencia",
    "DNI": "nombre|dni|fecha_nacimiento",
    "Hoja clínica": "diagnostico|indicaciones|fecha",
    "Orden de laboratorio": "examen|fecha|codigo",
    "Informe social": "resumen|evaluador|fecha",
}


def generar_capturas(rng: np.random.Generator, pacientes: pd.DataFrame, n: int = N_CAPTURAS) -> pd.DataFrame:
    tipo_doc = rng.choice(TIPOS_DOCUMENTO, size=n, p=[0.34, 0.18, 0.22, 0.16, 0.10])
    calidad = rng.choice(CALIDADES, size=n, p=[0.52, 0.33, 0.15])
    origen = rng.choice(ORIGENES_CAPTURA, size=n, p=[0.38, 0.44, 0.18])
    base_conf = np.where(calidad == "Alta", 0.90, np.where(calidad == "Media", 0.78, 0.60))
    ajuste_doc = np.where(tipo_doc == "DNI", 0.05, np.where(tipo_doc == "Informe social", -0.06, 0.0))
    confianza = np.clip(base_conf + ajuste_doc + rng.normal(0, 0.06, n), 0.2, 0.995).round(3)
    revision = ((confianza < 0.75) | (rng.random(n) < 0.05)).astype(int)
    fechas = pd.Timestamp("2025-06-01") + pd.to_timedelta(rng.integers(0, 420, n), unit="D")
    return pd.DataFrame({
        "captura_id": [f"OCR{i:05d}" for i in range(1, n + 1)],
        "fecha_captura": fechas.strftime("%Y-%m-%d"),
        "paciente_id": rng.choice(pacientes["paciente_id"].to_numpy(), size=n),
        "tipo_documento": tipo_doc,
        "calidad_imagen": calidad,
        "confianza_ocr": confianza,
        "campos_extraidos": [_CAMPOS_POR_DOC[d] for d in tipo_doc],
        "requiere_revision_manual": revision,
        "origen_captura": origen,
    })


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(seed: int = 42, data_dir: Path = DATA_DIR) -> None:
    rng = np.random.default_rng(seed)
    items = pd.read_csv(data_dir / "inventario_items.csv", encoding="utf-8-sig")

    print(f"Generando dataset v3 (seed={seed}) en {data_dir} ...")
    pacientes = generar_pacientes(rng)
    consumo = generar_consumo(rng, items)
    donaciones = generar_donaciones(rng, items)
    capturas = generar_capturas(rng, pacientes)

    pacientes.to_csv(data_dir / "pacientes_riesgo_sintetico.csv", index=False, encoding="utf-8-sig")
    consumo.to_csv(data_dir / "consumo_inventario_diario_sintetico.csv", index=False, encoding="utf-8-sig")
    donaciones.to_csv(data_dir / "donaciones_sinteticas.csv", index=False, encoding="utf-8-sig")
    capturas.to_csv(data_dir / "capturas_ia_sinteticas.csv", index=False, encoding="utf-8-sig")

    print(f"pacientes: {len(pacientes)} | prioridad: "
          f"{pacientes['nivel_prioridad_atencion'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"consumo: {len(consumo)} | critico 7d: {consumo['stock_critico_7d'].mean():.3f} "
          f"| 14d: {consumo['stock_critico_14d'].mean():.3f}")
    print(f"donaciones: {len(donaciones)} en {DONACIONES_MESES} meses")
    print(f"capturas: {len(capturas)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
