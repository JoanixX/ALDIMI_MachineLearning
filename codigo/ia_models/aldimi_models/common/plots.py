"""Generacion de graficos con matplotlib. Modulo compartido unico.

Todas las funciones guardan el PNG en disco y cierran la figura para poder
ejecutarse en modo batch (sin display).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_chart(
    labels: Sequence[str], values: Sequence[float], title: str,
    xlabel: str, ylabel: str, path: Path, rotate_xticks: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate_xticks:
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    _save(fig, path)


def histogram(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series.dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    _save(fig, path)


def boxplot_by_group(
    df: pd.DataFrame, value_col: str, group_col: str,
    groups: Sequence[str], title: str, path: Path,
) -> None:
    data = [df.loc[df[group_col] == g, value_col].dropna() for g in groups]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=list(groups))
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    _save(fig, path)


def line_chart(
    x: Sequence, y: Sequence[float], title: str, xlabel: str, ylabel: str, path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, path)


def scatter_chart(
    x: Sequence[float], y: Sequence[float], title: str, xlabel: str, ylabel: str, path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, path)


def correlation_heatmap(corr: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(corr, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title(title)
    _save(fig, path)


def confusion_matrix_heatmap(cm_df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_df.values, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            value = int(cm_df.iloc[i, j])
            threshold = cm_df.values.max() / 2 if cm_df.values.max() > 0 else 0
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > threshold else "black")
    ax.set_title(title)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    _save(fig, path)


def feature_importance_barh(importance_df: pd.DataFrame, title: str, path: Path, top_n: int = 20) -> None:
    """Espera columnas 'variable' e 'importancia' ordenadas descendentemente."""
    top = importance_df.head(top_n).sort_values("importancia", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["variable"], top["importancia"])
    ax.set_title(title)
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    _save(fig, path)


def predictions_vs_actual(
    y_true: Sequence[float], y_pred: Sequence[float], title: str, path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, alpha=0.4)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predicho")
    _save(fig, path)
