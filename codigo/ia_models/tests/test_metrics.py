"""Tests de que las metricas de evaluacion se generan sin error."""

import math

from aldimi_models.common.metrics import (
    classification_metrics,
    classification_report_frame,
    confusion_matrix_frame,
    regression_metrics,
)


def test_classification_metrics_keys_and_ranges() -> None:
    y_true = [0, 1, 2, 1, 0, 2, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 2, 0, 0]
    metrics = classification_metrics(y_true, y_pred)
    expected_keys = {
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    }
    assert set(metrics) == expected_keys
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_perfect_classification_gives_f1_macro_one() -> None:
    y = [0, 1, 2, 0, 1, 2]
    assert classification_metrics(y, y)["f1_macro"] == 1.0


def test_regression_metrics_values() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.5, 2.0, 2.5, 4.0]
    metrics = regression_metrics(y_true, y_pred)
    assert math.isclose(metrics["mae"], 0.25)
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["r2"] <= 1.0


def test_confusion_matrix_frame_shape_and_labels() -> None:
    cm = confusion_matrix_frame([0, 1, 1, 0], [0, 1, 0, 0], labels=[0, 1])
    assert cm.shape == (2, 2)
    assert list(cm.index) == ["Real_0", "Real_1"]
    assert int(cm.values.sum()) == 4


def test_classification_report_frame_has_macro_avg() -> None:
    report = classification_report_frame([0, 1, 1], [0, 1, 0])
    assert "macro avg" in report.index
