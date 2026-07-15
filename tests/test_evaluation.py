"""Tests for TeraiNet evaluation utilities."""

import numpy as np
import pytest

from terainet.evaluation import evaluate_dataset


class _TensorLike:
    """Minimal tensor stand-in for a CPU-only unit test."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class _Model:
    """Return supplied predictions in the same order as test batches."""

    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def predict(self, images, verbose=0):
        return next(self.predictions)


def test_evaluate_dataset_keeps_global_class_order_when_a_class_is_absent():
    """Reports retain configured classes even when test support is zero."""
    dataset = [
        (
            np.zeros((2, 1)),
            _TensorLike([[1, 0, 0], [0, 1, 0]]),
        )
    ]
    model = _Model([np.asarray([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]])])

    result = evaluate_dataset(model, dataset, ("tiger", "leopard", "bird"), name="test")

    assert result.metrics["accuracy"] == 1.0
    assert list(result.report.index[:3]) == ["tiger", "leopard", "bird"]
    assert result.report.loc["bird", "support"] == 0.0
    assert result.confusion_matrix.shape == (3, 3)


def test_evaluate_dataset_restricts_macro_metrics_to_observed_classes():
    """A single-class OOD split isn't diluted by configured classes with no support."""
    # All ground truth is "tiger" (index 0); one sample is misclassified as "bird".
    dataset = [
        (
            np.zeros((2, 1)),
            _TensorLike([[1, 0, 0], [1, 0, 0]]),
        )
    ]
    model = _Model([np.asarray([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])])

    result = evaluate_dataset(
        model,
        dataset,
        ("tiger", "leopard", "bird"),
        name="ood",
        restrict_macro_to_observed_classes=True,
    )

    # Macro metrics restricted to the single observed class ("tiger") collapse to its
    # own recall/precision/f1, instead of being averaged in with two always-absent
    # classes that would otherwise force a recall of 0.
    assert result.metrics["accuracy"] == 0.5
    assert result.metrics["recall_macro"] == 0.5
    # The full report and confusion matrix still retain every configured class.
    assert list(result.report.index[:3]) == ["tiger", "leopard", "bird"]
    assert result.confusion_matrix.shape == (3, 3)


def test_evaluate_dataset_handles_binary_sigmoid_predictions():
    """A single sigmoid output is thresholded at 0.5 instead of using argmax."""
    dataset = [
        (
            np.zeros((3, 1)),
            _TensorLike([0.0, 1.0, 1.0]),
        )
    ]
    model = _Model([np.asarray([[0.2], [0.7], [0.4]])])

    result = evaluate_dataset(model, dataset, ("negative", "positive"), name="test")

    assert result.metrics["accuracy"] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(result.predicted_labels, [0, 1, 0])
