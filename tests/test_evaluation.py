"""Tests for TeraiNet evaluation utilities."""

import numpy as np

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
