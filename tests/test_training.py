"""Tests for pure helper logic in terainet.training.

These cover only the dependency-free helpers (label mode and classifier head
selection). `build_model`/`compile_model`/`create_datasets` require TensorFlow, Keras,
and a downloaded backbone, so they are intentionally left to manual/notebook
verification rather than unit tests.
"""

from terainet.training import _classifier_head_config, _label_mode


def test_label_mode_is_binary_for_two_classes():
    """Two classes use Keras' scalar binary label mode."""
    assert _label_mode(2) == "binary"


def test_label_mode_is_categorical_for_more_than_two_classes():
    """More than two classes use one-hot categorical labels."""
    assert _label_mode(3) == "categorical"
    assert _label_mode(10) == "categorical"


def test_classifier_head_config_is_single_sigmoid_for_two_classes():
    """Binary classification uses a single sigmoid output unit."""
    assert _classifier_head_config(2) == (1, "sigmoid")


def test_classifier_head_config_is_softmax_per_class_otherwise():
    """Multiclass classification uses one softmax unit per class."""
    assert _classifier_head_config(3) == (3, "softmax")
    assert _classifier_head_config(10) == (10, "softmax")
