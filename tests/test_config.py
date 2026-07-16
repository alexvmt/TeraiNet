"""Tests for terainet.config module."""

from pathlib import Path

import pytest
import yaml

from terainet.config import load_config, load_training_config


def test_load_config(sample_config_path):
    """Test loading a YAML config file."""
    config = load_config(str(sample_config_path))

    assert config is not None
    assert "classes" in config
    assert config["classes"]["num_classes"] == 3
    assert config["classes"]["tiger"] == 1


def test_load_config_nonexistent():
    """Test loading a nonexistent config file raises error."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def _write_training_config(path: Path, classes: dict | None = None) -> None:
    """Write the smallest valid dedicated training configuration."""
    yaml.safe_dump(
        {
            "classes": classes
            or {
                "num_classes": 2,
                "tiger": 1,
                "leopard": 2,
            },
            "dataset": {
                "images_input_dir": "images",
                "images_filtered_dir": "filtered",
                "images_sampled_dir": "sampled",
                "class_directories": {"tiger": "class_1", "leopard": "class_2"},
                "subsets": {"train": "train", "val": "val", "test": "test"},
                "samples_per_class": {"train": 2, "val": 1, "test": 1},
            },
            "training": {
                "seed": 42,
                "image_size": 224,
                "batch_size": 2,
                "backbone": "EfficientNetV2B0",
                "epochs": 1,
                "learning_rate": 0.0001,
                "weight_decay": 0.0001,
                "early_stopping": {},
                "reduce_lr_on_plateau": {},
            },
            "artifacts": {
                "model": "artifacts/model.keras",
                "class_list": "artifacts/classes.yaml",
                "effective_config": "artifacts/config.yaml",
                "preparation_summary": "artifacts/preparation.yaml",
                "classification_report": "artifacts/report.csv",
                "confusion_matrix": "artifacts/matrix.png",
            },
        },
        path.open("w"),
    )


def test_load_training_config_orders_classes_and_resolves_paths(tmp_path):
    """Class indices, rather than YAML insertion order, determine the label contract."""
    config_path = tmp_path / "training.yaml"
    _write_training_config(
        config_path,
        {"num_classes": 2, "leopard": 2, "tiger": 1},
    )

    config = load_training_config(config_path)

    assert config.class_names == ("tiger", "leopard")
    assert config.dataset.class_directories == ("class_1", "class_2")
    assert config.dataset.images_input_dir == tmp_path / "images"
    assert config.artifact_paths["model"] == tmp_path / "artifacts" / "model.keras"


def test_load_training_config_rejects_non_contiguous_class_indices(tmp_path):
    """A malformed class mapping fails before TensorFlow can assign incorrect labels."""
    config_path = tmp_path / "training.yaml"
    _write_training_config(config_path, {"num_classes": 2, "tiger": 1, "leopard": 3})

    with pytest.raises(ValueError, match="contiguous"):
        load_training_config(config_path)


def test_load_training_config_rejects_missing_sample_subset(tmp_path):
    """All train, validation, and test sample counts are required."""
    config_path = tmp_path / "training.yaml"
    _write_training_config(config_path)
    config = yaml.safe_load(config_path.read_text())
    del config["dataset"]["samples_per_class"]["val"]
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="missing required subsets"):
        load_training_config(config_path)


def test_load_training_config_rejects_unknown_filter_exclusion(tmp_path):
    """Filtering exclusions must use configured class directory names."""
    config_path = tmp_path / "training.yaml"
    _write_training_config(config_path)
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["exclude_classes_from_filtering"] = ["class_99"]
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="unknown class directories"):
        load_training_config(config_path)


def test_load_training_config_defaults_cap_samples_to_available_to_true(tmp_path):
    """Sampling caps to the available image count by default when not configured."""
    config_path = tmp_path / "training.yaml"
    _write_training_config(config_path)

    config = load_training_config(config_path)

    assert config.dataset.cap_samples_to_available is True


def test_load_training_config_respects_cap_samples_to_available_override(tmp_path):
    """cap_samples_to_available can be disabled to require exact per-class counts."""
    config_path = tmp_path / "training.yaml"
    _write_training_config(config_path)
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["cap_samples_to_available"] = False
    config_path.write_text(yaml.safe_dump(config))

    config = load_training_config(config_path)

    assert config.dataset.cap_samples_to_available is False
