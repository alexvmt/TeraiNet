"""Configuration utilities for TeraiNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Parameters:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


@dataclass(frozen=True)
class DatasetPreparationConfig:
    """Dataset preparation settings for a training run."""

    images_input_dir: Path
    images_filtered_dir: Path
    images_sampled_dir: Path
    class_directories: tuple[str, ...]
    subsets: dict[str, str]
    samples_per_class: dict[str, int]
    filter_single_snippets: bool
    exclude_classes_from_filtering: tuple[str, ...]
    raw_prefix_validation_sampling: dict[str, Any] | None


@dataclass(frozen=True)
class TrainingConfig:
    """Validated configuration for a complete TeraiNet training run."""

    path: Path
    class_names: tuple[str, ...]
    dataset: DatasetPreparationConfig
    image_size: int
    batch_size: int
    seed: int
    apply_augmentations: bool
    backbone: str
    dropout_rate: float
    epochs: int
    learning_rate: float
    weight_decay: float
    amsgrad: bool
    label_smoothing: float
    early_stopping: dict[str, Any]
    reduce_lr_on_plateau: dict[str, Any]
    artifact_paths: dict[str, Path]
    wandb: dict[str, Any]


def _resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_dir / path


def _ordered_class_names(classes: dict[str, Any]) -> tuple[str, ...]:
    class_indices = {name: index for name, index in classes.items() if name != "num_classes"}
    if not class_indices:
        raise ValueError("Training configuration must define at least one class.")

    if not all(
        isinstance(index, int) and not isinstance(index, bool) for index in class_indices.values()
    ):
        raise ValueError("Training class indices must be integers.")

    expected_indices = list(range(1, len(class_indices) + 1))
    if sorted(class_indices.values()) != expected_indices:
        raise ValueError("Training class indices must be unique and contiguous starting at 1.")

    num_classes = classes.get("num_classes")
    if num_classes != len(class_indices):
        raise ValueError("classes.num_classes must match the number of configured classes.")

    return tuple(name for name, _ in sorted(class_indices.items(), key=lambda item: item[1]))


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Training configuration requires a '{key}' mapping.")
    return value


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """Load and validate the dedicated training configuration YAML file."""
    path = Path(config_path).resolve()
    with path.open("r") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    class_names = _ordered_class_names(_require_mapping(raw_config, "classes"))
    dataset_config = _require_mapping(raw_config, "dataset")
    subset_dirs = _require_mapping(dataset_config, "subsets")
    sample_counts = _require_mapping(dataset_config, "samples_per_class")
    training_config = _require_mapping(raw_config, "training")
    artifact_config = _require_mapping(raw_config, "artifacts")

    required_subsets = {"train", "val", "test"}
    missing_subsets = required_subsets - set(subset_dirs)
    if missing_subsets:
        raise ValueError(f"dataset.subsets is missing required subsets: {sorted(missing_subsets)}")
    missing_sample_subsets = required_subsets - set(sample_counts)
    if missing_sample_subsets:
        raise ValueError(
            "dataset.samples_per_class is missing required subsets: "
            f"{sorted(missing_sample_subsets)}"
        )
    unknown_sample_subsets = set(sample_counts) - required_subsets
    if unknown_sample_subsets:
        raise ValueError(
            "dataset.samples_per_class may only configure train, val, and test subsets."
        )
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in sample_counts.values()
    ):
        raise ValueError("dataset.samples_per_class values must be non-negative integers.")

    class_directories = _require_mapping(dataset_config, "class_directories")
    if set(class_directories) != set(class_names):
        raise ValueError("dataset.class_directories must map every configured class exactly once.")
    ordered_class_directories = tuple(class_directories[class_name] for class_name in class_names)
    if len(set(ordered_class_directories)) != len(ordered_class_directories):
        raise ValueError("dataset.class_directories values must be unique.")

    excluded_class_directories = dataset_config.get("exclude_classes_from_filtering", [])
    if not isinstance(excluded_class_directories, list) or not all(
        isinstance(class_directory, str) for class_directory in excluded_class_directories
    ):
        raise ValueError("dataset.exclude_classes_from_filtering must be a list of strings.")
    unknown_excluded_classes = set(excluded_class_directories) - set(ordered_class_directories)
    if unknown_excluded_classes:
        raise ValueError(
            "dataset.exclude_classes_from_filtering contains unknown class directories: "
            f"{sorted(unknown_excluded_classes)}"
        )

    raw_prefix_policy = dataset_config.get("raw_prefix_validation_sampling")
    if raw_prefix_policy is not None:
        if not isinstance(raw_prefix_policy, dict):
            raise ValueError("raw_prefix_validation_sampling must be a mapping when configured.")
        class_name = raw_prefix_policy.get("class_name")
        if class_name not in class_names:
            raise ValueError(
                "raw_prefix_validation_sampling.class_name must be a configured class."
            )
        required_policy_keys = {"class_name", "source_subset", "target_subset"}
        missing_policy_keys = required_policy_keys - set(raw_prefix_policy)
        if missing_policy_keys:
            raise ValueError(
                "raw_prefix_validation_sampling is missing required keys: "
                f"{sorted(missing_policy_keys)}"
            )
        if raw_prefix_policy["source_subset"] not in subset_dirs:
            raise ValueError("raw_prefix_validation_sampling.source_subset is not configured.")
        if raw_prefix_policy["target_subset"] not in sample_counts:
            raise ValueError(
                "raw_prefix_validation_sampling.target_subset must be a sampled subset."
            )

    config_dir = path.parent
    dataset = DatasetPreparationConfig(
        images_input_dir=_resolve_path(dataset_config["images_input_dir"], config_dir),
        images_filtered_dir=_resolve_path(dataset_config["images_filtered_dir"], config_dir),
        images_sampled_dir=_resolve_path(dataset_config["images_sampled_dir"], config_dir),
        class_directories=ordered_class_directories,
        subsets={name: str(directory) for name, directory in subset_dirs.items()},
        samples_per_class={name: int(count) for name, count in sample_counts.items()},
        filter_single_snippets=bool(dataset_config.get("filter_single_snippets", False)),
        exclude_classes_from_filtering=tuple(excluded_class_directories),
        raw_prefix_validation_sampling=raw_prefix_policy,
    )

    image_size = training_config["image_size"]
    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]
    for name, value in {
        "training.image_size": image_size,
        "training.batch_size": batch_size,
        "training.epochs": epochs,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    backbone = training_config["backbone"]
    supported_backbones = {
        "EfficientNetV2B0",
        "EfficientNetV2B2",
        "EfficientNetV2S",
        "EfficientNetV2M",
        "EfficientNetV2L",
        "EfficientNetV2XL",
    }
    if backbone not in supported_backbones:
        raise ValueError(f"training.backbone must be one of {sorted(supported_backbones)}.")

    artifact_paths = {
        name: _resolve_path(value, config_dir) for name, value in artifact_config.items()
    }
    required_artifacts = {
        "model",
        "class_list",
        "effective_config",
        "preparation_summary",
        "classification_report",
        "confusion_matrix",
    }
    missing_artifacts = required_artifacts - set(artifact_paths)
    if missing_artifacts:
        raise ValueError(f"artifacts is missing required paths: {sorted(missing_artifacts)}")
    if len(set(artifact_paths.values())) != len(artifact_paths):
        raise ValueError("Artifact paths must be distinct.")

    return TrainingConfig(
        path=path,
        class_names=class_names,
        dataset=dataset,
        image_size=image_size,
        batch_size=batch_size,
        seed=int(training_config["seed"]),
        apply_augmentations=bool(training_config.get("apply_augmentations", True)),
        backbone=backbone,
        dropout_rate=float(training_config.get("dropout_rate", 0.2)),
        epochs=epochs,
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        amsgrad=bool(training_config.get("amsgrad", True)),
        label_smoothing=float(training_config.get("label_smoothing", 0.0)),
        early_stopping=dict(_require_mapping(training_config, "early_stopping")),
        reduce_lr_on_plateau=dict(_require_mapping(training_config, "reduce_lr_on_plateau")),
        artifact_paths=artifact_paths,
        wandb=dict(raw_config.get("wandb", {})),
    )
