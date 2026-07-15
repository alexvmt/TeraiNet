"""Config-driven TensorFlow training pipeline for TeraiNet."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from terainet.config import TrainingConfig
from terainet.data import PreparationResult
from terainet.models import create_class_list_yaml_file


@dataclass(frozen=True)
class DatasetBundle:
    """Prepared TensorFlow datasets for a training run."""

    train: Any
    validation: Any
    test: Any
    ood: Any | None


@dataclass(frozen=True)
class ModelBuildResult:
    """A compiled-model-independent model and descriptive backbone metadata."""

    model: Any
    backbone_parameters: str
    frozen_file_size: str


@dataclass(frozen=True)
class TrainingRunResult:
    """Outputs of model training and artifact persistence."""

    model: Any
    history: Any
    training_time_minutes: float
    model_metadata: ModelBuildResult


_BACKBONES: dict[str, tuple[str, str, str]] = {
    "EfficientNetV2B0": ("EfficientNetV2B0", "5M", "26MB"),
    "EfficientNetV2B2": ("EfficientNetV2B2", "9M", "37MB"),
    "EfficientNetV2S": ("EfficientNetV2S", "21M", "84MB"),
    "EfficientNetV2M": ("EfficientNetV2M", "54M", "216MB"),
    "EfficientNetV2L": ("EfficientNetV2L", "119M", "475MB"),
    "EfficientNetV2XL": ("EfficientNetV2XL", "208M", "835MB"),
}


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow before building data or models."""
    import keras
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def _validate_dataset_directory(path: Path, expected_classes: tuple[str, ...]) -> None:
    if not path.is_dir():
        raise ValueError(f"Dataset directory not found: {path}")
    actual_classes = {child.name for child in path.iterdir() if child.is_dir()}
    expected_class_set = set(expected_classes)
    if actual_classes != expected_class_set:
        raise ValueError(
            f"Dataset directory {path} has class directories {sorted(actual_classes)}, "
            f"expected {sorted(expected_class_set)}."
        )


def _label_mode(num_classes: int) -> str:
    """Return the Keras dataset label mode for the configured number of classes."""
    return "binary" if num_classes == 2 else "categorical"


def _create_dataset(
    path: Path,
    class_directories: tuple[str, ...],
    config: TrainingConfig,
    shuffle: bool,
) -> Any:
    import keras

    _validate_dataset_directory(path, class_directories)
    return keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode=_label_mode(len(config.class_names)),
        class_names=list(class_directories),
        color_mode="rgb",
        batch_size=None,
        image_size=(config.image_size, config.image_size),
        shuffle=shuffle,
        seed=config.seed if shuffle else None,
    )


def build_augmentation_pipeline() -> Any:
    """Create the stochastic augmentation pipeline used only for training data."""
    import keras
    import tensorflow as tf

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
            keras.layers.Lambda(lambda images: tf.image.random_brightness(images, max_delta=0.1)),
            keras.layers.Lambda(
                lambda images: tf.image.random_saturation(images, lower=0.9, upper=1.1)
            ),
            keras.layers.Lambda(lambda images: tf.image.random_hue(images, max_delta=0.02)),
        ],
        name="training_augmentation",
    )


def _finalize_dataset(dataset: Any, config: TrainingConfig, augment: bool) -> Any:
    import tensorflow as tf

    if augment:
        augmentation = build_augmentation_pipeline()
        dataset = dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset = dataset.cache()

    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def _remap_ood_labels(
    dataset: Any,
    local_class_directories: tuple[str, ...],
    config: TrainingConfig,
) -> Any:
    """Map an OOD dataset's local labels to the global class index contract.

    Handles both the one-hot categorical label contract and the scalar binary label
    contract, matching whichever label mode `_create_dataset` used for this run.
    """
    import tensorflow as tf

    global_indices = [
        config.dataset.class_directories.index(name) for name in local_class_directories
    ]
    index_lookup = tf.constant(global_indices, dtype=tf.int32)
    num_classes = len(config.class_names)

    if num_classes == 2:
        return dataset.map(
            lambda images, labels: (
                images,
                tf.cast(tf.gather(index_lookup, tf.cast(labels, tf.int32)), tf.float32),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.map(
        lambda images, labels: (
            images,
            tf.one_hot(tf.gather(index_lookup, tf.argmax(labels, axis=-1)), num_classes),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def create_datasets(config: TrainingConfig, preparation: PreparationResult) -> DatasetBundle:
    """Load validated datasets with explicit class order and training-only augmentation."""
    train = _create_dataset(
        preparation.sampled_subset_paths["train"],
        config.dataset.class_directories,
        config,
        shuffle=True,
    )
    validation = _create_dataset(
        preparation.sampled_subset_paths["val"],
        config.dataset.class_directories,
        config,
        shuffle=False,
    )
    test = _create_dataset(
        preparation.sampled_subset_paths["test"],
        config.dataset.class_directories,
        config,
        shuffle=False,
    )

    ood = None
    ood_directory = config.dataset.subsets.get("ood")
    if ood_directory is not None:
        ood_path = config.dataset.images_input_dir / ood_directory
        ood_class_directories = tuple(
            class_directory
            for class_directory in config.dataset.class_directories
            if (ood_path / class_directory).is_dir()
        )
        if not ood_class_directories:
            raise ValueError(
                f"OOD dataset directory contains no configured class directories: {ood_path}"
            )
        ood = _create_dataset(ood_path, ood_class_directories, config, shuffle=False)
        ood = _remap_ood_labels(ood, ood_class_directories, config)

    return DatasetBundle(
        train=_finalize_dataset(train, config, augment=config.apply_augmentations),
        validation=_finalize_dataset(validation, config, augment=False),
        test=_finalize_dataset(test, config, augment=False),
        ood=_finalize_dataset(ood, config, augment=False) if ood is not None else None,
    )


def _classifier_head_config(num_classes: int) -> tuple[int, str]:
    """Return the (units, activation) for the final classification layer.

    Binary classification uses a single sigmoid unit; multiclass uses one softmax unit
    per class.
    """
    if num_classes == 2:
        return 1, "sigmoid"
    return num_classes, "softmax"


def build_model(config: TrainingConfig) -> ModelBuildResult:
    """Build the configured frozen EfficientNetV2 classification model."""
    import keras
    import kimm

    constructor_name, parameter_count, frozen_file_size = _BACKBONES[config.backbone]
    constructor = getattr(kimm.models, constructor_name)
    base_model = constructor(
        input_shape=(config.image_size, config.image_size, 3),
        include_preprocessing=True,
        include_top=False,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(config.image_size, config.image_size, 3))
    features = base_model(inputs, training=False)
    features = keras.layers.GlobalAveragePooling2D()(features)
    features = keras.layers.Dropout(config.dropout_rate)(features)
    units, activation = _classifier_head_config(len(config.class_names))
    outputs = keras.layers.Dense(units, activation=activation)(features)
    model = keras.Model(inputs, outputs, name="terainet_classifier")
    return ModelBuildResult(model, parameter_count, frozen_file_size)


def compile_model(model: Any, config: TrainingConfig) -> None:
    """Compile a classifier from the configured optimizer and loss settings.

    Uses binary cross-entropy for a 2-class configuration and categorical
    cross-entropy otherwise, matching the classification head built by `build_model`.
    """
    import keras

    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        amsgrad=config.amsgrad,
        weight_decay=config.weight_decay,
    )
    if len(config.class_names) == 2:
        loss = keras.losses.BinaryCrossentropy(label_smoothing=config.label_smoothing)
    else:
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=config.label_smoothing)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )


def _create_callbacks(config: TrainingConfig) -> list[Any]:
    import keras

    return [
        keras.callbacks.EarlyStopping(verbose=1, **config.early_stopping),
        keras.callbacks.ReduceLROnPlateau(verbose=1, **config.reduce_lr_on_plateau),
    ]


def _save_run_contract(config: TrainingConfig, preparation: PreparationResult) -> None:
    effective_config_path = config.artifact_paths["effective_config"]
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    with effective_config_path.open("w") as file:
        yaml.safe_dump(
            {
                "class_names": list(config.class_names),
                "class_directories": list(config.dataset.class_directories),
                "image_size": config.image_size,
                "seed": config.seed,
                "backbone": config.backbone,
                "preprocessing": {
                    "resize": [config.image_size, config.image_size],
                    "channels": "rgb",
                },
            },
            file,
            sort_keys=False,
        )

    preparation_path = config.artifact_paths["preparation_summary"]
    preparation_path.parent.mkdir(parents=True, exist_ok=True)
    with preparation_path.open("w") as file:
        yaml.safe_dump(
            {
                "dataset_root": str(preparation.dataset_root),
                "sampled_subset_paths": {
                    name: str(path) for name, path in preparation.sampled_subset_paths.items()
                },
                "samples_per_subset": preparation.samples_per_subset,
                "filter_stats": preparation.filter_stats,
            },
            file,
            sort_keys=False,
        )


def train_and_save(
    model_result: ModelBuildResult,
    datasets: DatasetBundle,
    config: TrainingConfig,
    preparation: PreparationResult,
) -> TrainingRunResult:
    """Compile, train, and save the configured model and run-contract artifacts."""
    import keras

    compile_model(model_result.model, config)
    start_time = time.monotonic()
    history = model_result.model.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=config.epochs,
        callbacks=_create_callbacks(config),
    )
    training_time_minutes = round((time.monotonic() - start_time) / 60, 2)

    model_path = config.artifact_paths["model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    keras.saving.save_model(model_result.model, model_path, include_optimizer=False)
    create_class_list_yaml_file(
        len(config.class_names),
        list(config.class_names),
        config.artifact_paths["class_list"],
    )
    _save_run_contract(config, preparation)

    return TrainingRunResult(
        model=model_result.model,
        history=history,
        training_time_minutes=training_time_minutes,
        model_metadata=model_result,
    )
