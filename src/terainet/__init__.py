"""TeraiNet: Wildlife species classification using deep learning."""

from terainet.config import TrainingConfig, load_config, load_training_config
from terainet.data import (
    PreparationResult,
    add_subset_column,
    check_location_split,
    filter_single_snippet_images,
    prepare_training_data,
    sample_images,
    sample_n_images_per_species,
    sample_validation_images_excluding_raw_prefixes,
)
from terainet.detection import contains_animal
from terainet.evaluation import evaluate_dataset, log_run_to_wandb, save_evaluation_artifacts
from terainet.models import create_class_list_yaml_file
from terainet.training import (
    DatasetBundle,
    ModelBuildResult,
    TrainingRunResult,
    build_model,
    create_datasets,
    set_random_seed,
    train_and_save,
)

__all__ = [
    "load_config",
    "load_training_config",
    "TrainingConfig",
    "sample_n_images_per_species",
    "add_subset_column",
    "check_location_split",
    "filter_single_snippet_images",
    "sample_validation_images_excluding_raw_prefixes",
    "prepare_training_data",
    "PreparationResult",
    "contains_animal",
    "create_class_list_yaml_file",
    "sample_images",
    "set_random_seed",
    "create_datasets",
    "DatasetBundle",
    "build_model",
    "ModelBuildResult",
    "train_and_save",
    "TrainingRunResult",
    "evaluate_dataset",
    "save_evaluation_artifacts",
    "log_run_to_wandb",
]
