"""Data utilities for TeraiNet."""

from __future__ import annotations

import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparationResult:
    """Paths and retention statistics produced by training-data preparation."""

    dataset_root: Path
    sampled_subset_paths: dict[str, Path]
    filter_stats: dict[str, dict[str, Any]] | None
    samples_per_subset: dict[str, dict[str, int]]


def sample_n_images_per_species(
    df: pl.DataFrame, species_samples_dict: dict, column_to_filter: str, seed: int = 42
) -> pl.DataFrame:
    """
    Filter and randomly sample a specified number of rows for each species.

    This function filters a Polars DataFrame by species and randomly samples a specified
    number of rows for each species, useful for creating balanced datasets.

    Parameters:
        df (pl.DataFrame): A Polars DataFrame containing image data and species labels.
        species_samples_dict (dict): A dictionary where keys are species names,
            and values are the number of samples to draw for each species.
        column_to_filter (str): The column name used for filtering species.
        seed (int, optional): Seed for reproducibility. Default is 42.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the sampled rows for each species.
    """
    sampled_dfs = []

    for species, sample_size in species_samples_dict.items():
        species_df = df.filter(pl.col(column_to_filter) == species)
        n = min(sample_size, species_df.height)

        if n > 0:
            sampled_df = species_df.sample(n=n, seed=seed, with_replacement=False)
            sampled_dfs.append(sampled_df)

    return pl.concat(sampled_dfs) if sampled_dfs else pl.DataFrame()


def add_subset_column(
    df: pl.DataFrame, train_ratio: float, seed: int = 42, tolerance: float = 0.02
) -> pl.DataFrame:
    """
    Add a 'subset' column splitting data into 'train', 'val', and 'test' subsets.

    Splits data based on unique 'location_id' to prevent spatial leakage (no location_id
    appears in more than one subset). Uses soft balancing with tolerance on target ratios.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame containing 'location_id' column.
        train_ratio (float): Target ratio for the 'train' subset (e.g., 0.8 for 80% train).
            The remaining portion is split equally into 'val' and 'test'.
        seed (int, optional): Seed for reproducibility. Default is 42.
        tolerance (float, optional): Acceptable deviation from target ratios (e.g., 0.02 for ±2%).

    Returns:
        pl.DataFrame: The DataFrame with an added 'subset' column containing
            'train', 'val', 'test', or 'unknown' values.

    Raises:
        ValueError: If `train_ratio` is not between 0 and 1.
    """
    if not 0 <= train_ratio <= 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    val_test_ratio = (1 - train_ratio) / 2
    target_ratios = {"train": train_ratio, "val": val_test_ratio, "test": val_test_ratio}
    upper_bounds = {k: v + tolerance for k, v in target_ratios.items()}

    total_images = df.shape[0]

    unique_locs = df.select("location_id").unique()
    loc_list = unique_locs["location_id"].to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(loc_list)

    image_counts = df.group_by("location_id").len().rename({"len": "n_images"})
    loc_to_n = dict(
        zip(
            image_counts["location_id"].to_list(),
            image_counts["n_images"].to_list(),
            strict=False,
        )
    )

    subsets = {"train": set(), "val": set(), "test": set()}
    counts = {"train": 0, "val": 0, "test": 0}

    for loc in loc_list:
        loc_n = loc_to_n[loc]
        current_ratios = {k: counts[k] / total_images for k in counts}
        candidates = [k for k in current_ratios if current_ratios[k] < upper_bounds[k]]
        if not candidates:
            candidates = sorted(current_ratios.items(), key=lambda x: x[1])
            chosen_subset = candidates[0][0]
        else:
            min_ratio = min(current_ratios[c] for c in candidates)
            min_candidates = [c for c in candidates if current_ratios[c] == min_ratio]
            chosen_subset = rng.choice(min_candidates)

        subsets[chosen_subset].add(loc)
        counts[chosen_subset] += loc_n

    df = df.with_columns(
        pl.when(pl.col("location_id").is_in(subsets["train"]))
        .then(pl.lit("train"))
        .when(pl.col("location_id").is_in(subsets["val"]))
        .then(pl.lit("val"))
        .when(pl.col("location_id").is_in(subsets["test"]))
        .then(pl.lit("test"))
        .otherwise(pl.lit("unknown"))
        .alias("subset")
    )

    if df.filter(pl.col("subset") == "unknown").height > 0:
        logger.warning("Some locations could not be assigned to a subset.")

    final_counts = df.group_by("subset").len().rename({"len": "count"}).sort("subset")
    logger.info(f"Subset distribution:\n{final_counts}")

    return df


def check_location_split(df: pl.DataFrame) -> None:
    """
    Check whether any 'location_id' appears in more than one subset.

    Ensures spatial leakage prevention: each location should appear in exactly one
    of 'train', 'val', or 'test' subsets.

    Parameters:
        df (pl.DataFrame): The Polars DataFrame containing 'location_id' and 'subset' columns.

    Raises:
        ValueError: If the DataFrame does not contain required columns.

    Prints:
        - A success message if there are no violations.
        - Otherwise, prints the violating 'location_id's and their subset distribution.
    """
    if "location_id" not in df.columns or "subset" not in df.columns:
        raise ValueError("DataFrame must contain location_id and subset columns.")

    violations = (
        df.select(["location_id", "subset"])
        .unique()
        .group_by("location_id")
        .agg(pl.col("subset").n_unique().alias("num_subsets"))
        .filter(pl.col("num_subsets") > 1)
    )

    if violations.is_empty():
        logger.info("No violations found: each location_id appears in only one subset.")
    else:
        logger.warning(f"Violations found in {violations.height} location_id(s):")
        logger.warning(f"{violations}")


def sample_images(
    source_dir: str | Path,
    target_dir: str | Path,
    samples_per_class: int,
    seed: int = 42,
    class_directories: tuple[str, ...] | None = None,
) -> None:
    """
    Sample a fixed number of images per class from a directory structure.

    Expects source_dir to contain subdirectories (one per class), each containing images.
    Samples `samples_per_class` images from each class and copies them to target_dir,
    maintaining the class directory structure.

    Parameters:
        source_dir (str): Path to the source dataset directory with class subdirectories.
        target_dir (str): Path to the target dataset directory to store sampled data.
        samples_per_class (int): Number of images to sample per class.
        seed (int): Random seed for reproducibility. Default is 42.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    if not source_path.is_dir():
        raise ValueError(f"Source directory not found: {source_path}")
    if samples_per_class < 0:
        raise ValueError("samples_per_class must be non-negative.")
    if target_path.exists() and any(target_path.iterdir()):
        raise FileExistsError(f"Target directory already exists and is not empty: {target_path}")
    target_path.mkdir(parents=True, exist_ok=True)

    expected_class_directories = class_directories or tuple(
        path.name for path in sorted(source_path.iterdir()) if path.is_dir()
    )
    missing_classes = [
        class_name
        for class_name in expected_class_directories
        if not (source_path / class_name).is_dir()
    ]
    if missing_classes:
        raise ValueError(f"Source directory is missing class directories: {missing_classes}")

    rng = random.Random(seed)
    for class_name in expected_class_directories:
        class_dir = source_path / class_name
        sampled_class_dir = target_path / class_name
        sampled_class_dir.mkdir(parents=True, exist_ok=True)

        all_images = sorted(path for path in class_dir.iterdir() if path.is_file())
        if len(all_images) < samples_per_class:
            raise ValueError(
                f"Class '{class_name}' has {len(all_images)} images, "
                f"but {samples_per_class} were requested."
            )
        sampled_images = rng.sample(all_images, samples_per_class)

        for image_path in sampled_images:
            target_image_path = sampled_class_dir / image_path.name
            shutil.copy2(image_path, target_image_path)


def sample_validation_images_excluding_raw_prefixes(
    source_dir: str | Path,
    training_dir: str | Path,
    target_dir: str | Path,
    samples_per_class: int,
    seed: int,
    delimiter: str = "-",
) -> None:
    """Sample validation images whose raw-image prefixes are absent from training."""
    source_path = Path(source_dir)
    training_path = Path(training_dir)
    target_path = Path(target_dir)
    if not source_path.is_dir() or not training_path.is_dir():
        raise ValueError("Source and training directories must exist for raw-prefix sampling.")
    if target_path.exists() and any(target_path.iterdir()):
        raise FileExistsError(f"Target directory already exists and is not empty: {target_path}")

    def raw_prefix(path: Path) -> str:
        return path.stem.split(delimiter, maxsplit=1)[0]

    training_prefixes = {raw_prefix(path) for path in training_path.iterdir() if path.is_file()}
    eligible_paths = [
        path
        for path in sorted(source_path.iterdir())
        if path.is_file() and raw_prefix(path) not in training_prefixes
    ]
    if len(eligible_paths) < samples_per_class:
        raise ValueError(
            f"Only {len(eligible_paths)} raw-prefix-disjoint validation images are available, "
            f"but {samples_per_class} were requested."
        )

    target_path.mkdir(parents=True, exist_ok=True)
    for image_path in random.Random(seed).sample(eligible_paths, samples_per_class):
        shutil.copy2(image_path, target_path / image_path.name)

    sampled_prefixes = {raw_prefix(path) for path in target_path.iterdir() if path.is_file()}
    if training_prefixes & sampled_prefixes:
        raise RuntimeError("Raw-image prefixes overlap between training and validation samples.")


def prepare_training_data(config: Any) -> PreparationResult:
    """Filter and sample configured data without changing held-out OOD data."""
    dataset = config.dataset
    sampled_root = dataset.images_sampled_dir
    if sampled_root.exists() and any(sampled_root.iterdir()):
        raise FileExistsError(
            f"Sampled dataset directory already exists and is not empty: {sampled_root}"
        )

    filter_stats = None
    dataset_root = dataset.images_input_dir
    if dataset.filter_single_snippets:
        filter_stats = filter_single_snippet_images(
            str(dataset.images_input_dir),
            str(dataset.images_filtered_dir),
            exclude_classes=list(dataset.exclude_classes_from_filtering),
        )
        dataset_root = dataset.images_filtered_dir

    sampled_subset_paths: dict[str, Path] = {}
    samples_per_subset: dict[str, dict[str, int]] = {}
    for subset in ("train", "val", "test"):
        source_path = dataset_root / dataset.subsets[subset]
        target_path = sampled_root / dataset.subsets[subset]
        sample_images(
            source_path,
            target_path,
            dataset.samples_per_class[subset],
            seed=config.seed,
            class_directories=dataset.class_directories,
        )
        sampled_subset_paths[subset] = target_path
        samples_per_subset[subset] = {
            class_directory: dataset.samples_per_class[subset]
            for class_directory in dataset.class_directories
        }

    policy = dataset.raw_prefix_validation_sampling
    if policy is not None:
        class_index = config.class_names.index(policy["class_name"])
        class_directory = dataset.class_directories[class_index]
        target_path = sampled_subset_paths[policy["target_subset"]] / class_directory
        shutil.rmtree(target_path)
        sample_validation_images_excluding_raw_prefixes(
            dataset.images_input_dir / dataset.subsets[policy["source_subset"]] / class_directory,
            sampled_subset_paths["train"] / class_directory,
            target_path,
            dataset.samples_per_class[policy["target_subset"]],
            seed=config.seed,
            delimiter=str(policy.get("delimiter", "-")),
        )

    return PreparationResult(
        dataset_root=dataset_root,
        sampled_subset_paths=sampled_subset_paths,
        filter_stats=filter_stats,
        samples_per_subset=samples_per_subset,
    )


def filter_single_snippet_images(
    data_dir: str,
    target_dir: str,
    exclude_classes: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Filter dataset to keep only images with single snippet extraction.

    From a source directory structure with train/val/test subdirectories containing class
    subdirectories, keeps only files ending in -0 (indicating single snippet extraction).
    Files with -1, -2, etc. are removed as they represent multiple snippets from the
    same original image, leading to potentially duplicate or mislabeled data.

    Copies filtered images to target_dir, which can be on a different filesystem
    (useful for Kaggle where input is read-only).

    Excluded classes are copied unchanged.

    Parameters:
        data_dir:
            Source root directory (read-only) containing train/val/test subdirectories.

        target_dir:
            Target root directory (writable) where filtered images will be copied.

        exclude_classes:
            Class subdirectory names to skip filtering.
            These classes are copied unchanged.

    Returns:
        Dictionary containing filtering statistics per subset and class.

    Raises:
        ValueError:
            If data_dir does not exist.
    """
    exclude_classes_set = set(exclude_classes or [])
    source_path = Path(data_dir)
    target_root = Path(target_dir)

    if not source_path.exists():
        raise ValueError(f"Source data directory not found: {data_dir}")

    target_root.mkdir(parents=True, exist_ok=True)

    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"Target directory already exists and is not empty: {target_root}")

    stats: dict[str, dict[str, Any]] = {}
    subsets = ["train", "val", "test"]

    for subset in subsets:
        subset_source = source_path / subset

        if not subset_source.exists() or not subset_source.is_dir():
            logger.warning(
                "Skipping subset '%s': directory not found",
                subset,
            )
            continue

        subset_target = target_root / subset
        subset_target.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting filtering for subset '%s'",
            subset,
        )

        subset_stats: dict[str, Any] = {
            "per_class": {},
            "original_total": 0,
            "filtered_total": 0,
            "removed_total": 0,
        }

        try:
            # Process each class directory
            for class_dir in sorted(subset_source.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                target_class_dir = subset_target / class_name
                target_class_dir.mkdir(parents=True, exist_ok=True)

                original_count = 0
                filtered_count = 0
                excluded = class_name in exclude_classes_set

                logger.info(
                    "Processing subset='%s', class='%s', mode='%s'",
                    subset,
                    class_name,
                    "copy_unchanged" if excluded else "filter_single_snippet",
                )

                # Streaming iteration instead of loading all files into memory
                for file_path in class_dir.iterdir():
                    if not file_path.is_file():
                        continue

                    original_count += 1

                    should_copy = excluded or file_path.stem.endswith("-0")

                    if should_copy:
                        shutil.copy2(
                            file_path,
                            target_class_dir / file_path.name,
                        )
                        filtered_count += 1

                removed_count = original_count - filtered_count

                subset_stats["per_class"][class_name] = {
                    "original": original_count,
                    "filtered": filtered_count,
                    "removed": removed_count,
                }

                subset_stats["original_total"] += original_count
                subset_stats["filtered_total"] += filtered_count

                retention_pct = (
                    (filtered_count / original_count * 100) if original_count > 0 else 0.0
                )

                logger.info(
                    (
                        "Completed subset='%s', class='%s': "
                        "original=%d, filtered=%d, removed=%d, retention=%.2f%%"
                    ),
                    subset,
                    class_name,
                    original_count,
                    filtered_count,
                    removed_count,
                    retention_pct,
                )

            subset_stats["removed_total"] = (
                subset_stats["original_total"] - subset_stats["filtered_total"]
            )

            total_retention_pct = (
                (subset_stats["filtered_total"] / subset_stats["original_total"] * 100)
                if subset_stats["original_total"] > 0
                else 0.0
            )

            logger.info(
                ("Finished subset='%s': original=%d, filtered=%d, removed=%d, retention=%.2f%%"),
                subset,
                subset_stats["original_total"],
                subset_stats["filtered_total"],
                subset_stats["removed_total"],
                total_retention_pct,
            )

            stats[subset] = subset_stats

        except Exception:
            logger.exception(
                "Filtering failed for subset '%s'",
                subset,
            )
            raise

    return stats
