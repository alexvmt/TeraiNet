"""Data utilities for TeraiNet."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


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


def sample_images(source_dir: str, target_dir: str, samples_per_class: int, seed: int = 42) -> None:
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
    random.seed(seed)

    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    for class_dir in source_path.iterdir():
        if class_dir.is_dir():
            sampled_class_dir = target_path / class_dir.name
            sampled_class_dir.mkdir(parents=True, exist_ok=True)

            all_images = list(class_dir.iterdir())
            random.shuffle(all_images)
            sampled_images = all_images[:samples_per_class]

            for image_path in sampled_images:
                target_image_path = sampled_class_dir / image_path.name
                shutil.copy(image_path, target_image_path)


def filter_single_snippet_images(
    data_dir: str,
    exclude_classes: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Filter dataset to keep only images with single snippet extraction.

    From a directory structure with train/val/test subdirectories containing class
    subdirectories, keeps only files ending in -0 (indicating single snippet extraction).
    Files with -1, -2, etc. are removed as they represent multiple snippets from the
    same original image, leading to potentially duplicate or mislabeled data.

    Crash-safe workflow:
    1. Create temporary filtered directory (e.g. train_filtered_tmp)
    2. Populate temp directory completely
    3. Rename original directory to backup
    4. Atomically rename temp directory to final train/val/test dir

    Excluded classes are copied unchanged.

    Parameters:
        data_dir:
            Root directory containing train/val/test subdirectories.

        exclude_classes:
            Class subdirectory names to skip filtering.
            These classes are copied unchanged.

    Returns:
        Dictionary containing filtering statistics per subset and class.

    Raises:
        ValueError:
            If data_dir does not exist.

        FileExistsError:
            If backup or temporary directories already exist.
    """
    exclude_classes_set = set(exclude_classes or [])
    root_path = Path(data_dir)

    if not root_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    stats: dict[str, dict[str, Any]] = {}

    subsets = ["train", "val", "test"]

    for subset in subsets:
        subset_path = root_path / subset

        if not subset_path.exists() or not subset_path.is_dir():
            logger.warning(
                "Skipping subset '%s': directory not found",
                subset,
            )
            continue

        backup_path = root_path / f"{subset}_original_backup"
        temp_path = root_path / f"{subset}_filtered_tmp"

        # Safer behavior: never overwrite backups or temp dirs
        if backup_path.exists():
            raise FileExistsError(f"Backup directory already exists: {backup_path}")

        if temp_path.exists():
            raise FileExistsError(f"Temporary directory already exists: {temp_path}")

        logger.info(
            "Starting filtering for subset '%s'",
            subset,
        )

        temp_path.mkdir(parents=True, exist_ok=False)

        subset_stats: dict[str, Any] = {
            "per_class": {},
            "original_total": 0,
            "filtered_total": 0,
            "removed_total": 0,
        }

        try:
            # Build filtered dataset in temp dir first
            for class_dir in sorted(subset_path.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                temp_class_dir = temp_path / class_name
                temp_class_dir.mkdir(parents=True, exist_ok=True)

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
                            temp_class_dir / file_path.name,
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

            # Atomic-ish swap only after successful temp build
            subset_path.rename(backup_path)
            temp_path.rename(subset_path)

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

            # Cleanup temp dir if something failed
            if temp_path.exists():
                shutil.rmtree(temp_path)

            raise

    return stats
