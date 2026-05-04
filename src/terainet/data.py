"""Data utilities for TeraiNet."""

import os
import random
import shutil

import numpy as np
import polars as pl


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
        print("⚠️ Warning: Some locations could not be assigned to a subset.")

    final_counts = df.group_by("subset").len().rename({"len": "count"}).sort("subset")
    print(f"ℹ️ Subset distribution:\n{final_counts}")

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
        print("✅ No violations found: each location_id appears in only one subset.")
    else:
        print(f"❌ Violations found in {violations.height} location_id(s):")
        print(violations)


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

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            sampled_class_dir = os.path.join(target_dir, class_name)
            os.makedirs(sampled_class_dir, exist_ok=True)

            all_images = list(os.listdir(class_path))
            random.shuffle(all_images)
            sampled_images = all_images[:samples_per_class]

            for image_name in sampled_images:
                source_image_path = os.path.join(class_path, image_name)
                target_image_path = os.path.join(sampled_class_dir, image_name)
                shutil.copy(source_image_path, target_image_path)
