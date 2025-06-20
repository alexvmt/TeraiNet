import os
import yaml
import json
import shutil
import random
import subprocess
import numpy as np
import pandas as pd
import polars as pl


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def sample_n_images_per_species(
    df: pl.DataFrame,
    species_samples_dict: dict,
    column_to_filter: str,
    seed: int = 42
) -> pl.DataFrame:
    """
    Filters and randomly samples a specified number of rows for each species
    from a Polars DataFrame.

    Parameters:
    - df (pl.DataFrame): A Polars DataFrame containing image data and species labels.
    - species_samples_dict (dict): A dictionary where keys are species names,
      and values are the number of samples to draw for each species.
    - column_to_filter (str): The column name used for filtering species.
    - seed (int, optional): Seed for reproducibility. Default is 42.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing the sampled rows for each species.
    """
    # Initialize a list to collect sampled DataFrames
    sampled_dfs = []

    # Sample rows for each species
    for species, sample_size in species_samples_dict.items():
        # Filter rows where the specified column matches the species
        species_df = df.filter(pl.col(column_to_filter) == species)

        # Determine how many rows to sample (do not exceed available rows)
        n = min(sample_size, species_df.height)

        # Sample if there are enough rows
        if n > 0:
            sampled_df = species_df.sample(n=n, seed=seed, with_replacement=False)
            sampled_dfs.append(sampled_df)

    # Concatenate all sampled DataFrames into a single Polars DataFrame
    return pl.concat(sampled_dfs) if sampled_dfs else pl.DataFrame()


def add_subset_column(df: pl.DataFrame, train_ratio: float, seed: int = 42) -> pl.DataFrame:
    """
    Adds a 'subset' column to the Polars DataFrame, splitting data into 'train', 'val', and 'test' subsets
    based on unique 'location_id'. Ensures that no 'location_id' appears in more than one subset to prevent
    spatial leakage, which is critical for generalization in camera trap datasets.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame.
        train_ratio (float): The ratio of the 'train' subset (e.g., 0.8 for 80% train).
        seed (int, optional): Seed for reproducibility. Default is 42.

    Returns:
        pl.DataFrame: The DataFrame with an added 'subset' column.
    """
    if not 0 <= train_ratio <= 1:
        raise ValueError('train_ratio must be between 0 and 1.')

    # Get unique location_ids and shuffle them
    unique_locs = df.select('location_id').unique()
    loc_list = unique_locs['location_id'].to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(loc_list)

    # Determine how many locations go into each subset
    n_locs = len(loc_list)
    n_train = int(n_locs * train_ratio)
    n_remaining = n_locs - n_train
    n_val = n_remaining // 2
    n_test = n_remaining - n_val  # handles odd remainder

    # Assign subsets based on shuffled location list
    train_locs = set(loc_list[:n_train])
    val_locs = set(loc_list[n_train:n_train + n_val])
    test_locs = set(loc_list[n_train + n_val:])

    # Assign subset label to each row based on its location
    df = df.with_columns(
        pl.when(pl.col('location_id').is_in(train_locs)).then(pl.lit('train'))
        .when(pl.col('location_id').is_in(val_locs)).then(pl.lit('val'))
        .when(pl.col('location_id').is_in(test_locs)).then(pl.lit('test'))
        .otherwise(pl.lit('unknown'))
        .alias('subset')
    )
    
    if df.filter(pl.col('subset') == 'unknown').shape[0] > 0:
        print('Warning: Unassigned locations detected.')

    return df


def check_location_split(df: pl.DataFrame) -> None:
    """
    Checks whether any 'location_id' appears in more than one subset.

    Parameters:
        df (pl.DataFrame): The Polars DataFrame containing 'location_id' and 'subset' columns.

    Prints:
        - A success message if there are no violations.
        - Otherwise, prints the violating 'location_id's and the number of subsets they appear in.
    """
    if 'location_id' not in df.columns or 'subset' not in df.columns:
        raise ValueError('DataFrame must contain location_id and subset columns.')

    violations = (
        df.select(['location_id', 'subset'])
          .unique()
          .group_by('location_id')
          .agg(pl.col('subset').n_unique().alias('num_subsets'))
          .filter(pl.col('num_subsets') > 1)
    )

    if violations.is_empty():
        print('✅ No violations found: each location_id appears in only one subset.')
    else:
        print(f'❌ Violations found in {violations.height} location_id(s):')
        print(violations)


# from mewc-detect
def contains_animal(json_image):
    if 'detections' in json_image.keys():
        n = len(json_image['detections'])
        animal_there = False
        for i in range(0,n):
            if json_image['detections'][i]['category'] == "1":
                animal_there = True
        return(animal_there)
    else:
        return(False)


def create_class_list_yaml_file(num_classes, class_names, file_path):
    """
    Create a YAML file that maps numerical indices to class names, ensuring numerical sorting.

    Args:
        num_classes (int): The number of classes. Must match the length of `class_names`.
        class_names (list of str): A list of class names to include in the YAML file.
        file_path (str): The full file path where the YAML file will be saved.
    """
    if len(class_names) != num_classes:
        raise ValueError('The number of class names must match num_classes.')

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Create a dictionary with indices as keys and class names as values, ensuring numerical order
    class_dict = {str(i): class_names[i - 1] for i in range(1, num_classes + 1)}

    # Write to YAML file
    with open(file_path, 'w') as file:
        yaml.dump(class_dict, file, default_flow_style=False, sort_keys=True)


def sample_images(source_dir, target_dir, samples_per_class, exclude_dir=None, seed=42):
    """
    Samples a fixed number of images per class from a directory structure, ensuring that
    validation samples do not overlap with the test dataset.

    Args:
        source_dir (str): Path to the source dataset directory (e.g., test images).
        target_dir (str): Path to the target dataset directory to store sampled data.
        samples_per_class (int): Number of images to sample per class.
        exclude_dir (str, optional): Path to directory which images have to be excluded during sampling.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            sampled_class_dir = os.path.join(target_dir, class_name)
            os.makedirs(sampled_class_dir, exist_ok=True)

            # List all files in class directory
            all_images = set(os.listdir(class_path))

            # Exclude certain images
            if exclude_dir:
                test_class_path = os.path.join(exclude_dir, class_name)
                if os.path.exists(test_class_path):
                    test_images = set(os.listdir(test_class_path))
                    all_images -= test_images

            # Shuffle and select desired number of samples
            all_images = list(all_images)
            random.shuffle(all_images)
            sampled_images = all_images[:samples_per_class]

            # Copy sampled images to new directory
            for image_name in sampled_images:
                source_image_path = os.path.join(class_path, image_name)
                target_image_path = os.path.join(sampled_class_dir, image_name)
                shutil.copy(source_image_path, target_image_path)
