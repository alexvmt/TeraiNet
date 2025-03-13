import os
import yaml
import subprocess
import numpy as np
import pandas as pd


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def run_get_image_urls_script(get_image_urls_script, urls_and_labels, column_to_filter, values_to_filter, samples_file):

    command = [
        'bash',
        get_image_urls_script,
        urls_and_labels,
        column_to_filter,
        values_to_filter,
        samples_file
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    return result.stdout, result.stderr


def get_image_urls(
    samples_dir: str,
    species: str,
    get_image_urls_script: str,
    urls_and_labels: str,
    column_to_filter: str,
    values_to_filter: str
  ):
  """
    Retrieves image URLs for a specified species, filters them based on given criteria,
    and assigns a train/test subset label.

    Parameters:
    - samples_dir (str): The directory where the samples file will be saved.
    - species (str): The target species for filtering image URLs.
    - get_image_urls_script (str): Path to the script used for retrieving image URLs.
    - urls_and_labels (str): Path to the CSV file containing image URLs and labels.
    - column_to_filter (str): The column in the CSV file used for filtering (e.g., 'scientific_name').
    - values_to_filter (str): Comma-separated values to filter within the specified column.

    Returns:
    The function writes the filtered image URLs to a CSV file and returns the file path.
  """

  # get image urls
  samples_file = samples_dir + '/lila_bc_image_urls_' + species + '.csv'
  stdout, stderr = run_get_image_urls_script(get_image_urls_script, urls_and_labels, column_to_filter, values_to_filter, samples_file)
  print(stdout)
  print(stderr)

  return samples_file


def sample_n_images_per_species(image_urls: str, species_samples_dict: dict, column_to_filter: str, seed: int = 42) -> pd.DataFrame:
    """
    Filters and randomly samples a specified number of rows for each species from a CSV file.

    Parameters:
    - image_urls (str): Path to the image URLs CSV file.
    - species_samples_dict (dict): A dictionary where keys are species names,
      and values are the number of samples to draw for each species.
    - column_to_filter (str): The column name used for filtering species.
    - seed (int, optional): Seed for reproducibility. Default is 42.

    Returns:
    - pd.DataFrame: A DataFrame containing the sampled rows for each species.
    """
    # Load the CSV file
    df = pd.read_csv(image_urls, low_memory=False)

    # Initialize an empty list to store sampled DataFrames
    sampled_dfs = []

    # Sample rows per species
    for species, sample_size in species_samples_dict.items():
        species_df = df[df[column_to_filter] == species]
        sampled_df = species_df.sample(n=min(sample_size, len(species_df)), random_state=seed)
        sampled_dfs.append(sampled_df)

    # Concatenate all sampled data into a single DataFrame
    return pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else pd.DataFrame()


def add_subset_column(df: pd.DataFrame, train_ratio: float, seed: int = 42) -> pd.DataFrame:
    """
    Adds a 'subset' column to the DataFrame, splitting data into 'train' and 'test' subsets.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        train_ratio (float): The ratio of the 'train' subset (e.g., 0.8 for 80% train).
        seed (int, optional): Seed for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'subset' column.
    """
    if not 0 <= train_ratio <= 1:
        raise ValueError('train_ratio must be between 0 and 1.')

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Generate random values to assign subsets
    random_values = np.random.rand(len(df))

    # Assign subsets based on the train_ratio
    df['subset'] = np.where(random_values < train_ratio, 'train', 'test')

    return df
