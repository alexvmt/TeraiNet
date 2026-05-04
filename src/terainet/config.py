"""Configuration utilities for TeraiNet."""

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
