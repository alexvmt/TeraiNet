"""Model utilities for TeraiNet. Adapted from MEWC."""

from pathlib import Path

import yaml


def create_class_list_yaml_file(num_classes: int, class_names: list[str], file_path: str) -> None:
    """
    Create a YAML file that maps numerical indices to class names.

    The file ensures numerical sorting so that index 1 corresponds to the first class name,
    index 2 to the second, and so on.

    Parameters:
        num_classes (int): The number of classes. Must match the length of `class_names`.
        class_names (list[str]): A list of class names to include in the YAML file.
        file_path (str): The full file path where the YAML file will be saved.

    Raises:
        ValueError: If the number of class names does not match `num_classes`.
    """
    if len(class_names) != num_classes:
        raise ValueError("The number of class names must match num_classes.")

    # Ensure the directory exists
    file_path_obj = Path(file_path)
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Create a dictionary with indices as keys and class names as values
    class_dict = {str(i): class_names[i - 1] for i in range(1, num_classes + 1)}

    # Write to YAML file
    with open(file_path_obj, "w") as file:
        yaml.dump(class_dict, file, default_flow_style=False, sort_keys=True)
