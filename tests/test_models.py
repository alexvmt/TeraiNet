"""Tests for terainet.models module."""

import os
import tempfile

import pytest
import yaml

from terainet.models import create_class_list_yaml_file


def test_create_class_list_yaml_file(temp_dir):
    """Test creating a class list YAML file."""
    num_classes = 3
    class_names = ["tiger", "leopard", "bear"]
    file_path = os.path.join(temp_dir, "class_list.yaml")

    create_class_list_yaml_file(num_classes, class_names, file_path)

    assert os.path.exists(file_path)

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    assert data == {"1": "tiger", "2": "leopard", "3": "bear"}


def test_create_class_list_yaml_file_mismatch():
    """Test that mismatched num_classes and class_names raises ValueError."""
    with pytest.raises(ValueError, match="number of class names must match"):
        create_class_list_yaml_file(3, ["tiger", "leopard"], "/tmp/dummy.yaml")


def test_create_class_list_yaml_creates_directory(temp_dir):
    """Test that create_class_list_yaml_file creates missing directories."""
    nested_path = os.path.join(temp_dir, "nested", "dir", "class_list.yaml")
    num_classes = 2
    class_names = ["cat", "dog"]

    create_class_list_yaml_file(num_classes, class_names, nested_path)

    assert os.path.exists(nested_path)
    with open(nested_path, "r") as f:
        data = yaml.safe_load(f)
    assert data == {"1": "cat", "2": "dog"}
