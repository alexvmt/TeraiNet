"""Tests for terainet.config module."""

import pytest

from terainet.config import load_config


def test_load_config(sample_config_path):
    """Test loading a YAML config file."""
    config = load_config(str(sample_config_path))

    assert config is not None
    assert "classes" in config
    assert config["classes"]["num_classes"] == 3
    assert config["classes"]["tiger"] == 1


def test_load_config_nonexistent():
    """Test loading a nonexistent config file raises error."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")
