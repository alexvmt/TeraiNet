"""Shared test fixtures and utilities."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def fixtures_dir():
    """Provide path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_config_path(fixtures_dir):
    """Provide path to sample config YAML file."""
    return fixtures_dir / "sample_config.yaml"
