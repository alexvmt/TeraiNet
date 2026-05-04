"""Tests for terainet.data module."""

import os

import polars as pl
import pytest

from terainet.data import (
    add_subset_column,
    check_location_split,
    sample_images,
    sample_n_images_per_species,
)


class TestSampleNImagesPerSpecies:
    """Tests for sample_n_images_per_species function."""

    def test_sample_n_images_per_species_basic(self):
        """Test basic sampling of images per species."""
        df = pl.DataFrame(
            {
                "species": ["tiger", "tiger", "leopard", "leopard", "bear"],
                "image_id": [1, 2, 3, 4, 5],
            }
        )

        species_samples = {"tiger": 1, "leopard": 1}
        result = sample_n_images_per_species(df, species_samples, "species", seed=42)

        assert result.height == 2
        assert result.select("species").n_unique() == 2

    def test_sample_n_images_per_species_more_than_available(self):
        """Test sampling more images than available for a species."""
        df = pl.DataFrame(
            {
                "species": ["tiger", "tiger", "leopard"],
                "image_id": [1, 2, 3],
            }
        )

        species_samples = {"tiger": 10, "leopard": 1}
        result = sample_n_images_per_species(df, species_samples, "species", seed=42)

        # Should sample min(10, 2) = 2 tigers and min(1, 1) = 1 leopard
        assert result.height == 3


class TestAddSubsetColumn:
    """Tests for add_subset_column function."""

    def test_add_subset_column_basic(self):
        """Test adding subset column to DataFrame."""
        df = pl.DataFrame(
            {
                "location_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                "image_id": list(range(10)),
            }
        )

        result = add_subset_column(df, train_ratio=0.6, seed=42)

        assert "subset" in result.columns
        assert result.select("subset").n_unique() <= 3
        subsets = result.select("subset").unique().to_series().to_list()
        assert set(subsets).issubset({"train", "val", "test", "unknown"})

    def test_add_subset_column_invalid_train_ratio(self):
        """Test that invalid train_ratio raises ValueError."""
        df = pl.DataFrame({"location_id": [1], "image_id": [1]})

        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            add_subset_column(df, train_ratio=1.5, seed=42)


class TestCheckLocationSplit:
    """Tests for check_location_split function."""

    def test_check_location_split_no_violations(self, capsys):
        """Test check_location_split with clean data."""
        df = pl.DataFrame(
            {
                "location_id": [1, 1, 2, 2, 3, 3],
                "subset": ["train", "train", "val", "val", "test", "test"],
            }
        )

        check_location_split(df)
        captured = capsys.readouterr()
        assert "✅ No violations found" in captured.out

    def test_check_location_split_missing_columns(self):
        """Test check_location_split raises error if required columns are missing."""
        df = pl.DataFrame({"location_id": [1]})

        with pytest.raises(ValueError, match="must contain location_id and subset columns"):
            check_location_split(df)


class TestSampleImages:
    """Tests for sample_images function."""

    def test_sample_images_basic(self, temp_dir):
        """Test basic image sampling."""
        # Create source directory structure
        class1_dir = os.path.join(temp_dir, "source", "class1")
        os.makedirs(class1_dir)

        # Create dummy image files
        for i in range(5):
            open(os.path.join(class1_dir, f"image_{i}.jpg"), "a").close()

        target_dir = os.path.join(temp_dir, "target")

        sample_images(os.path.join(temp_dir, "source"), target_dir, samples_per_class=3, seed=42)

        assert os.path.exists(os.path.join(target_dir, "class1"))
        sampled_files = os.listdir(os.path.join(target_dir, "class1"))
        assert len(sampled_files) == 3

    def test_sample_images_multiple_classes(self, temp_dir):
        """Test image sampling with multiple classes."""
        source_dir = os.path.join(temp_dir, "source")

        for class_name in ["class1", "class2"]:
            class_dir = os.path.join(source_dir, class_name)
            os.makedirs(class_dir)
            for i in range(5):
                open(os.path.join(class_dir, f"image_{i}.jpg"), "a").close()

        target_dir = os.path.join(temp_dir, "target")

        sample_images(source_dir, target_dir, samples_per_class=2, seed=42)

        assert os.path.exists(os.path.join(target_dir, "class1"))
        assert os.path.exists(os.path.join(target_dir, "class2"))
        assert len(os.listdir(os.path.join(target_dir, "class1"))) == 2
        assert len(os.listdir(os.path.join(target_dir, "class2"))) == 2
