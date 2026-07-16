"""Tests for terainet.data module."""

import os
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from terainet.data import (
    add_subset_column,
    check_location_split,
    filter_single_snippet_images,
    prepare_training_data,
    sample_images,
    sample_n_images_per_species,
    sample_validation_images_excluding_raw_prefixes,
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

    def test_check_location_split_no_violations(self, caplog):
        """Test check_location_split with clean data."""
        import logging

        caplog.set_level(logging.INFO)

        df = pl.DataFrame(
            {
                "location_id": [1, 1, 2, 2, 3, 3],
                "subset": ["train", "train", "val", "val", "test", "test"],
            }
        )

        check_location_split(df)
        assert "No violations found" in caplog.text

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

    def test_sample_images_rejects_insufficient_images_when_capping_disabled(self, temp_dir):
        """With capping disabled, sampling never silently returns fewer than requested."""
        source_dir = os.path.join(temp_dir, "source")
        class_dir = os.path.join(source_dir, "class1")
        os.makedirs(class_dir)
        open(os.path.join(class_dir, "image.jpg"), "a").close()

        with pytest.raises(ValueError, match="but 2 were requested"):
            sample_images(
                source_dir,
                os.path.join(temp_dir, "target"),
                samples_per_class=2,
                cap_to_available=False,
            )

    def test_sample_images_caps_to_available_by_default(self, temp_dir):
        """By default, a class short on images is sampled fully instead of raising."""
        source_dir = os.path.join(temp_dir, "source")
        class_dir = os.path.join(source_dir, "class1")
        os.makedirs(class_dir)
        open(os.path.join(class_dir, "image.jpg"), "a").close()

        target_dir = os.path.join(temp_dir, "target")
        sampled_counts = sample_images(source_dir, target_dir, samples_per_class=2)

        assert sampled_counts == {"class1": 1}
        assert len(os.listdir(os.path.join(target_dir, "class1"))) == 1

    def test_sample_images_uses_explicit_class_source_override(self, temp_dir):
        """An intentionally unfiltered class can use its raw source directory."""
        filtered_source = os.path.join(temp_dir, "filtered")
        raw_class_dir = os.path.join(temp_dir, "raw", "class_1")
        os.makedirs(filtered_source)
        os.makedirs(raw_class_dir)
        for index in range(2):
            open(os.path.join(raw_class_dir, f"image_{index}.jpg"), "a").close()

        target_dir = os.path.join(temp_dir, "sampled")
        sample_images(
            filtered_source,
            target_dir,
            samples_per_class=2,
            class_directories=("class_1",),
            source_paths_by_class={"class_1": Path(raw_class_dir)},
        )

        assert len(os.listdir(os.path.join(target_dir, "class_1"))) == 2

    def test_sample_images_skips_intentionally_absent_class_directory(self, temp_dir):
        """A class created by a later split policy need not exist in this source subset."""
        source_dir = os.path.join(temp_dir, "source")
        class_2_dir = os.path.join(source_dir, "class_2")
        os.makedirs(class_2_dir)
        for index in range(2):
            open(os.path.join(class_2_dir, f"image_{index}.jpg"), "a").close()

        target_dir = os.path.join(temp_dir, "sampled")
        sample_images(
            source_dir,
            target_dir,
            samples_per_class=2,
            class_directories=("class_1", "class_2"),
            skip_class_directories={"class_1"},
        )

        assert not os.path.exists(os.path.join(target_dir, "class_1"))
        assert len(os.listdir(os.path.join(target_dir, "class_2"))) == 2

    def test_sample_images_rejects_unknown_skipped_class_directory(self, temp_dir):
        """Typos in skipped class names fail before data is sampled."""
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(os.path.join(source_dir, "class_1"))

        with pytest.raises(ValueError, match="Cannot skip unknown"):
            sample_images(
                source_dir,
                os.path.join(temp_dir, "sampled"),
                samples_per_class=0,
                class_directories=("class_1",),
                skip_class_directories={"class_99"},
            )


def test_sample_validation_images_excluding_raw_prefixes(temp_dir):
    """Validation sampling excludes every raw image represented in training."""
    source_dir = os.path.join(temp_dir, "source")
    training_dir = os.path.join(temp_dir, "training")
    target_dir = os.path.join(temp_dir, "validation")
    os.makedirs(source_dir)
    os.makedirs(training_dir)

    for filename in ["raw1-0.jpg", "raw2-0.jpg", "raw3-0.jpg"]:
        open(os.path.join(source_dir, filename), "a").close()
    open(os.path.join(training_dir, "raw1-1.jpg"), "a").close()

    actual_count = sample_validation_images_excluding_raw_prefixes(
        source_dir,
        training_dir,
        target_dir,
        samples_per_class=2,
        seed=42,
    )

    assert actual_count == 2
    assert {path.split("-")[0] for path in os.listdir(target_dir)} == {"raw2", "raw3"}


def test_sample_validation_images_excluding_raw_prefixes_caps_to_available_by_default(temp_dir):
    """Fewer eligible images than requested are sampled fully instead of raising."""
    source_dir = os.path.join(temp_dir, "source")
    training_dir = os.path.join(temp_dir, "training")
    target_dir = os.path.join(temp_dir, "validation")
    os.makedirs(source_dir)
    os.makedirs(training_dir)

    open(os.path.join(source_dir, "raw2-0.jpg"), "a").close()
    open(os.path.join(training_dir, "raw1-1.jpg"), "a").close()

    actual_count = sample_validation_images_excluding_raw_prefixes(
        source_dir,
        training_dir,
        target_dir,
        samples_per_class=2,
        seed=42,
    )

    assert actual_count == 1
    assert os.listdir(target_dir) == ["raw2-0.jpg"]


def test_sample_validation_images_excluding_raw_prefixes_rejects_when_capping_disabled(temp_dir):
    """With capping disabled, insufficient eligible images raise instead of under-sampling."""
    source_dir = os.path.join(temp_dir, "source")
    training_dir = os.path.join(temp_dir, "training")
    os.makedirs(source_dir)
    os.makedirs(training_dir)

    open(os.path.join(source_dir, "raw2-0.jpg"), "a").close()

    with pytest.raises(ValueError, match="but 2 were requested"):
        sample_validation_images_excluding_raw_prefixes(
            source_dir,
            training_dir,
            os.path.join(temp_dir, "validation"),
            samples_per_class=2,
            seed=42,
            cap_to_available=False,
        )


def test_prepare_training_data_creates_missing_policy_target_class(temp_dir):
    """Raw-prefix validation sampling owns an intentionally missing target class."""
    input_dir = Path(temp_dir) / "input"
    for subset in ("train", "val", "test"):
        for class_directory in ("class_1", "class_2"):
            if subset == "val" and class_directory == "class_1":
                continue
            class_path = input_dir / subset / class_directory
            class_path.mkdir(parents=True)
            filenames = (
                ["raw1-0.jpg", "raw2-0.jpg", "raw3-0.jpg"]
                if subset == "train" and class_directory == "class_1"
                else ["image1.jpg", "image2.jpg"]
            )
            for filename in filenames:
                (class_path / filename).touch()

    config = SimpleNamespace(
        class_names=("tiger", "leopard"),
        seed=42,
        dataset=SimpleNamespace(
            images_input_dir=input_dir,
            images_filtered_dir=Path(temp_dir) / "filtered",
            images_sampled_dir=Path(temp_dir) / "sampled",
            class_directories=("class_1", "class_2"),
            subsets={"train": "train", "val": "val", "test": "test"},
            samples_per_class={"train": 1, "val": 1, "test": 1},
            filter_single_snippets=False,
            exclude_classes_from_filtering=(),
            raw_prefix_validation_sampling={
                "class_name": "tiger",
                "source_subset": "train",
                "target_subset": "val",
            },
            cap_samples_to_available=True,
        ),
    )

    result = prepare_training_data(config)

    train_filenames = os.listdir(result.sampled_subset_paths["train"] / "class_1")
    validation_filenames = os.listdir(result.sampled_subset_paths["val"] / "class_1")
    assert len(validation_filenames) == 1
    assert train_filenames[0].split("-")[0] != validation_filenames[0].split("-")[0]
    assert result.samples_per_subset["val"]["class_1"] == 1


class TestFilterSingleSnippetImages:
    """Tests for filter_single_snippet_images function."""

    def test_filter_single_snippet_basic(self, temp_dir):
        """Test basic filtering of single snippet images."""
        # Create structure: data_dir/train/class_1/files
        data_dir = temp_dir
        train_dir = os.path.join(data_dir, "train")
        class_dir = os.path.join(train_dir, "class_1")
        os.makedirs(class_dir)

        # Create files: some with -0 (keep), some with -1, -2 (remove)
        open(os.path.join(class_dir, "image_1-0.jpg"), "a").close()
        open(os.path.join(class_dir, "image_1-1.jpg"), "a").close()
        open(os.path.join(class_dir, "image_2-0.jpg"), "a").close()
        open(os.path.join(class_dir, "image_2-1.jpg"), "a").close()
        open(os.path.join(class_dir, "image_2-2.jpg"), "a").close()

        target_dir = os.path.join(temp_dir, "filtered")
        stats = filter_single_snippet_images(data_dir, target_dir)

        # Check target filtered directory created with correct files
        new_class_dir = os.path.join(target_dir, "train", "class_1")
        assert os.path.exists(new_class_dir)
        filtered_files = os.listdir(new_class_dir)
        assert len(filtered_files) == 2
        assert "image_1-0.jpg" in filtered_files
        assert "image_2-0.jpg" in filtered_files

        # Check source directory unchanged
        original_files = os.listdir(class_dir)
        assert len(original_files) == 5

        # Check statistics
        assert stats["train"]["original_total"] == 5
        assert stats["train"]["filtered_total"] == 2
        assert stats["train"]["removed_total"] == 3
        assert stats["train"]["per_class"]["class_1"]["original"] == 5
        assert stats["train"]["per_class"]["class_1"]["filtered"] == 2

    def test_filter_single_snippet_multiple_subsets(self, temp_dir):
        """Test filtering across train, val, and test subsets."""
        data_dir = temp_dir

        # Create train, val, test dirs with class subdirs
        for subset in ["train", "val", "test"]:
            class_dir = os.path.join(data_dir, subset, "class_1")
            os.makedirs(class_dir)

            # Create files
            open(os.path.join(class_dir, "img-0.jpg"), "a").close()
            open(os.path.join(class_dir, "img-1.jpg"), "a").close()

        target_dir = os.path.join(temp_dir, "filtered")
        stats = filter_single_snippet_images(data_dir, target_dir)

        # Check all subsets processed
        for subset in ["train", "val", "test"]:
            assert subset in stats
            assert stats[subset]["original_total"] == 2
            assert stats[subset]["filtered_total"] == 1
            assert os.path.exists(os.path.join(target_dir, subset, "class_1"))

    def test_filter_single_snippet_exclude_classes(self, temp_dir):
        """Test exclude_classes parameter."""
        data_dir = temp_dir
        train_dir = os.path.join(data_dir, "train")

        # Create two classes
        for class_name in ["class_1", "class_2"]:
            class_dir = os.path.join(train_dir, class_name)
            os.makedirs(class_dir)

            open(os.path.join(class_dir, "img-0.jpg"), "a").close()
            open(os.path.join(class_dir, "img-1.jpg"), "a").close()

        target_dir = os.path.join(temp_dir, "filtered")
        stats = filter_single_snippet_images(
            data_dir,
            target_dir,
            exclude_classes=["class_2"],
        )

        # class_1 should be filtered
        assert stats["train"]["per_class"]["class_1"]["original"] == 2
        assert stats["train"]["per_class"]["class_1"]["filtered"] == 1
        assert stats["train"]["per_class"]["class_1"]["removed"] == 1

        # class_2 should be copied unchanged
        assert stats["train"]["per_class"]["class_2"]["original"] == 2
        assert stats["train"]["per_class"]["class_2"]["filtered"] == 2
        assert stats["train"]["per_class"]["class_2"]["removed"] == 0

        new_class_2_dir = os.path.join(target_dir, "train", "class_2")

        assert os.path.exists(new_class_2_dir)

        copied_files = os.listdir(new_class_2_dir)

        assert len(copied_files) == 2
        assert "img-0.jpg" in copied_files
        assert "img-1.jpg" in copied_files

    def test_filter_single_snippet_missing_data_dir(self, temp_dir):
        """Test error handling for missing data directory."""
        nonexistent_dir = os.path.join(temp_dir, "nonexistent")
        target_dir = os.path.join(temp_dir, "target")

        with pytest.raises(ValueError, match="Source data directory not found"):
            filter_single_snippet_images(nonexistent_dir, target_dir)

    def test_filter_single_snippet_missing_subset(self, temp_dir):
        """Test handling of missing subset directories."""
        data_dir = temp_dir

        # Create only train, not val or test
        train_dir = os.path.join(data_dir, "train")
        class_dir = os.path.join(train_dir, "class_1")
        os.makedirs(class_dir)
        open(os.path.join(class_dir, "img-0.jpg"), "a").close()

        target_dir = os.path.join(temp_dir, "filtered")
        stats = filter_single_snippet_images(data_dir, target_dir)

        # Should only have train in stats
        assert "train" in stats
        assert "val" not in stats
        assert "test" not in stats

    def test_filter_single_snippet_preserves_filenames(self, temp_dir):
        """Test that original filenames are preserved."""
        data_dir = temp_dir
        train_dir = os.path.join(data_dir, "train")
        class_dir = os.path.join(train_dir, "class_1")
        os.makedirs(class_dir)

        # Create files with various naming patterns
        filenames = [
            "class_7_image_1015-0.JPG",
            "species_tiger_snippet-0.jpg",
            "raw_photo_name-0.PNG",
        ]
        for filename in filenames:
            open(os.path.join(class_dir, filename), "a").close()

        target_dir = os.path.join(temp_dir, "filtered")
        filter_single_snippet_images(data_dir, target_dir)

        new_class_dir = os.path.join(target_dir, "train", "class_1")
        filtered_files = os.listdir(new_class_dir)

        assert len(filtered_files) == 3
        for filename in filenames:
            assert filename in filtered_files

    def test_filter_single_snippet_existing_nonempty_target_raises(self, temp_dir):
        """Test error raised when target directory already exists and is non-empty."""
        data_dir = os.path.join(temp_dir, "source")
        target_dir = os.path.join(temp_dir, "filtered")

        class_dir = os.path.join(data_dir, "train", "class_1")
        os.makedirs(class_dir)

        open(os.path.join(class_dir, "img-0.jpg"), "a").close()

        os.makedirs(target_dir)
        open(os.path.join(target_dir, "stale_file.txt"), "a").close()

        with pytest.raises(FileExistsError):
            filter_single_snippet_images(data_dir, target_dir)
