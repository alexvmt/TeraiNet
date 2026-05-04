"""Tests for terainet.detection module."""

from terainet.detection import contains_animal


def test_contains_animal_with_detection():
    """Test contains_animal with a valid animal detection."""
    json_image = {
        "detections": [
            {"category": "1", "conf": 0.95, "bbox": [0.1, 0.2, 0.3, 0.4]},
        ]
    }
    assert contains_animal(json_image) is True


def test_contains_animal_without_detection():
    """Test contains_animal with no detections."""
    json_image = {"detections": []}
    assert contains_animal(json_image) is False


def test_contains_animal_with_non_animal_category():
    """Test contains_animal with non-animal category."""
    json_image = {
        "detections": [
            {"category": "2", "conf": 0.95, "bbox": [0.1, 0.2, 0.3, 0.4]},
        ]
    }
    assert contains_animal(json_image) is False


def test_contains_animal_no_detections_key():
    """Test contains_animal when 'detections' key is missing."""
    json_image = {"metadata": {}}
    assert contains_animal(json_image) is False
