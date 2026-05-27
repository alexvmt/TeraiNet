"""Detection utilities for TeraiNet. Adapted from MEWC."""


def contains_animal(json_image: dict) -> bool:
    """
    Check if a JSON image object from MegaDetector contains an animal detection.

    Parameters:
        json_image (dict): A MegaDetector detection JSON object.

    Returns:
        bool: True if an animal (category "1") is detected, False otherwise.
    """
    if "detections" in json_image.keys():
        n = len(json_image["detections"])
        animal_there = False
        for i in range(0, n):
            if json_image["detections"][i]["category"] == "1":
                animal_there = True
        return animal_there
    else:
        return False
