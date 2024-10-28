#!/usr/bin/python3

from megadetector.detection.run_detector_batch import \
  load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils
import os

# Pick a folder to run MD on recursively, and an output file
image_folder = os.path.expanduser('images/panthera_tigris')
output_file = os.path.expanduser('megadetector_output_panthera_tigris.json')

# Recursively find images
image_file_names = path_utils.find_images(image_folder, recursive=True)

# This will automatically download MDv5a; you can also specify a filename.
detector_filename = 'megadetector/md_v5a.0.0.pt'
results = load_and_run_detector_batch(detector_filename, image_file_names)

# Write results to a format that Timelapse and other downstream tools like.
write_results_to_file(results,
                      output_file,
                      relative_path_base=image_folder,
                      detector_file=detector_filename)
