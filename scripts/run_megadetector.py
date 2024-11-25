#!/usr/bin/python3

import os
import argparse
import shutil
from megadetector.detection.run_detector_batch import \
  load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils

# Set up argument parser
parser = argparse.ArgumentParser(description="Run MegaDetector on a folder of images.")
parser.add_argument('image_folder', type=str, help="Path to the folder containing images.")
parser.add_argument('detector_filename', type=str, help="Path to the MegaDetector model file.")
args = parser.parse_args()

# Ensure the input image folder exists
if not os.path.isdir(args.image_folder):
    raise ValueError(f"The provided folder '{args.image_folder}' does not exist.")

# Ensure the MegaDetector model file exists
if not os.path.isfile(args.detector_filename):
    raise ValueError(f"The provided detector model file '{args.detector_filename}' does not exist.")

# Get the image folder from arguments
image_folder = os.path.expanduser(args.image_folder)

# Set the output file name and path
output_file = os.path.join(image_folder, 'md_out.json')

# Recursively find images in the folder
image_file_names = path_utils.find_images(image_folder, recursive=True)

# Specify MD model from argument
detector_filename = os.path.expanduser(args.detector_filename)

# Run MegaDetector on the images
results = load_and_run_detector_batch(detector_filename, image_file_names)

# Write the results to the output file
write_results_to_file(results,
                      output_file,
                      relative_path_base=image_folder,
                      detector_file=detector_filename)
