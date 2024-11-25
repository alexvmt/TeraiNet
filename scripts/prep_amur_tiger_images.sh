#!/bin/bash

# Check if the user has provided all required arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <source_directory> <class_number> <subset_name> <target_directory>"
    exit 1
fi

# Input arguments
source_dir="$1"
class_number="$2"
subset_name="$3"
target_dir="$4"

# Validate the source directory
if [ ! -d "$source_dir" ]; then
    echo "Error: Source directory $source_dir does not exist."
    exit 1
fi

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Process each file in the source directory
for file in "$source_dir"/*; do
    # Skip if it's not a file
    [ -f "$file" ] || continue

    # Extract the base name and extension of the file
    base_name=$(basename "$file")
    ext="${file##*.}"

    # Construct the new file name by appending class and subset
    new_name="${target_dir}/class_${class_number}_${subset_name}_${base_name}"

    # Move and rename the file
    mv "$file" "$new_name" || { echo "Error moving $file"; exit 1; }
done

# Remove the source directory
rm -rf "$source_dir" || { echo "Error removing source directory $source_dir"; exit 1; }

echo "All files moved and renamed. Source directory $source_dir removed."
