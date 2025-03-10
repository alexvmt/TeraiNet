#!/bin/bash

# Directory to store all images
destination_folder="images"

# Create the destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Loop through all ZIP files in the current directory
for zip_file in *.zip; do

    echo "Processing $zip_file..."

    # Unzip the ZIP file into a temporary folder
    unzip -q -d temp "$zip_file"

    # Find the only folder inside the ZIP (assuming one folder only)
    folder_name=$(find temp -mindepth 1 -maxdepth 1 -type d)

    # Copy images from that folder to the destination folder
    if [ -d "$folder_name" ]; then
        cp "$folder_name"/* "$destination_folder"/
    fi

    # Clean up the temporary folder
    rm -r temp
done

