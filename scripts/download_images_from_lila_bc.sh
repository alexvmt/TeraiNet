#!/bin/bash

# Check if the user has provided all required arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file.csv>"
    exit 1
fi

# Store the input arguments
input_file="$1"

# Extract the base directory for downloading images
base_dir=$(dirname "$input_file")
download_dir="${base_dir}/downloaded_images"

# Create the 'downloaded_images' directory if it doesn't exist
mkdir -p "$download_dir"

echo "Downloading images based on $input_file into $download_dir..."

# Extract URLs from the input file
urls=$(awk -F ',' 'NR > 1 {print $2}' "$input_file")

# Initialize the failed download counter
failed_download_count=0

# Download images only if they don't exist
for url in $urls; do
    filename=$(basename "$url")
    if [ ! -f "$download_dir/$filename" ]; then
        wget -P "$download_dir" "$url" -q || ((failed_download_count++))
    fi
done

# Print total failed downloads
if [ "$failed_download_count" -gt 0 ]; then
    echo "Total failed downloads: $failed_download_count"
else
    echo "All images successfully downloaded."
fi

# Fix file names for .jpg.* and .png.* extensions in one loop
find "$download_dir" -type f \( -name "*.jpg.*" -o -name "*.png.*" \) | while read -r file; do
    new_name=$(echo "$file" | sed -E 's/(.*)\.(jpg|png)\.(.*)/\1_\3.\2/')
    mv "$file" "$new_name"
done

echo "Downloading complete. File names fixed."
