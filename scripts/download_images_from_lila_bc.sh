#!/bin/bash

# Validate input arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <input_file.csv> <class_number> <download_directory>"
    exit 1
fi

# Input arguments
input_file="$1"
class_number="$2"
download_dir="$3"

# Validate input file
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist."
    exit 1
fi

# Validate class_number (must be a positive integer)
if ! [[ "$class_number" =~ ^[0-9]+$ ]]; then
    echo "Error: Class number must be a positive integer."
    exit 1
fi

# Ensure the download directory exists
mkdir -p "$download_dir" || {
    echo "Error: Could not create directory '$download_dir'. Check permissions."
    exit 1
}

# Initialize counters
failed_download_count=0
image_count=0

# Extract URLs and subsets from the CSV (URL in column 2, subset in the last column)
urls_and_subsets=$(awk -F ',' 'NR > 1 {print $2, $NF}' "$input_file")

echo "Downloading images into directory: $download_dir..."

# Download and rename files
while IFS=" " read -r url subset; do
    # Skip empty URLs or subsets
    if [ -z "$url" ] || [ -z "$subset" ]; then
        echo "Warning: Skipping entry with missing URL or subset."
        continue
    fi

    # Sanitize subset name (replace spaces or special characters with underscores)
    subset=$(echo "$subset" | tr -s '[:blank:]' '_')

    # Generate file name
    image_count=$((image_count + 1))
    ext="${url##*.}" # Extract the extension from the URL
    new_name="class_${class_number}_${subset}_${image_count}.${ext}"

    # Download the file
    wget --tries=3 --timeout=10 -q "$url" -O "$download_dir/$new_name" || {
        echo "Error downloading $url"
        ((failed_download_count++))
    }
done <<< "$urls_and_subsets"

# Print summary
if [ "$failed_download_count" -gt 0 ]; then
    echo "Total failed downloads: $failed_download_count"
else
    echo "All images successfully downloaded."
fi

# Clean up temporary files
trap 'rm -f /tmp/shuf_input' EXIT

echo "Download complete. Files saved to $download_dir."
