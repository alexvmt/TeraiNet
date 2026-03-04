#!/bin/bash

# Validate input arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_file.csv> <download_directory>"
    exit 1
fi

# Input arguments
input_file="$1"
download_dir="$2"

# Validate input file
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist."
    exit 1
fi

# Validate download directory
if [ ! -d "$download_dir" ]; then
    echo "Error: Download directory '$download_dir' does not exist."
    exit 1
fi

# Extract class number from the filename (integer before .csv)
class_number=$(basename "$input_file" | grep -oE '[0-9]+' | head -1)

# Extract URLs and subsets from the CSV (URL in column 1, subset in the last column)
urls=$(awk -F, 'NR > 1 {gsub(/"/, "", $1); print $1}' "$input_file")
subsets=$(awk -F, 'NR > 1 {gsub(/"/, "", $NF); print $NF}' "$input_file")

# Count total URLs
total_lines=$(echo "$urls" | wc -l)

# Initialize counters
total_urls=0
success_download_count=0
failed_download_count=0

echo "Downloading images into $download_dir..."

# Use paste to combine URLs and subsets line by line
paste <(echo "$urls") <(echo "$subsets") | while IFS=$'\t' read -r url subset; do
    # Skip empty URLs or subsets
    if [ -z "$url" ] || [ -z "$subset" ]; then
        echo "Warning: Skipping entry with missing URL or subset."
        continue
    fi

    # Generate file name
    ((total_urls++))
    ext="${url##*.}"  # Extract the extension from the URL
    new_name="class_${class_number}_${subset}_${total_urls}.${ext}"

    # Check if URL is reachable
    if wget --spider -q "$url"; then
        # Download file
        if wget --tries=3 --timeout=10 -q "$url" -O "$download_dir/$new_name"; then
            ((success_download_count++))
        else
            echo "Error downloading: $url"
            ((failed_download_count++))
        fi
    else
        echo "Error: URL is unreachable - $url"
        ((failed_download_count++))
    fi

	# Print summary on last iteration
    if [ "$total_urls" -eq "$total_lines" ]; then
        echo "Total URLs: $total_urls"
        echo "Total successful downloads: $success_download_count"
        echo "Total failed downloads: $failed_download_count"
    fi
done

echo "Final file count in directory: $(ls -1 "$download_dir" | wc -l)"
echo "Downloading images complete."
