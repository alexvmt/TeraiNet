#!/bin/bash

# Validate input arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_file.csv> <download_directory>"
    exit 1
fi

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

# Extract class number from the last column of the first data row (skipping header)
class_number=$(awk -F ',' 'NR==2 {gsub(/"/, "", $NF); print $NF}' "$input_file")

# Validate class_number
if ! [[ "$class_number" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid class number '$class_number'."
    exit 1
fi

# Initialize counters
total_urls=0
success_download_count=0
failed_download_count=0

# Read CSV file line by line, skipping the header
first_line_skipped=false
while IFS=, read -r _ url _ class; do
    # Skip the first row (header)
    if [ "$first_line_skipped" = false ]; then
        first_line_skipped=true
        continue
    fi

    # Trim spaces safely
    url=$(echo "$url" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    # Skip empty URLs
    if [ -z "$url" ]; then
        echo "Warning: Skipping empty URL entry."
        continue
    fi

    ((total_urls++))

    # Extract file extension
    file_extension="${url##*.}"
    [[ -z "$file_extension" || "$file_extension" == "$url" ]] && file_extension="jpg"

    # Construct unique filename
    unique_name="class_${class_number}_${total_urls}.$file_extension"

    # Check if URL is reachable
    if wget --spider -q "$url"; then
        # Download file
        if wget --tries=3 --timeout=10 -q "$url" -O "$download_dir/$unique_name"; then
            ((success_download_count++))
        else
            echo "Error downloading: $url"
            ((failed_download_count++))
        fi
    else
        echo "Error: URL is unreachable - $url"
        ((failed_download_count++))
    fi
done < "$input_file"

# Print summary
echo "Total URLs: $total_urls"
echo "Total successful downloads: $success_download_count"
echo "Total failed downloads: $failed_download_count"
echo "Final file count in directory: $(ls -1 "$download_dir" | wc -l)"
echo "Download complete. Files saved to $download_dir."
