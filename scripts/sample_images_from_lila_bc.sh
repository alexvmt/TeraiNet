#!/bin/bash

# Check if the user has provided all required arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <input_file.csv> <column_to_filter> <values_to_filter> <sample_size> <output_path> [random_seed]"
    echo "<column_to_filter> should be 'scientific_name' (column 10), 'common_name' (column 11), or 'genus' (column 29)."
    echo "<values_to_filter> should be a comma-separated list of values (e.g., 'value1,value2')."
    exit 1
fi

# Store the input arguments
input_file="$1"
column_to_filter="$2"
values_to_filter="$3"  # Comma-separated list of values
sample_size="$4"
output_path="$5"
random_seed="${6:-42}"  # Use provided seed or default to 42 for reproducibility

# Validate the input file
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found. Please provide a valid CSV file."
    exit 1
fi

# Validate the output directory
if [ ! -d "$output_path" ]; then
    echo "Error: Output path '$output_path' does not exist. Exiting."
    exit 1
fi

# Validate the filter column
header_columns=$(head -n 1 "$input_file" | awk -F ',' '{print NF}')
case "$column_to_filter" in
    "scientific_name") filter_column=10 ;;
    "common_name") filter_column=11 ;;
    "genus") filter_column=29 ;;
    *)
        echo "Invalid column_to_filter: choose 'scientific_name' for column 10, 'common_name' for column 11, or 'genus' for column 29."
        exit 1
        ;;
esac

if [ "$filter_column" -gt "$header_columns" ]; then
    echo "Error: Specified column number $filter_column exceeds the number of columns ($header_columns) in the file."
    exit 1
fi

echo "Sampling $sample_size rows with $column_to_filter equal to any of {$values_to_filter} using seed $random_seed..."

# Convert the comma-separated values into a regex pattern for awk
values_pattern=$(echo "$values_to_filter" | sed 's/,/|/g')

# Create a temporary file safely
temp_filtered_data=$(mktemp)

# Filter rows based on the specified column and values
awk -F ',' -v column="$filter_column" -v pattern="^(${values_pattern})\$" 'NR==1 || ($column ~ pattern)' "$input_file" > "$temp_filtered_data"

# Count the number of filtered rows excluding the header
filtered_row_count=$(tail -n +2 "$temp_filtered_data" | wc -l)
if [ "$filtered_row_count" -lt "$sample_size" ]; then
    echo "Error: Not enough rows to sample $sample_size rows. Only $filtered_row_count rows match the filter. Exiting."
    rm -f "$temp_filtered_data"
    exit 1
fi

# Create the output file name
sanitized_values=$(echo "$values_to_filter" | tr ' ' '_' | tr ',' '_')
output_file="${output_path}/${sanitized_values}_sample_${sample_size}.csv"

# Sample rows
{
    head -n 1 "$temp_filtered_data"
    tail -n +2 "$temp_filtered_data" | shuf --random-source=<(yes "$random_seed") -n "$sample_size"
} > "$output_file"

# Clean up temporary file
rm -f "$temp_filtered_data"

# Final message
echo "Sampling complete. Result written to $output_file."
