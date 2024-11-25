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

# Validate the output directory
if [ ! -d "$output_path" ]; then
    echo "Error: Output path $output_path does not exist. Exiting."
    exit 1
fi

echo "Sampling $sample_size rows with $column_to_filter equal to any of {$values_to_filter} using seed $random_seed..."

# Determine which column to filter on based on the second argument
case "$column_to_filter" in
    "scientific_name") filter_column=10 ;;
    "common_name") filter_column=11 ;;
    "genus") filter_column=29 ;;
    *)
        echo "Invalid column_to_filter: choose 'scientific_name' for column 10, 'common_name' for column 11, or 'genus' for column 29."
        exit 1
        ;;
esac

# Convert the comma-separated values into a regex pattern for awk
values_pattern=$(echo "$values_to_filter" | sed 's/,/|/g')

# Extract the header and filtered rows in one pass
{
    IFS= read -r header
    echo "$header" > "${output_path}/temp_header.csv"
    awk -F ',' -v column="$filter_column" -v pattern="^(${values_pattern})\$" '($column ~ pattern)' >> "${output_path}/temp_filtered_data.csv"
} < "$input_file"

# Count the number of filtered rows
filtered_row_count=$(wc -l < "${output_path}/temp_filtered_data.csv")

# Print the number of filtered rows
echo "Number of filtered rows (excluding header) to sample from: $filtered_row_count"

# Check if there are enough rows to sample from
if [ "$filtered_row_count" -lt "$sample_size" ]; then
    echo "Not enough rows to sample $sample_size rows. Exiting."
    rm -f "${output_path}/temp_header.csv" "${output_path}/temp_filtered_data.csv"
    exit 1
fi

# Create a dynamic output file name based on sanitized values
sanitized_values=$(echo "$values_to_filter" | tr ' ' '_' | tr ',' '_')
output_file="${output_path}/${sanitized_values}_sample_${sample_size}.csv"

# Shuffle and sample rows
shuf --random-source=<(yes "$random_seed") -n "$sample_size" "${output_path}/temp_filtered_data.csv" > "${output_path}/temp_sample.csv"

# Combine header and sampled data
cat "${output_path}/temp_header.csv" "${output_path}/temp_sample.csv" > "$output_file"

# Clean up temporary files
rm -f "${output_path}/temp_header.csv" "${output_path}/temp_filtered_data.csv" "${output_path}/temp_sample.csv"

# Final message
echo "Sampling complete. Result written to $output_file."
