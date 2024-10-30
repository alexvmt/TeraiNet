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

echo "Sampling $sample_size rows with $column_to_filter equal to any of {$values_to_filter} using seed $random_seed..."

# Determine which column to filter on based on the second argument
if [ "$column_to_filter" == "scientific_name" ]; then
    filter_column=10
elif [ "$column_to_filter" == "common_name" ]; then
    filter_column=11
elif [ "$column_to_filter" == "genus" ]; then
    filter_column=29
else
    echo "Invalid column_to_filter: choose 'scientific_name' for column 10, 'common_name' for column 11, or 'genus' for column 29."
    exit 1
fi

# Convert the comma-separated values into a regex pattern for awk
values_pattern=$(echo "$values_to_filter" | sed 's/,/|/g')

# Extract the header separately
header=$(head -n 1 "$input_file")

# Filter rows based on the chosen column and multiple values, excluding the header
filtered_data=$(awk -F ',' -v column="$filter_column" -v pattern="^(${values_pattern})\$" \
    'NR > 1 && ($column ~ pattern)' "$input_file")

# Count the number of filtered rows
filtered_row_count=$(echo "$filtered_data" | wc -l)

# Print the number of filtered rows
echo "Number of filtered rows (excluding header) to sample from: $filtered_row_count"

# Check if there are enough rows to sample from
if [ "$filtered_row_count" -lt "$sample_size" ]; then
    echo "Not enough rows to sample $sample_size rows. Exiting."
    exit 1
fi

# Create a dynamic output file name based on values to filter, joined by underscores
filtered_values=$(echo "$values_to_filter" | sed 's/,/_/g')
output_file="${output_path}/${filtered_values}_sample_${sample_size}.csv"

# Write the header to the output file, then shuffle and sample the filtered data
echo "$header" > "$output_file"
echo "$filtered_data" | shuf -n "$sample_size" --random-source=<(yes "$random_seed") >> "$output_file"

# Final message without output length
echo "Sampling complete. Result written to $output_file."
