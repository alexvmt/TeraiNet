#!/bin/bash

# Check if the user has provided all required arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <input_file.csv> <column_to_filter> <values_to_filter> <output_file>"
    echo "<column_to_filter> should be 'scientific_name' (column 10), 'common_name' (column 11), or 'genus' (column 29)."
    echo "<values_to_filter> should be a comma-separated list of values (e.g., 'value1,value2')."
    exit 1
fi

# Store the input arguments
input_file="$1"
column_to_filter="$2"
values_to_filter="$3"
output_file="$4"

# Validate the input file
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found. Please provide a valid CSV file."
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

echo "Getting image URLs based on $column_to_filter equal to any of {$values_to_filter}..."

# Convert the comma-separated values into a regex pattern
values_pattern=$(echo "$values_to_filter" | sed 's/,/|/g')

# Filter rows based on the specified column and values
awk -F ',' -v column="$filter_column" -v pattern="^(${values_pattern})\$" 'NR==1 || ($column ~ pattern)' "$input_file" > "$output_file"

# Count the number of filtered rows without the header
filtered_row_count=$(tail -n +2 "$output_file" | wc -l)
echo "Total matching rows found: $filtered_row_count"

# Final message
echo "Filtering complete. Result written to $output_file."
