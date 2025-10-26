import pandas as pd
import csv
from collections import Counter

# First, let's read the entire CSV file as a single text file
with open('2021.csv', 'r') as file:
    lines = file.readlines()

# Parse the header line to get feature names
header_line = lines[0].strip()
feature_names = header_line.split(',')

# Find indices of features we want to extract
columns_to_extract = ['MARITAL', 'EDUCA', 'EMPLOY1', 'INCOME3', '_SMOKER3', 'ALCDAY5', '_METSTAT', '_AGEG5YR', '_BMI5',
                      'ECIGNOW1']
feature_indices = {}

for column in columns_to_extract:
    try:
        feature_indices[column] = feature_names.index(column)
    except ValueError:
        print(f"Warning: Column '{column}' not found in the data")

# Define unwanted values for specific columns
unwanted_values = {
    'MARITAL': ['9', '9.0'],
    'EDUCA': ['9', '9.0'],
    'EMPLOY1': ['9', '9.0'],
    'INCOME3': ['77', '77.0', '99', '99.0'],
    '_SMOKER3': ['9', '9.0'],
    'ALCDAY5': ['777', '777.0'],
    '_AGEG5YR': ['14', '14.0'],
    'ECIGNOW1': ['9', '9.0']
}

# Extract and filter rows
extracted_rows = []
row_indices = []  # To keep track of original row indices

for i in range(1, len(lines)):
    row_values = lines[i].strip().split(',')

    # Check if the row is valid
    valid_row = True

    # Extract values for our target columns
    extracted_values = []

    for column in columns_to_extract:
        idx = feature_indices[column]

        # Check if index is within range
        if idx < len(row_values):
            value = row_values[idx]

            # Check for nan
            if value.lower() == 'nan':
                valid_row = False
                break

            # Check for unwanted values
            if column in unwanted_values and value in unwanted_values[column]:
                valid_row = False
                break

            extracted_values.append(value)
        else:
            # Missing value - consider as nan
            valid_row = False
            break

    # Add row to output if valid
    if valid_row and len(extracted_values) == len(columns_to_extract):
        # Convert to tuple so it can be used as a dictionary key
        extracted_rows.append(tuple(extracted_values))
        row_indices.append(i)  # Store the original row index

# Count duplicates
row_counter = Counter(extracted_rows)

# Identify only the duplicate rows (rows that appear more than once)
duplicates = [row for row, count in row_counter.items() if count > 1]

# Create a list of all duplicate instances
all_duplicate_instances = []
for i, row in enumerate(extracted_rows):
    if row in duplicates:
        all_duplicate_instances.append((row_indices[i], row))

# Create a new CSV with only the duplicate rows
with open('duplicate_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns_to_extract)  # Write header

    # Write all instances of the duplicate rows
    for _, row in all_duplicate_instances:
        writer.writerow(row)

# Statistics
total_rows = len(extracted_rows)
unique_rows = len(row_counter)
duplicate_rows = total_rows - unique_rows
duplicate_patterns = len(duplicates)
duplicate_instances = len(all_duplicate_instances)

print(f"Analysis of dataset:")
print(f"Total rows (after filtering): {total_rows}")
print(f"Unique row patterns: {unique_rows}")
print(f"Number of row patterns that appear multiple times: {duplicate_patterns}")
print(f"Total instances of duplicate rows: {duplicate_instances}")
print(f"Created 'duplicate_data.csv' with {duplicate_instances} rows that are duplicates.")

# Optional: Show some examples of the duplicates
if duplicate_patterns > 0:
    print("\nExamples of duplicate patterns:")
    for i, row in enumerate(duplicates[:5]):  # Show up to 5 examples
        count = row_counter[row]
        print(f"\nDuplicate pattern #{i + 1} (appears {count} times):")
        for j, column in enumerate(columns_to_extract):
            print(f"  {column}: {row[j]}")