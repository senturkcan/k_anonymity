import pandas as pd
import csv

# First, let's read the entire CSV file as a single text file
with open('2021.csv', 'r') as file:
    lines = file.readlines()

# Parse the header line to get feature names
header_line = lines[0].strip()
feature_names = header_line.split(',')

# Find indices of features we want to extract
columns_to_extract = ['MARITAL', 'EDUCA', 'EMPLOY1', 'INCOME3', '_SMOKER3', '_AGEG5YR', '_BMI5',
                      'ECIGNOW1', "FRUIT2", "FVGREEN1", "FRENCHF1", 'ALCDAY5',"_DRDXAR3","CHECKUP1",
                      "EXERANY2", "_ASTHMS1", "_RFHYPE6" , "_RFCHOL3", "ADDEPEV3", "DIFFWALK","GENHLTH"]
feature_indices = {}

for column in columns_to_extract:
    try:
        feature_indices[column] = feature_names.index(column)
    except ValueError:
        print(f"Warning: Column '{column}' not found in the data")

# Define unwanted values for specific columns
# Now including both formats (with and without decimal)
unwanted_values = {
    'MARITAL': ['9', '9.0'],
    'EDUCA': ['9', '9.0'],
    'EMPLOY1': ['9', '9.0'],
    'INCOME3': ['77', '77.0', '99', '99.0'],
    '_SMOKER3': ['9', '9.0'],
    'ALCDAY5': ['777', '777.0', '999', '999.0'],
    '_AGEG5YR': ['14', '14.0'],
    'ECIGNOW1': ['9', '9.0'],
    "FRUIT2" : ["777.0", "999.0"],
    "FVGREEN1" : ["777.0", "999.0"],
	"FRENCHF1": ["777.0", "999.0"],
    "CHECKUP1" : ["7.0","8.0","9.0"],
    "EXERANY2" : ["7.0", "9.0"],
    "_ASTHMS1" : ["9.0"],
    "_RFHYPE6" : ["9.0"],
    "_RFCHOL3": ["9.0"],
    "ADDEPEV3" : ["7.0", "9.0"],
    "DIFFWALK": ["7.0", "9.0"],
    "GENHLTH": ["7.0", "9.0"],
}

# Prepare the output data
output_rows = [columns_to_extract]  # First row is the header
filtered_count = 0
nan_count = 0
total_kept = 0

# For more detailed reporting
filter_reasons = {column: 0 for column in unwanted_values}

# Process each row (starting from row 2, index 1)
for i in range(1, len(lines)):
    row_values = lines[i].strip().split(',')

    # Check if the row is valid
    valid_row = True
    filter_reason = None

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
                nan_count += 1
                break

            # Check for unwanted values
            if column in unwanted_values and value in unwanted_values[column]:
                valid_row = False
                filter_reason = column
                break

            extracted_values.append(value)
        else:
            # Missing value - consider as nan
            valid_row = False
            nan_count += 1
            break

    # Add row to output if valid
    if valid_row and len(extracted_values) == len(columns_to_extract):
        output_rows.append(extracted_values)
        total_kept += 1
    elif filter_reason:
        filter_reasons[filter_reason] += 1
        filtered_count += 1

# Write to output file
with open('extracted_cleaned_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)

print(f"Extraction complete. New file 'extracted_data.csv' has been created.")
print(f"Total rows in original file: {len(lines) - 1}")
print(f"Rows kept after filtering: {total_kept}")
print(f"Rows removed due to 'nan' values: {nan_count}")
print(f"Rows removed due to unwanted values: {filtered_count}")
print("\nBreakdown of filtered rows by column:")
for column, count in filter_reasons.items():
    if count > 0:
        print(f"  {column}: {count} rows (unwanted values: {unwanted_values[column]})")