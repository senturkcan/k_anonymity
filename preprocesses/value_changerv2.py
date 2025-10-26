import pandas as pd

# Find the position of ALCDAY5 in the header
with open('modified_alcday5_data.csv', 'r') as file:
    header = file.readline().strip()

header_fields = header.split(',')
alcday5_index = header_fields.index('ALCDAY5')

# Read the file
rows = []
with open('modified_alcday5_data.csv', 'r') as file:
    for i, line in enumerate(file):
        if i == 0:  # Add header row unchanged
            rows.append(line.strip())
        else:
            fields = line.strip().split(',')

            # Check if we have enough fields and process ALCDAY5 value
            if len(fields) > alcday5_index:
                try:
                    alcday_value = float(fields[alcday5_index])

                    # Apply transformation rules
                    if 100.0 <= alcday_value <= 110.0:
                        alcday_value = (alcday_value - 100.0) * 4
                    elif 200.0 <= alcday_value <= 240.0:
                        alcday_value = alcday_value - 200.0

                    # Update the value in the row
                    fields[alcday5_index] = str(alcday_value)
                except ValueError:
                    # Skip transformation if value is not a valid float
                    pass

            rows.append(','.join(fields))

# Write the modified data back to a new file
with open('transformed_alcday5_data.csv', 'w', newline='') as file:
    for row in rows:
        file.write(row + '\n')

print("Processing complete. Modified data saved to 'transformed_alcday5_data.csv'")