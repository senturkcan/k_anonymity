import pandas as pd

# Load the Excel file
file_path = "new_dataset.xlsx.xlsx"
df = pd.read_excel(file_path)

# Ensure we target the 15th column (index 14)
income3_col = df.columns[3]

# Apply the transformation
def adjust_income3(value):
    if value in [2]:
        return value - 1
    elif value in [3]:
        return value - 2
    elif value in [4]:
        return value - 3
    elif value in [5, 6]:
        return value - 4
    elif value in [7, 8]:
        return value - 5
    elif value in [9]:
        return value - 6
    elif value in [10]:
        return value - 7
    elif value in [11]:
        return value - 8
    else:
        return value

df[income3_col] = df[income3_col].apply(adjust_income3)

# Save the updated DataFrame to a new file (optional)
df.to_excel("updated_non_anonymized_mlready.xlsx", index=False)

print(f"income3 column '{income3_col}' updated successfully.")