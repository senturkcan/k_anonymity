import pandas as pd

# Load the Excel file
file_path = "new_dataset.xlsx"
df = pd.read_excel(file_path)

# Ensure we target the 15th column (index 20)
genhlth_col = df.columns[20]

# Apply the transformation
def adjust_genhlth(value):
    if value in [2,3]:
        return value - 1
    elif value in [4]:
        return value - 2
    elif value in [5]:
        return value - 3
    else:
        return value

df[genhlth_col] = df[genhlth_col].apply(adjust_genhlth)

# Save the updated DataFrame to a new file (optional)
df.to_excel("2value_dataset.xlsx", index=False)

print(f"GENHLTH column '{genhlth_col}' updated successfully.")