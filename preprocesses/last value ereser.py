import pandas as pd

# Load the CSV file
file_path = "transformed_alcday5_data.csv"
df = pd.read_csv(file_path, dtype=str)

# Remove the last column
df = df.iloc[:, :-1]

# Save the cleaned file
df.to_csv("cleaned_data.csv", index=False, header=False)
