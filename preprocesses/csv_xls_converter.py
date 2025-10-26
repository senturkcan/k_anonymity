import pandas as pd

# Read your CSV file (it is already comma-separated)
df = pd.read_csv('C:/Users/canse/PycharmProjects/cdcanalysis/extracted_cleaned_data.csv')

# Save it as an Excel file
df.to_excel('C:/Users/canse/PycharmProjects/cdcanalysis/output.xlsx', index=False)

print("Done! Your Excel file has been created successfully.")
