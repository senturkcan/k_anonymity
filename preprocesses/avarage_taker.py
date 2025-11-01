import pandas as pd
import numpy as np

# Load Excel
df = pd.read_excel('C:/Users/canse/PycharmProjects/cdcanalysis/k=200_anonymity_with_genILoss.xlsx')

# replace with their average
def replace_with_average(x):
    if isinstance(x, str) and '-' in x:
        try:
            parts = x.split('-')
            numbers = [float(part.strip()) for part in parts]
            return np.mean(numbers)
        except:
            return x  # If it fails (e.g., non-numeric values), return original
    return x  # If no hyphen, return original

# Apply to the whole dataframe
df = df.applymap(replace_with_average)

# Save the modified file if needed
df.to_excel('C:/Users/canse/PycharmProjects/cdcanalysis/anonymized_newdataset.xlsx', index=False)

print("Done! The updated file is saved as 'anonymized_newdataset.xlsx'.")
