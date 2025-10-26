import pandas as pd
from collections import Counter


def find_least_common_combinations(filepath, n=10):
    """
    Find the n least common combinations in the dataset and report how many times each occurs.

    Args:
        filepath: Path to the CSV file
        n: Number of least common combinations to find (default: 10)
    """
    # Read the CSV file
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading standard CSV format: {e}")
        try:
            # Try alternative parsing for single-column format
            df = pd.read_csv(filepath, header=None)
            headers = df.iloc[0, 0].split(',')
            rows = []
            for i in range(1, len(df)):
                values = df.iloc[i, 0].split(',')
                rows.append(values)
            df = pd.DataFrame(rows, columns=headers)
            print(f"Successfully loaded single-column dataset with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e2:
            print(f"Alternative parsing also failed: {e2}")
            return

    # Convert each row to a tuple of values to make it hashable for counting
    row_tuples = [tuple(row) for _, row in df.iterrows()]

    # Count occurrences of each unique combination
    combination_counts = Counter(row_tuples)

    # Find the n combinations with lowest counts
    least_common = combination_counts.most_common()[:-n - 1:-1]

    print(f"\n===== {n} Least Common Combinations =====")
    print(f"Total unique combinations in dataset: {len(combination_counts)}")

    print("\nLeast common combinations (sorted by frequency):")
    for i, (combination, count) in enumerate(least_common, 1):
        print(f"\nCombination #{i}: Occurs {count} times")
        for col_name, value in zip(df.columns, combination):
            print(f"  {col_name}: {value}")

    # Summary statistics
    counts = [count for _, count in least_common]
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        print(f"\nSummary of {n} least common combinations:")
        print(f"  Minimum occurrences: {min_count}")
        print(f"  Maximum occurrences: {max_count}")
        print(f"  Average occurrences: {avg_count:.2f}")

    # Check if k-anonymity (k=30) is maintained
    if min_count < 30:
        print("\n K-anonymity (k=30) check: FAILED")
        violations = sum(1 for _, count in combination_counts.items() if count < 30)
        print(f"  {violations} combinations appear fewer than 30 times")
    else:
        print("\n K-anonymity (k=30) check: PASSED")
        print("  All combinations appear at least 30 times")


if __name__ == "__main__":
    # Analyze the anonymized dataset
    filepath = "anonymizedk30.csv"
    find_least_common_combinations(filepath, n=10)