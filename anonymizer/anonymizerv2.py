import pandas as pd
import numpy as np
from collections import defaultdict


class MondrianKAnonymity:
    def __init__(self, k=30):
        """
        Initialize the Mondrian k-anonymity algorithm

        Args:
            k: The k value for k-anonymity (minimum group size)
        """
        self.k = k
        self.categorical_map = {}  # Maps categorical values to numerical values
        self.categorical_columns = []  # Keeps track of categorical columns
        self.feature_ranges = {}  # Stores ranges for generalization

    def fit_transform(self, data, qi_columns=None):
        """
        Apply Mondrian k-anonymity to the dataset

        Args:
            data: DataFrame to anonymize
            qi_columns: List of quasi-identifier columns (default: all columns)

        Returns:
            Anonymized DataFrame
        """
        # If no QI columns are specified, use all columns
        if qi_columns is None:
            qi_columns = data.columns.tolist()

        # Make a copy of the original data
        self.original_data = data.copy()
        self.quasi_identifiers = qi_columns

        # Identify categorical and numerical columns
        self.identify_column_types(data, qi_columns)

        # Preprocess data: convert categorical to numerical
        processed_data = self.preprocess_data(data, qi_columns)

        # Initial partition contains all records
        partition = processed_data.index.tolist()

        # Start recursive partitioning
        result = []
        self.partition_dataset(processed_data, partition, qi_columns, result)

        # Create anonymized dataframe
        anonymized_df = self.create_anonymized_dataframe(data, result)

        return anonymized_df

    def identify_column_types(self, data, columns):
        """Identify categorical and numerical columns"""
        self.categorical_columns = []
        self.numerical_columns = []

        for col in columns:
            # Try to convert to numeric
            try:
                pd.to_numeric(data[col])
                self.numerical_columns.append(col)
            except:
                self.categorical_columns.append(col)

    def preprocess_data(self, data, columns):
        """Convert categorical values to numerical for processing"""
        processed_data = data.copy()

        # Handle categorical columns
        for col in self.categorical_columns:
            if col in columns:
                # Create mapping for categorical values
                unique_values = sorted(data[col].unique())
                self.categorical_map[col] = {val: idx for idx, val in enumerate(unique_values)}

                # Convert categorical to numerical
                processed_data[col] = data[col].map(self.categorical_map[col])

                # Store original values for later generalization
                self.feature_ranges[col] = unique_values

        # Store ranges for numerical columns
        for col in self.numerical_columns:
            if col in columns:
                self.feature_ranges[col] = [data[col].min(), data[col].max()]

        return processed_data

    def choose_dimension(self, data, columns):
        """Choose the dimension (column) with the widest normalized range"""
        ranges = {}

        for col in columns:
            col_min = data[col].min()
            col_max = data[col].max()

            if col in self.categorical_columns:
                # For categorical, use the number of distinct values
                distinct_values = len(set(data[col]))
                if distinct_values <= 1:
                    ranges[col] = 0
                else:
                    ranges[col] = distinct_values / len(self.categorical_map[col])
            else:
                # For numerical, use normalized range
                if col_min == col_max:
                    ranges[col] = 0
                else:
                    feature_range = self.feature_ranges[col][1] - self.feature_ranges[col][0]
                    if feature_range == 0:
                        ranges[col] = 0
                    else:
                        ranges[col] = (col_max - col_min) / feature_range

        # Return the column with the widest normalized range
        if not ranges:
            return None

        return max(ranges.items(), key=lambda x: x[1])[0]

    def partition_dataset(self, data, partition, columns, result):
        """
        Recursively partition the dataset until k-anonymity is satisfied

        Args:
            data: Preprocessed DataFrame
            partition: List of indices in the current partition
            columns: List of columns to consider for partitioning
            result: List to store resulting partitions
        """
        # If partition size is less than 2k, don't split further
        if len(partition) < 2 * self.k:
            result.append(partition)
            return

        # Choose dimension (column) to split on
        dim = self.choose_dimension(data.loc[partition], columns)

        # If no suitable dimension is found, don't split
        if dim is None or data.loc[partition, dim].nunique() <= 1:
            result.append(partition)
            return

        # Find median value for the chosen dimension
        if dim in self.categorical_columns:
            # For categorical, split at median category index
            dim_values = data.loc[partition, dim].values
            unique_values = sorted(set(dim_values))
            split_value = unique_values[len(unique_values) // 2]
        else:
            # For numerical, split at median value
            split_value = data.loc[partition, dim].median()

        # Split the partition
        lhs = [idx for idx in partition if data.loc[idx, dim] <= split_value]
        rhs = [idx for idx in partition if idx not in lhs]

        # Check if split was successful
        if len(lhs) < self.k or len(rhs) < self.k:
            # If split doesn't satisfy k-anonymity, don't split
            result.append(partition)
        else:
            # Continue recursively
            self.partition_dataset(data, lhs, columns, result)
            self.partition_dataset(data, rhs, columns, result)

    def create_anonymized_dataframe(self, original_data, partitions):
        """
        Create anonymized dataframe by generalizing values within each partition

        Args:
            original_data: Original DataFrame
            partitions: List of partitions (lists of indices)

        Returns:
            Anonymized DataFrame
        """
        anonymized_data = original_data.copy()

        for partition in partitions:
            for col in self.quasi_identifiers:
                partition_data = original_data.loc[partition, col]

                if col in self.categorical_columns:
                    # For categorical, use the set of unique values
                    unique_values = sorted(partition_data.unique())
                    if len(unique_values) == 1:
                        generalized_value = unique_values[0]
                    else:
                        # Create range like "A-Z" or join with commas
                        if len(unique_values) <= 3:
                            generalized_value = ",".join(map(str, unique_values))
                        else:
                            generalized_value = f"{unique_values[0]}-{unique_values[-1]}"
                else:
                    # For numerical, use range
                    min_val = partition_data.min()
                    max_val = partition_data.max()
                    if min_val == max_val:
                        generalized_value = str(min_val)
                    else:
                        generalized_value = f"{min_val}-{max_val}"

                # Apply generalized value to all records in the partition
                anonymized_data.loc[partition, col] = generalized_value

        return anonymized_data


def parse_single_column_csv(filepath):
    """Parse a CSV file with a single column containing comma-separated values"""
    try:
        # Try standard CSV parsing first
        df = pd.read_csv(filepath)
        return df
    except:
        # If that fails, try parsing as a single column with comma-separated values
        try:
            df = pd.read_csv(filepath, header=None)

            # Extract headers from the first row
            headers = df.iloc[0, 0].split(',')

            # Process each row to split the comma-separated values
            rows = []
            for i in range(1, len(df)):
                values = df.iloc[i, 0].split(',')
                # Make sure each row has the correct number of columns
                if len(values) != len(headers):
                    values = values[:len(headers)] if len(values) > len(headers) else values + [''] * (
                                len(headers) - len(values))
                rows.append(values)

            # Create properly formatted DataFrame
            result_df = pd.DataFrame(rows, columns=headers)
            return result_df
        except Exception as e:
            print(f"Error parsing file: {e}")
            raise


if __name__ == "__main__":
    # Load the dataset
    filepath = "extracted_cleaned_data.csv"
    try:
        df = parse_single_column_csv(filepath)
        print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")

        # Apply Mondrian k-anonymity
        k = 30
        mondrian = MondrianKAnonymity(k=k)
        anonymized_df = mondrian.fit_transform(df)

        # Save anonymized data
        output_file = f"mondrian_k{k}_anonymized.csv"
        anonymized_df.to_csv(output_file, index=False)
        print(f"Anonymized data saved to {output_file}")

        # Print statistics
        print("\nStatistics:")
        print(f"Original unique combinations: {df.drop_duplicates().shape[0]}")
        print(f"Anonymized unique combinations: {anonymized_df.drop_duplicates().shape[0]}")

        # Verify k-anonymity
        # Count how many times each combination appears
        combination_counts = anonymized_df.groupby(list(anonymized_df.columns)).size()
        min_count = combination_counts.min() if not combination_counts.empty else 0

        print(f"\nMinimum group size: {min_count}")
        if min_count >= k:
            print(f" k-anonymity satisfied (k={k})")
        else:
            print(f" k-anonymity not satisfied (min={min_count}, required k={k})")

        # Show the 10 least common combinations
        print("\n10 least common combinations:")
        for i, (combo, count) in enumerate(combination_counts.sort_values().head(10).items(), 1):
            print(f"Combination #{i}: {count} occurrences")
            for col_name, value in zip(anonymized_df.columns, combo):
                print(f"  {col_name}: {value}")
            print()

    except Exception as e:
        print(f"An error occurred: {e}")