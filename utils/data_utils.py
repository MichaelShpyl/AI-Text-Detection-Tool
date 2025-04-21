"""
Utility functions for loading and preparing dataset splits.
Reads configuration to avoid hardcoded file paths.
"""

import pandas as pd
import glob
import yaml
from sklearn.model_selection import train_test_split
# Load configuration once at module import
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
_DATA_PATH = config['paths']['raw_data']

def load_raw_data():
    """
    Load the raw dataset from CSV into a pandas DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing the raw dataset.
    """
    # Read the raw CSV containing all data (human, paraphrased, AI texts)
    df = pd.read_csv(_DATA_PATH)
    # Logging the shape for debug (to training.log if needed)
    print(f"[data_utils] Loaded raw data: {df.shape[0]} records, {df.shape[1]} columns.")
    return df

def flatten_dataset(df):
    """
    Flatten raw dataset into a standard format with 'text' and 'label' columns.
    If multiple text columns (e.g., 'title' and 'content') exist, combine them.
    Args:
        df (pd.DataFrame): Raw dataset DataFrame.
    Returns:
        pd.DataFrame: Flattened DataFrame with at least 'text' and 'label' columns.
    """
    # If the DataFrame already has a 'text' column, assume it's flat
    if 'text' in df.columns:
        flat_df = df.copy()
    else:
        # If not, attempt to find common text fields to concatenate (e.g., title + content)
        text_columns = [col for col in df.columns if col.lower() in ('title', 'content', 'body', 'text')]
        if text_columns:
            # Combine columns into one text (separated by newline)
            flat_df = df.copy()
            flat_df['text'] = flat_df[text_columns].apply(lambda row: ' '.join(str(val) for val in row if not pd.isna(val)), axis=1)
            flat_df.drop(columns=text_columns, inplace=True)
        else:
            # If no obvious text fields, raise an error
            raise ValueError("No text column found to flatten.")
    # Ensure the label column is named 'label'
    if 'label' not in flat_df.columns:
        # Identify label-like column (e.g., 'category' or 'class')
        label_cols = [col for col in flat_df.columns if col.lower() in ('label', 'category', 'class')]
        if label_cols:
            flat_df.rename(columns={label_cols[0]: 'label'}, inplace=True)
        else:
            raise ValueError("No label column found in dataset.")
    # Reorder columns to have 'text' and 'label' first for clarity
    cols = ['text', 'label'] + [c for c in flat_df.columns if c not in ('text', 'label')]
    flat_df = flat_df[cols]
    print(f"[data_utils] Flattened dataset: {flat_df.shape[0]} records with columns {list(flat_df.columns)}")
    return flat_df


def train_val_test_split(df, val_fraction=0.1, test_fraction=0.1, random_state=42):
    """
    Split the DataFrame into training, validation, and test sets.
    Args:
        df (pd.DataFrame): Cleaned dataset with 'text' and 'label'.
        val_fraction (float): Proportion of data to use for validation.
        test_fraction (float): Proportion of data to use for test.
        random_state (int): Seed for reproducibility.
    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame): train_df, val_df, test_df splits.
    """
    # First split off the test set from the full dataset
    train_val_df, test_df = train_test_split(
        df, test_size=test_fraction, stratify=df['label'], random_state=random_state)
    # Now split the remaining into train and val
    val_size = val_fraction / (1 - test_fraction)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, stratify=train_val_df['label'], random_state=random_state)
    # Reset indices for cleanliness
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    print(f"[data_utils] Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")
    return train_df, val_df, test_df