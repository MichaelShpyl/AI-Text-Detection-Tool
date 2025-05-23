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
    Supports melting human_written / ai_paraphrased / ai_generated into text+label.
    """
    # 1) Melt the three variants if they exist
    variants = ['human_written', 'ai_paraphrased', 'ai_generated']
    text_columns = [c for c in variants if c in df.columns]
    if text_columns:
        flat_df = df.melt(
            value_vars=text_columns,
            var_name='label',
            value_name='text',
            ignore_index=True
        )
    # 2) Already flat?
    elif 'text' in df.columns:
        flat_df = df.copy()
    # 3) Fallback: combine title/content/body
    else:
        text_columns = [col for col in df.columns if col.lower() in ('title', 'content', 'body')]
        if not text_columns:
            raise ValueError(f"No text column found to flatten; expected one of {variants!r} or title/content/body.")
        flat_df = df.copy()
        flat_df['text'] = flat_df[text_columns] \
            .apply(lambda row: ' '.join(str(v) for v in row if not pd.isna(v)), axis=1)
        flat_df.drop(columns=text_columns, inplace=True)

    # Ensure the label column exists
    if 'label' not in flat_df.columns:
        # maybe your original label was called category or class?
        label_cols = [c for c in flat_df.columns if c.lower() in ('label', 'category', 'class')]
        if label_cols:
            flat_df.rename(columns={label_cols[0]: 'label'}, inplace=True)
        else:
            raise ValueError("No label column found in dataset.")

    # Reorder so text+label come first
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


def load_modern_articles():
    """
    Load any new "modern articles" from the specified directory.
    This function looks for CSV files in the modern_data_dir and concatenates them.
    Returns:
        pd.DataFrame: DataFrame of modern articles with a 'text' column (and no labels).
    """
    modern_dir = config['paths']['modern_data_dir']
    files = glob.glob(f"{modern_dir}/*.csv")
    articles = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Assume each modern article file has at least a 'text' column
            if 'text' not in df.columns:
                # Flatten if needed (similar approach as flatten_dataset)
                text_cols = [c for c in df.columns if c.lower() in ('title', 'content', 'body')]
                if text_cols:
                    df['text'] = df[text_cols].apply(lambda row: ' '.join(str(val) for val in row if not pd.isna(val)), axis=1)
            articles.append(df[['text']].copy())
        except Exception as e:
            print(f"[data_utils] Warning: Skipping file {file} due to read error: {e}")
    if articles:
        modern_df = pd.concat(articles, ignore_index=True)
        print(f"[data_utils] Loaded {modern_df.shape[0]} modern articles from {len(files)} file(s).")
        return modern_df
    else:
        print("[data_utils] No modern article files found.")
        return pd.DataFrame(columns=['text'])
