"""
Utility functions for loading and preparing dataset splits.
Reads configuration to avoid hardcoded file paths.
"""

import pandas as pd
import glob
import yaml

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
