# scripts/trend_data_prep.py

import os
import pandas as pd
from utils.text_cleaner import clean_text

def load_and_clean(path, date_col='date', text_col='article_text'):
    df = pd.read_csv(path)
    df['year'] = pd.to_datetime(df[date_col]).dt.year
    df['clean_text'] = df[text_col].apply(clean_text)
    return df[['year', 'clean_text']]

if __name__ == "__main__":
    parts = [
        'data/guardian_2016_2022.csv',
        'data/guardian_2015.csv',
        'data/guardian_2023_2025.csv'
    ]
    dfs = []
    for p in parts:
        if os.path.exists(p):
            dfs.append(load_and_clean(p))
        else:
            print(f"⚠️  Source not found: {p}")
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet('data/trends_raw.parquet', index=False)
    print(f"✅ Saved {len(combined)} articles to data/trends_raw.parquet")
