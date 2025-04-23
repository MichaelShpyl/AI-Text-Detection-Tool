#!/usr/bin/env python
"""
trends_analysis.py

Loads multiple news-article sources (Guardian CSV plus any extras),
cleans text, runs the AI Text Detector model on each article, aggregates
counts and percentages by year, and writes out data/trends_by_year.csv.
"""

import os
import glob
import pandas as pd
from utils.text_cleaner import clean_text
from utils.dashboard_utils import load_final_model, predict_text

# 1. Load raw data files
#   - Guardian dataset (CSV)
#   - Any additional JSON/CSV files placed in data/news_extra/
guardian_path = "data/guardian_news.csv"
extra_pattern = "data/news_extra/*.*"

print("Loading datasets...")
df_list = []
if os.path.exists(guardian_path):
    gdf = pd.read_csv(guardian_path, parse_dates=["date"])
    df_list.append(gdf)
for fp in glob.glob(extra_pattern):
    if fp.lower().endswith(".csv"):
        df_list.append(pd.read_csv(fp, parse_dates=["date"]))
    elif fp.lower().endswith(".json"):
        df_list.append(pd.read_json(fp))
if not df_list:
    raise FileNotFoundError("No news data files found in data/ folder.")

df = pd.concat(df_list, ignore_index=True)
print(f"Total articles loaded: {len(df)}")

# 2. Clean text and extract year
print("Cleaning text and extracting year...")
df["clean_text"] = df["article_text"].astype(str).apply(clean_text)
df["year"] = df["date"].dt.year

# Keep only 2015–2025
df = df[(df["year"] >= 2015) & (df["year"] <= 2025)].reset_index(drop=True)
print(f"Articles in 2015–2025 range: {len(df)}")

# 3. Load model and tokenizer once
print("Loading model and tokenizer...")
tokenizer, model = load_final_model()  # returns (tokenizer, model)
model.eval()

# 4. Predict labels for each article
print("Classifying articles (this may take a while)...")
preds = []
for idx, row in df.iterrows():
    label, _probs = predict_text(row["clean_text"], tokenizer, model)
    preds.append((row["year"], label))
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx+1}/{len(df)} articles")

pred_df = pd.DataFrame(preds, columns=["year","label"])

# 5. Map labels to human-readable and aggregate counts
print("Aggregating counts by year and label...")
label_map = {
    "human":         "Human-written",
    "ai_paraphrased":"AI-paraphrased",
    "ai_generated":  "AI-generated"
}
pred_df["label_hr"] = pred_df["label"].map(label_map)

counts = (
    pred_df
    .groupby(["year","label_hr"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Ensure all three columns exist
for col in label_map.values():
    if col not in counts:
        counts[col] = 0

counts = counts[["year",
                 "Human-written",
                 "AI-paraphrased",
                 "AI-generated"]]

# 6. Compute totals and percentages
counts["total"] = (
    counts["Human-written"] +
    counts["AI-paraphrased"] +
    counts["AI-generated"]
)
counts["human_percent"]        = counts["Human-written"]       / counts["total"]
counts["ai_paraphrased_percent"]= counts["AI-paraphrased"]      / counts["total"]
counts["ai_generated_percent"]  = counts["AI-generated"]        / counts["total"]

# 7. Save to CSV
out_path = "data/trends_by_year.csv"
print(f"Saving aggregated trends to {out_path} …")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
counts.to_csv(out_path, index=False)

print("Done.")
