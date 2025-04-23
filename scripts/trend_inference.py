import pandas as pd
from utils.dashboard_utils import load_final_model, predict_text

def main():
    # Load cleaned data
    df = pd.read_parquet('data/trends_raw.parquet')
    print(f"🔍 Loaded {len(df)} articles")

    # Load model & tokenizer
    tokenizer, model = load_final_model()
    print("🤖 Model loaded, starting inference...")

    # Prepare lists for results
    years, labels, confidences = [], [], []

    for text, year in zip(df['clean_text'], df['year']):
        label, confs = predict_text(text, tokenizer, model)
        years.append(year)
        labels.append(label)
        confidences.append(confs[label])

    # Build results DataFrame
    results = pd.DataFrame({
        'year': years,
        'predicted_label': labels,
        'confidence': confidences
    })
    results.to_parquet('data/trends_predictions.parquet', index=False)
    print("✅ Predictions saved to data/trends_predictions.parquet")

if __name__ == "__main__":
    main()
