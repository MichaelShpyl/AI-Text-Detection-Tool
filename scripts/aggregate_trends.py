import pandas as pd

# Load raw prediction records (assumes predictions.parquet with columns ['year', 'label'])
df = pd.read_parquet('data/predictions.parquet')

# Map labels to readable names if needed
label_map = {
    'human': 'Human-written',
    'ai_paraphrased': 'AI-paraphrased',
    'ai_generated': 'AI-generated'
}
df['label'] = df['label'].map(label_map)

# Count occurrences per year and label
counts = df.groupby(['year', 'label']).size().reset_index(name='count')

# Pivot to wide format
pivot = counts.pivot(index='year', columns='label', values='count').fillna(0)
pivot = pivot.rename_axis(None, axis=1).reset_index()

# Ensure all expected columns exist
for col in ['Human-written', 'AI-paraphrased', 'AI-generated']:
    if col not in pivot:
        pivot[col] = 0

# Calculate totals and percentages
pivot['total_count'] = pivot[['Human-written', 'AI-paraphrased', 'AI-generated']].sum(axis=1)
pivot['human_percent'] = pivot['Human-written'] / pivot['total_count']
pivot['ai_paraphrased_percent'] = pivot['AI-paraphrased'] / pivot['total_count']
pivot['ai_generated_percent'] = pivot['AI-generated'] / pivot['total_count']

# Sort by year and save to CSV
pivot = pivot.sort_values('year')
pivot.to_csv('data/trends_by_year.csv', index=False)
