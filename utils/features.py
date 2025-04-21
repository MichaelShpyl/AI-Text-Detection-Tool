"""
Functions to compute additional textual features (readability, sentiment, lexical diversity).
These can help in exploratory analysis or alternative modeling approaches.
"""
from textstat import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import pandas as pd
# Initialize VADER sentiment analyzer once
_analyzer = SentimentIntensityAnalyzer()

def compute_readability(text):
    """
    Compute the Flesch Reading Ease score of the given text.
    A higher score indicates easier readability.
    Returns:
        float: The readability score (Flesch Reading Ease).
    """
    if not text or not isinstance(text, str):
        return 0.0
    # textstat.flesch_reading_ease returns a float (higher = easier)
    try:
        score = textstat.flesch_reading_ease(text)
    except Exception:
        score = 0.0
    return score

def compute_sentiment(text):
    """
    Compute a sentiment compound score for the text using VADER.
    Returns:
        float: Sentiment compound score in [-1, 1].
    """
    if not text or not isinstance(text, str):
        return 0.0
    scores = _analyzer.polarity_scores(text)
    return scores.get('compound', 0.0)

def compute_lexical_diversity(text):
    """
    Compute lexical diversity of the text as the ratio of unique words to total words.
    Returns:
        float: Lexical diversity ratio (0 to 1).
    """
    if not text or not isinstance(text, str):
        return 0.0
    # Tokenize by word using a simple regex (alphanumeric words)
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) == 0:
        return 0.0
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens)

def compute_all_features(df):
    """
    Compute all supported features for each text in the DataFrame.
    Adds columns 'readability', 'sentiment', and 'lexical_diversity' to the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with a 'text' column (and optionally 'label').
    Returns:
        pd.DataFrame: DataFrame with new feature columns added.
    """
    if 'text' not in df.columns:
        raise KeyError("DataFrame must contain a 'text' column.")
    # Compute each feature and add as new column
    df['readability'] = df['text'].apply(compute_readability)
    df['sentiment'] = df['text'].apply(compute_sentiment)
    df['lexical_diversity'] = df['text'].apply(compute_lexical_diversity)
    return df
