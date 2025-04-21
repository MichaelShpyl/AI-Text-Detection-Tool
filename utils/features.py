"""
Functions to compute additional textual features (readability, sentiment, lexical diversity).
These can help in exploratory analysis or alternative modeling approaches.
"""
from textstat import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
