import math
from textstat import flesch_reading_ease, flesch_kincaid_grade
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER only once
_vader = SentimentIntensityAnalyzer()

def get_readability(text):
    """Return a tuple of (flesch_reading_ease, flesch_kincaid_grade) for the text."""
    try:
        ease = flesch_reading_ease(text)
        grade = flesch_kincaid_grade(text)
    except Exception as e:
        # textstat can sometimes error on weird input; handle gracefully
        ease, grade = 0.0, 0.0
    return ease, grade

def get_sentiment_score(text):
    """Return compound sentiment score of text using VADER."""
    scores = _vader.polarity_scores(text)
    return scores['compound']

def get_lexical_diversity(text):
    """Compute simple lexical diversity (type-token ratio)."""
    words = text.split()
    if len(words) == 0:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

def get_avg_word_length(text):
    """Average character length of words in text."""
    words = text.split()
    if len(words) == 0:
        return 0.0
    return sum(len(w) for w in words) / len(words)

# NER count stub (could integrate spaCy if time allows)
def get_ner_count(text):
    """Count named entities in text (stub implementation: returns 0 or uses simple heuristic)."""
    return 0  # Will integrate spaCy or another NER tool later
