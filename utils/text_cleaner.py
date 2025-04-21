"""
Text cleaning utility.
Cleans raw text by lowercasing, removing unwanted characters, etc.
Lemmatization can be optionally applied in a later step.
"""
import re

try:
    import spacy
    # Load spaCy English model if available (to be done once)
    _nlp = spacy.load("en_core_web_sm")
except ImportError:
    _nlp = None

def clean_text(text, lemmatize=False):
    """
    Clean a text string by removing or normalizing unwanted characters.
    Args:
        text (str): The raw text to clean.
        lemmatize (bool): If True, perform lemmatization (reduce words to their lemma).
                          This initial version ignores lemmatization (to be added later).
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # Normalize whitespace
    text = text.replace("\n", " ").replace("\r", " ")
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Basic normalization: lowercase
    text = text.lower()
    # (Lemmatization step will be added in a later commit if needed)
    if lemmatize and _nlp:
    # Use spaCy to lemmatize if model is loaded
    doc = _nlp(text)
    # Join lemmas of tokens that are alphabetic
    text = " ".join(token.lemma_ for token in doc)
    return text
