"""
Text cleaning utility.
Cleans raw text by lowercasing, removing unwanted characters, etc.
Lemmatization can be optionally applied in a later step.
"""
import re

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
    return text
