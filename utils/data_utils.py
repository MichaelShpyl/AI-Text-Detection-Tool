import re
from bs4 import BeautifulSoup
# We assume NLTK and langdetect are installed for optional use
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.stem import WordNetLemmatizer
except ImportError:
    WordNetLemmatizer = None

def clean_text(text, do_lower=True, remove_html=True, remove_non_ascii=True, do_lemmatize=False):
    """Clean text by lowercasing, removing HTML tags, non-ASCII chars, URLs, extra whitespace. 
       Optionally lemmatize words."""
    if not isinstance(text, str):
        text = str(text)
    # Lowercase
    if do_lower:
        text = text.lower()
    # Remove HTML tags
    if remove_html:
        text = BeautifulSoup(text, "html.parser").get_text()
    # Remove non-ASCII or non-printable characters
    if remove_non_ascii:
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lemmatization 
    if do_lemmatize and WordNetLemmatizer is not None:
        lemmatizer = WordNetLemmatizer()
        lem_words = [lemmatizer.lemmatize(word) for word in text.split()]
        text = " ".join(lem_words)
    return text

def detect_language(text):
    """Detect language code for a given text. Requires langdetect library. Returns 'en' or language code."""
    try:
        from langdetect import detect
    except ImportError:
        return "unknown"
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    return lang
