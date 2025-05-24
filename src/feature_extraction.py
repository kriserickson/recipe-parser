from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import re

# This example assumes we extract raw text and tag type (e.g., <h1>, <li>, <p>) from HTML elements

def extract_features(elements):
      return [
        {
            "tag": el["tag"],
            "depth": el["depth"],
            "text_len": len(el["text"]),
            "starts_with_digit": el["text"][0].isdigit(),
            "comma_count": el["text"].count(","),
            "dot_count": el["text"].count("."),
            "raw": el["text"]
        }
        for el in elements
    ]

def build_feature_pipeline():
    """Build a basic sklearn Pipeline with TF-IDF features."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000))
    ])