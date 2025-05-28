"""
Feature extraction module for recipe parsing.

This module provides functions to extract features from HTML elements
and build feature processing pipelines for machine learning models.
"""

from typing import List, Dict, Any, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Constants
MAX_FEATURES: int = 1000
NGRAM_RANGE: tuple[int, int] = (1, 2)

def extract_features(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract features from HTML elements.

    Parameters
    ----------
    elements : List[Dict[str, Any]]
        List of dictionaries containing HTML element data with keys:
        'tag', 'depth', 'text'

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing extracted features

    Raises
    ------
    KeyError
        If required keys are missing from elements
    ValueError
        If elements list is empty or contains invalid data
    """
    if not elements:
        raise ValueError("Elements list cannot be empty")

    try:
        return [
            {
                "tag": el["tag"],
                "depth": el["depth"],
                "text_len": len(el["text"]),
                "starts_with_digit": bool(el["text"] and el["text"][0].isdigit()),
                "comma_count": el["text"].count(","),
                "dot_count": el["text"].count("."),
                "raw": el["text"]
            }
            for el in elements
        ]
    except KeyError as e:
        raise KeyError(f"Missing required key in element: {e}")

def build_feature_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline for feature processing.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline with TF-IDF vectorizer

    Notes
    -----
    The pipeline includes:
    - TF-IDF vectorization with n-grams
    - Feature limit of 1000 most frequent terms
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=NGRAM_RANGE,
            max_features=MAX_FEATURES
        ))
    ])