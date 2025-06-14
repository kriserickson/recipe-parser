from typing import Any, Dict, List, Tuple, Union
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class ItemSelector(BaseEstimator, TransformerMixin):
    """
    For FeatureUnion to extract a specific item from a tuple or dict in the pipeline.
    """
    def __init__(self, key: str):
        """
        Initialize the ItemSelector.

        Args:
            key (str): The key to select ('structured' or 'text').
        """
        self.key = key

    def fit(self, X: List[Any], y: Any = None) -> 'ItemSelector':
        """
        Fit method (does nothing, present for compatibility).

        Args:
            X (List[Any]): Input data.
            y (Any, optional): Target values (ignored).

        Returns:
            ItemSelector: self
        """
        return self

    def transform(self, X: List[Tuple[Dict[str, Any], str]]) -> List[Union[Dict[str, Any], str]]:
        """
        Transform the input data by selecting the specified key.

        Args:
            X (List[Tuple[Dict[str, Any], str]]): Input data as list of tuples.

        Returns:
            List[Union[Dict[str, Any], str]]: Selected data based on key.
        """
        if self.key == 'structured':
            return [x[0] for x in X]  # structured features (dict)
        elif self.key == 'text':
            return [x[1] for x in X]  # text field
        else:
            raise ValueError(f"Unknown key: {self.key}")

def split_features_and_text(features: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Splits features into two lists: one with dicts (without 'raw'), and one with the 'raw' text.

    Args:
        features (List[Dict[str, Any]]): List of feature dictionaries.

    Returns:
        Tuple[List[Dict[str, Any]], List[str]]: Tuple of (features without 'raw', list of raw texts).
    """
    features_wo_text = []
    texts = []
    for feat in features:
        texts.append(feat['raw'])
        f = feat.copy()
        del f['raw']
        features_wo_text.append(f)
    return features_wo_text, texts

def preprocess_data(features: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
    """
    Prepares data for feature extraction by splitting and zipping features and texts.

    Args:
        features (List[Dict[str, Any]]): List of feature dictionaries.

    Returns:
        List[Tuple[Dict[str, Any], str]]: List of tuples (features, text).
    """
    features_wo_text, texts = split_features_and_text(features)
    # FeatureUnion requires parallel lists, passed as tuples
    combined = list(zip(features_wo_text, texts))
    return combined

def extract_features(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts structured features from a list of elements.

    Args:
        elements (List[Dict[str, Any]]): List of elements, each with 'tag', 'depth', and 'text'.

    Returns:
        List[Dict[str, Any]]: List of feature dictionaries with extracted features.
    """
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

def build_transformer() -> FeatureUnion:
    """
    Builds a FeatureUnion transformer for structured and text features.

    Returns:
        FeatureUnion: A scikit-learn FeatureUnion object combining structured and text pipelines.
    """
    dict_vect = DictVectorizer(sparse=True)
    transformer = FeatureUnion([
        ("structured", make_pipeline(ItemSelector('structured'), dict_vect)),
        ("text", make_pipeline(ItemSelector('text'), TfidfVectorizer(max_features=500, ngram_range=(1, 2))))
    ])

    return transformer

