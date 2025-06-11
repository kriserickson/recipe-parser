from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# This example assumes we extract raw text and tag type (e.g., <h1>, <li>, <p>) from HTML elements
class ItemSelector(BaseEstimator, TransformerMixin):
    """
    For FeatureUnion to extract a specific item from a tuple or dict in the pipeline.
    """
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.key == 'structured':
            return [x[0] for x in X]  # structured features (dict)
        elif self.key == 'text':
            return [x[1] for x in X]  # text field
        else:
            raise ValueError(f"Unknown key: {self.key}")

def split_features_and_text(features):
    """Splits features into (dicts without 'text', texts)."""
    features_wo_text = []
    texts = []
    for feat in features:
        texts.append(feat['raw'])
        f = feat.copy()
        del f['raw']
        features_wo_text.append(f)
    return features_wo_text, texts

def preprocess_data(features):
    features_wo_text, texts = split_features_and_text(features)
    # FeatureUnion requires parallel lists, passed as tuples
    combined = list(zip(features_wo_text, texts))
    return combined

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

def build_transformer():
    dict_vect = DictVectorizer(sparse=True)
    transformer = FeatureUnion([
        ("structured", make_pipeline(ItemSelector('structured'), dict_vect)),
        ("text", make_pipeline(ItemSelector('text'), TfidfVectorizer(max_features=500, ngram_range=(1, 2))))
    ])

    return transformer