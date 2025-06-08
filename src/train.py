"""
Train an HTML block classifier using labeled JSON and HTML pairs.

This module handles the training pipeline for classifying HTML blocks
into recipe components (ingredients, directions, title, etc.) using
supervised learning.
"""

import gc
import logging
import time
from typing import List, Dict, Any, Generator

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion

from config import HTML_DIR, LABEL_DIR, MODEL_PATH
from feature_extraction import (filter_valid_features, load_labeled_blocks)

def split_features_and_text(features):
    """Splits features into (dicts without 'text', texts)."""
    features_wo_text = []
    texts = []
    for feat in features:
        texts.append(feat['text'])
        f = feat.copy()
        del f['text']
        features_wo_text.append(f)
    return features_wo_text, texts


def preprocess_data(features, use_nlp_features):
    if use_nlp_features:
        features_wo_text, texts = split_features_and_text(features)
        # FeatureUnion requires parallel lists, passed as tuples
        combined = list(zip(features_wo_text, texts))
        return combined
    else:
        features_wo_text, _ = split_features_and_text(features)
        return features_wo_text

def validate_data(features: list | np.ndarray, labels: list | np.ndarray) -> None:
    """
    Validate that features and labels meet the required format.

    Args:
        features: Training features as list or numpy array
        labels: Training labels as list or numpy array

    Raises:
        ValueError: If data validation fails
    """
    if len(features) != len(labels):
        raise ValueError(
            f"Features and labels must have the same length. "
            f"Got {len(features)} and {len(labels)}"
        )


def batch_generator(
    features: List[Dict[str, Any]],
    labels: List[str],
    batch_size: int
) -> Generator[tuple[list[dict[str, Any]], list[str]], Any, None]:
    """
    Yield batches of features and labels.

    Args:
        features: List of feature dicts.
        labels: List of labels.
        batch_size: Size of each batch.

    Yields:
        Tuple of (features_batch, labels_batch).
    """
    for i in range(0, len(features), batch_size):
        yield features[i:i + batch_size], labels[i:i + batch_size]



def balance_training_data(x_train, y_train, max_samples_per_class=None):
    # Put into DataFrame for easier manipulation
    df = pd.DataFrame(x_train)
    df['label'] = y_train

    # Find the size of the minority (non-none) classes
    class_counts = df['label'].value_counts()
    min_count = min(class_counts[class_counts.index != 'none'])
    # Optionally, cap the max per class to save memory
    n_samples = max_samples_per_class or min_count

    # Downsample "none"
    df_none = df[df['label'] == 'none']
    df_none_down = resample(df_none, replace=False, n_samples=n_samples, random_state=42)

    # Upsample each minority class to n_samples
    frames = [df_none_down]
    for label in class_counts.index:
        if label == 'none': continue
        df_label = df[df['label'] == label]
        df_label_up = resample(df_label, replace=True, n_samples=n_samples, random_state=42)
        frames.append(df_label_up)

    # Concatenate and shuffle
    df_bal = pd.concat(frames).sample(frac=1, random_state=42)

    # Return balanced x_train/y_train
    y_train_bal = df_bal['label'].tolist()
    x_train_bal = df_bal.drop('label', axis=1).to_dict(orient='records')
    return x_train_bal, y_train_bal

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

def build_pipeline(use_nlp_features: bool):
    dict_vect = DictVectorizer(sparse=True)
    if use_nlp_features:
        transformer = FeatureUnion([
            ("structured", make_pipeline(ItemSelector('structured'), dict_vect)),
            ("text", make_pipeline(ItemSelector('text'), TfidfVectorizer(max_features=500, ngram_range=(1, 2))))
        ])
    else:
        transformer = dict_vect

    pipeline = make_pipeline(
        transformer,
        RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ),
    )
    return pipeline

def train(limit: int | None = None, batch_size: int = 10000, use_nlp_features: bool = False) -> None:
    start_time = time.time()
    print("🔄 Loading labeled data...")
    x_raw, y = load_labeled_blocks(LABEL_DIR, HTML_DIR, limit=limit)
    print(f"✅ Loaded {len(x_raw)} blocks.")

    x_raw, y = filter_valid_features(x_raw, y)

    print("🎯 Splitting train/test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    y_test = np.array(y_test)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"X_train length: {len(x_train)}")
    logging.info(f"y_train length: {len(y_train)}")

    x_train_bal, y_train_bal = balance_training_data(x_train, y_train)

    logging.info(f"X_train length: {len(x_train_bal)}")
    logging.info(f"y_train length: {len(y_train_bal)}")

    validate_data(x_train_bal, y_train_bal)

    print(f"🧠 Preprocessing data (use_nlp_features={use_nlp_features})...")
    x_train_proc = preprocess_data(x_train_bal, use_nlp_features)
    x_test_proc = preprocess_data(x_test, use_nlp_features)

    print("🧠 Building model pipeline...")
    model = build_pipeline(use_nlp_features)

    print("🧠 Training model...")
    # No more batch training—fit all at once for pipeline compatibility
    model.fit(x_train_proc, y_train_bal)

    print("📊 Evaluating...")
    y_pred = model.predict(x_test_proc)
    print(classification_report(y_test, y_pred, zero_division=0))

    dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"⏱️ Total time: {time.time() - start_time:.2f}s")



if __name__ == "__main__":
    train(limit=1000, use_nlp_features=True)