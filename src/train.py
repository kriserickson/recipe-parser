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

def preprocess_data(features):
    features_wo_text, texts = split_features_and_text(features)
    # FeatureUnion requires parallel lists, passed as tuples
    combined = list(zip(features_wo_text, texts))
    return combined

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


def balance_training_data(
    x_train: list,
    y_train: list,
    *,
    ratio_none_to_minor: int = 3,
    min_target_per_class: int = 1_000,
    random_state: int = 42,
):
    """
    • Keeps all minority-class rows.
    • Down-samples 'none' so:   len(none) <= ratio_none_to_minor * len(all minorities)
    • Up-samples a minority class if it has < min_target_per_class rows.
    Prints for each class: upsampled/downsampled/unchanged.
    """
    import pandas as pd
    from sklearn.utils import resample

    df = pd.DataFrame(x_train)
    df["label"] = y_train

    counts = df["label"].value_counts().to_dict()
    n_none = counts.get("none", 0)
    n_minor = sum(v for k, v in counts.items() if k != "none")

    print("\n📊 raw class counts")
    for k, v in counts.items():
        print(f"  {k:<10}: {v}")

    # -- keep all minorities
    df_minor = df[df["label"] != "none"].copy()

    # -- none: downsample if needed
    none_target = min(n_none, ratio_none_to_minor * len(df_minor))
    action_none = "unchanged"
    if none_target < n_none:
        action_none = "downsampled"
    elif none_target > n_none:
        action_none = "upsampled"

    df_none = df[df["label"] == "none"].copy()
    df_none_down = resample(
        df_none, replace=False, n_samples=none_target, random_state=random_state
    )

    print(f"  {'none':<10}: {n_none} → {none_target} ({action_none})")

    # -- handle minorities
    frames = [df_none_down]
    for label, orig_count in counts.items():
        if label == "none":
            continue

        df_label = df_minor[df_minor["label"] == label]
        new_count = orig_count
        action = "unchanged"
        # Upsample tiny minorities if needed
        if orig_count < min_target_per_class:
            df_label = resample(
                df_label,
                replace=True,
                n_samples=min_target_per_class,
                random_state=random_state,
            )
            new_count = min_target_per_class
            action = "upsampled"
        elif orig_count > min_target_per_class:
            # Optionally downsample massive classes (rare for minorities, usually not needed)
            # df_label = resample(
            #     df_label,
            #     replace=False,
            #     n_samples=min_target_per_class,
            #     random_state=random_state,
            # )
            # new_count = min_target_per_class
            # action = "downsampled"
            pass  # currently not downsampling minorities

        frames.append(df_label)
        print(f"  {label:<10}: {orig_count} → {new_count} ({action})")

    # -- concat & shuffle
    df_bal = pd.concat(frames).sample(frac=1, random_state=random_state)

    print(f"\n🟢 Final balanced set: {len(df_bal)} rows "
          f"( none={len(df_none_down)}, minorities={len(df_bal)-len(df_none_down)} )")

    y_bal = df_bal["label"].tolist()
    x_bal = df_bal.drop("label", axis=1).to_dict(orient="records")
    return x_bal, y_bal




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

def build_pipeline():
    dict_vect = DictVectorizer(sparse=True)
    transformer = FeatureUnion([
        ("structured", make_pipeline(ItemSelector('structured'), dict_vect)),
        ("text", make_pipeline(ItemSelector('text'), TfidfVectorizer(max_features=500, ngram_range=(1, 2))))
    ])
    pipeline = make_pipeline(
        transformer,
        RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ),
    )
    return pipeline

def train(limit: int | None = None) -> None:
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

    print(f"🧠 Preprocessing data...")
    x_train_proc = preprocess_data(x_train_bal)
    x_test_proc = preprocess_data(x_test)

    print("🧠 Building model pipeline...")
    model = build_pipeline()

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
    train()
