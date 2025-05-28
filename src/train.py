"""
Train an HTML block classifier using labeled JSON and HTML pairs.

This module handles the training pipeline for classifying HTML blocks
into recipe components (ingredients, directions, title, etc.) using
supervised learning.
"""

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

from html_parser import parse_html
from feature_extraction import extract_features, build_feature_pipeline

import json
from pathlib import Path
from time import time
import logging
import numpy as np
import pandas as pd

LABELS_DIR = Path("../data/labels")
HTML_DIR = Path("../data/html_pages")
MODEL_PATH = Path("../models/model.joblib")

def label_element(text: str, label_data: dict) -> str:
    """
    Classify a text element based on recipe label data.

    Args:
        text: The text content to classify
        label_data: Dictionary containing recipe components (ingredients, directions, title)

    Returns:
        str: Classification label ('ingredient', 'direction', 'title', or 'none')
    """
    t = text.strip().lower()
    if not t or t.isdigit():
        return 'none'
    if any(t in i.lower() for i in label_data.get("ingredients", [])):
        return 'ingredient'
    if any(t in d.lower() for d in label_data.get("directions", [])):
        return 'direction'
    if label_data.get("title", "").strip().lower() == t:
        return 'title'
    return 'none'

def load_labeled_blocks(limit: int | None = None) -> tuple[list, list]:
    """
    Load and parse labeled HTML blocks from files.

    Args:
        limit: Optional maximum number of files to process

    Returns:
        tuple: (features_list, labels_list) containing the training data
    """
    features_list, labels_list = [], []
    json_files = sorted(LABELS_DIR.glob("recipe_*.json"))
    total = len(json_files)

    if limit:
        total = min(total, limit)

    report_size = max(10, int(total / 100))  # Report every 1% of total files

    for i, json_file in enumerate(json_files):
        if limit and i >= limit:
            break

        base = json_file.stem
        html_file = HTML_DIR / f"{base}.html"
        if not html_file.exists():
            continue

        label_data = json.loads(json_file.read_text(encoding="utf-8"))
        html = html_file.read_text(encoding="utf-8")
        elements = parse_html(html)

        for el in elements:
            label = label_element(el["text"], label_data)
            features_list.append(el)
            labels_list.append(label)

        if (i + 1) % report_size == 0 or (i + 1) == total:
            percent = ((i + 1) / total) * 100
            print(f"📦 Processed {i + 1}/{total} files ({percent:.1f}%)")
    return features_list, labels_list

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

def train() -> None:
    """
    Execute the complete training pipeline.
    
    Loads data, extracts features, trains the model, and saves it to disk.
    Prints progress and evaluation metrics throughout the process.
    """
    start = time()
    print("🔄 Loading labeled data...")
    X_raw, y = load_labeled_blocks(limit=100)
    print(f"✅ Loaded {len(X_raw)} blocks.")

    print("🔧 Extracting features...")
    X_features = [str(f) for f in extract_features(X_raw)]  # Ensure X_features is a list of strings

    print("🎯 Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"X_train length: {len(X_train)}")
    logging.info(f"y_train length: {len(y_train)}")
    validate_data(X_train, y_train)

    print("🧠 Training model...")
    model = make_pipeline(
        build_feature_pipeline(),
        LogisticRegression(max_iter=1000, class_weight='balanced')
    )

    print(f"X_train length: {len(X_train)}, type: {type(X_train)}")
    print(f"y_train length: {len(y_train)}, type: {type(y_train)}")
    print(f"X_test length: {len(X_test)}, type: {type(X_test)}")
    print(f"y_test length: {len(y_test)}, type: {type(y_test)}")

    model.fit(X_train, y_train)

    print("📊 Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"⏱️ Total time: {time() - start:.2f}s")

if __name__ == "__main__":
    train()
