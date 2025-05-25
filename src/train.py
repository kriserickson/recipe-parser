# train.py
# Train an HTML block classifier using labeled JSON and HTML pairs

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

def label_element(text, label_data):
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

def load_labeled_blocks(limit=None):
    X, y = [], []
    json_files = sorted(LABELS_DIR.glob("recipe_*.json"))
    total = len(json_files)
    if limit:
        total = min(total, limit)

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
            X.append(el)
            y.append(label)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            percent = ((i + 1) / total) * 100
            print(f"📦 Processed {i + 1}/{total} files ({percent:.1f}%)")
    return X, y

def validate_data(X, y):
    """Validate data formats allowing for lists as well as numpy arrays"""
    if len(X) != len(y):
        raise ValueError(f"X_train and y_train must have the same length. Got {len(X)} and {len(y)}")
    
    # Additional validation can be added here if needed
    # But we won't require numpy arrays or pandas DataFrames anymore

def train():
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
