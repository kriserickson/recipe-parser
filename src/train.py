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
    for i, json_file in enumerate(json_files):
        if limit and len(X) >= limit:
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
    if len(X) != len(y):
        raise ValueError(f"X_train and y_train must have the same length. Got {len(X)} and {len(y)}")

def extract_features(elements):
    """
    Convert raw HTML elements into text features for the model.
    """
    return [el["text"] for el in elements]  # Extract only the text field from each element

def train(limit: int | None = None):
    start = time()
    print("🔄 Loading labeled data...")
    X_raw, y = load_labeled_blocks(limit=limit)
    print(f"✅ Loaded {len(X_raw)} blocks.")

    print("🔧 Extracting features...")
    X_features = extract_features(X_raw)  # Ensure X_features is a list of strings

    print("🎯 Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    logging.basicConfig(level=logging.INFO)

    validate_data(X_train, y_train)

    print("🧠 Training model...")
    model = make_pipeline(
        build_feature_pipeline(),
        LogisticRegression(max_iter=1000, class_weight='balanced')
    )

    model.fit(X_train, y_train)

    print("📊 Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"⏱️ Total time: {time() - start:.2f}s")

if __name__ == "__main__":
    train(limit=1000)

