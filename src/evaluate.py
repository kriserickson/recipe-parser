# evaluate.py
# Evaluate the trained HTML block classifier against labeled JSON + HTML pairs

from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from html_parser import parse_html
from feature_extraction import extract_features, build_transformer

import json
from pathlib import Path
import pandas as pd

LABELS_DIR = Path("../data/labels")
HTML_DIR = Path("../data/html_pages")


def label_element(text, label_data):
    """
    Given a block of text and JSON labels, determine its label.
    Return 'ingredient', 'direction', 'title', or 'none'.
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


def load_labeled_blocks():
    """
    Load all labeled HTML blocks with features and true labels
    """
    X, y = [], []
    for json_file in sorted(LABELS_DIR.glob("recipe_*.json")):
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
    return X, y


def evaluate():
    # Load labeled HTML text chunks and their true labels
    X_raw, y = load_labeled_blocks()
    X_features = extract_features(X_raw)

    # Convert list of dicts to DataFrame if needed
    if isinstance(X_features, list) and isinstance(X_features[0], dict):
        X_features = pd.DataFrame(X_features)

    print(f"Feature matrix shape: {X_features.shape}, type: {type(X_features)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    # Build model pipeline
    model = make_pipeline(
        build_transformer(),
        LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate()