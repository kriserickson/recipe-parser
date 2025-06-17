# train.py
# Train an HTML block classifier using labeled JSON and HTML pairs

# Standard library imports
import json
import logging
import argparse
from pathlib import Path
from time import time
from typing import Dict, List, Tuple, Any

# Third-party imports
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed

# Local/application imports
from html_parser import parse_html
from feature_extraction import extract_features, build_transformer, preprocess_data

LABELS_DIR = Path("../data/labels")
HTML_DIR = Path("../data/html_pages")
MODEL_PATH = Path("../models/model.joblib")

def label_element(text: str, label_data: Dict[str, Any]) -> str:
    """
    Determine the label for a text element based on labeled data.

    Parameters
    ----------
    text : str
        The text content to be labeled.
    label_data : Dict[str, Any]
        Dictionary containing labeled data with keys: 'ingredients', 'directions', 'title'.

    Returns
    -------
    str
        Label for the text element: 'ingredient', 'direction', 'title', or 'none'.
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

def process_pair(json_file_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    base = json_file_path.stem
    html_file = HTML_DIR / f"{base}.html"
    if not html_file.exists():
        return [], []
    label_data = json.loads(json_file_path.read_text(encoding="utf-8"))
    html = html_file.read_text(encoding="utf-8")
    elements = parse_html(html)
    X, y = [], []
    for el in elements:
        label = label_element(el["text"], label_data)
        X.append(el)
        y.append(label)
    return X, y

def load_labeled_blocks(limit=None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load HTML blocks and their corresponding labels from JSON and HTML files.

    Parameters
    ----------
    limit : Optional[int], default=None
        Maximum number of elements to load, if specified.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[str]]
        A tuple containing features (X) and labels (y).
    """
    start = time()
    X, y = [], []
    json_files = sorted(LABELS_DIR.glob("recipe_*.json"))
    if limit is not None:
        json_files = json_files[:limit]
    total = len(json_files)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_pair, f) for f in json_files]
        for i, future in enumerate(as_completed(futures), 1):
            Xi, yi = future.result()
            X.extend(Xi)
            y.extend(yi)
            if i % 100 == 0 or i == total:
                percent = (i / total) * 100
                print(f"Processed {i}/{total} files ({percent:.1f}%)")

    print(f"Block loading time: {time() - start:.2f}s")
    return X, y

def validate_data(X: List, y: List) -> None:
    """
    Validate that the features and labels arrays have the same length.

    Parameters
    ----------
    X : List
        List of features.
    y : List
        List of labels.

    Raises
    ------
    ValueError
        If X and y have different lengths.
    """
    if len(X) != len(y):
        raise ValueError(f"X_train and y_train must have the same length. Got {len(X)} and {len(y)}")

def train(limit: int | None = None) -> None:
    """
    Train and save a text classification model for recipe components.

    This function loads labeled data, extracts features, trains a model,
    evaluates its performance, and saves the model to disk.

    Parameters
    ----------
    limit : Optional[int], default=None
        Maximum number of elements to load for training.
    """
    start = time()
    print("Loading labeled data...")
    X_raw, y = load_labeled_blocks(limit=limit)
    print(f"Loaded {len(X_raw)} blocks.")

    print("Extracting features...")
    X_features = extract_features(X_raw)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    logging.basicConfig(level=logging.INFO)

    validate_data(X_train, y_train)

    print("Preprocessing data...")
    X_train_proc = preprocess_data(X_train)
    X_test_proc = preprocess_data(X_test)

    print("Training model...")
    model = make_pipeline(
        build_transformer(),
        StandardScaler(with_mean=False),
        LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
    )

    model.fit(X_train_proc, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_proc)
    print(classification_report(y_test, y_pred))

    dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"️Total time: {time() - start:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a recipe component classifier.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of elements to load for training')
    args = parser.parse_args()

    train(limit=args.limit)

