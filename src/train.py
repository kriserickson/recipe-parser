# train.py
# Train an HTML block classifier using labeled JSON and HTML pairs

# Standard library imports
import json
import logging
import argparse
import os
import pandas as pd
import pickle

from pathlib import Path
from time import time
from typing import Dict, List, Tuple, Any

# Third-party imports
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher

from config import HTML_DIR, MODEL_PATH, LABEL_DIR, CACHE_DIR
# Local/application imports
from html_parser import parse_html
from feature_extraction import extract_features, build_transformer, preprocess_data, get_section_header

def similar(a: str, b: str) -> float:
    """
    Compute the similarity ratio between two strings using SequenceMatcher.

    Parameters
    ----------
    a : str
        First string to compare.
    b : str
        Second string to compare.

    Returns
    -------
    float
        Similarity ratio between 0.0 and 1.0, where 1.0 means identical strings.
    """
    return SequenceMatcher(None, a, b).ratio()

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
    if any(similar(t, i.lower()) > 0.8 for i in label_data.get("ingredients", [])):
        return 'ingredient'
    if any(similar(t, d.lower()) > 0.8 for d in label_data.get("directions", [])):
        return 'direction'
    if similar(label_data.get("title", "").strip().lower(), t) > 0.8:
        return 'title'
    return 'none'

def process_pair(json_file_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    base = json_file_path.stem
    html_file = HTML_DIR / f"{base}.html"
    if not html_file.exists():
        return [], []

    X, y = [], []

    try:
        label_data = json.loads(json_file_path.read_text(encoding="utf-8"))
        html = html_file.read_text(encoding="utf-8")
        elements = parse_html(html)

        current_section_heading = None

        for idx, el in enumerate(elements):

            elem_text = el["text"]
            label = label_element(elem_text, label_data)

            current_section_heading = get_section_header(current_section_heading, el)

            features = extract_features(el, idx, elements, current_section_heading)
            X.append(features)
            y.append(label)

    except Exception as e:
        print(f"Error processing {json_file_path.name}: {e}")

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
    cache_file = os.path.join(CACHE_DIR, f"labeled_blocks_limit_{limit}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    start = time()
    X, y = [], []
    json_files = sorted(LABEL_DIR.glob("recipe_*.json"))
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

    # If limit is set, apply it
    if limit is not None:
        X = X[:limit]
        y = y[:limit]

    with open(cache_file, "wb") as f:
        pickle.dump((X, y), f)

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


def balance_training_data(
    x_train: list,
    y_train: list,
    *,
    ratio_none_to_minor: int = 3,
    random_state: int = 42,
) -> tuple[list, list]:
    """
    Balance the training data by downsampling the majority class ('none') and upsampling minority classes.

    This function ensures that the 'none' class does not dominate the training set by downsampling it
    to a ratio relative to the total number of minority class samples. Minority classes are upsampled
    so that each has at least 33% of the mean count of all non-'none' classes.

    Parameters
    ----------
    x_train : list
        List of feature dictionaries for training samples.
    y_train : list
        List of labels corresponding to x_train.
    ratio_none_to_minor : int, optional
        The maximum allowed ratio of 'none' class samples to the total number of minority class samples (default: 3).
    random_state : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    tuple[list, list]
        A tuple containing the balanced feature list and label list: (x_bal, y_bal).
    """
    df = pd.DataFrame(x_train)
    df["label"] = y_train

    counts = df["label"].value_counts().to_dict()
    n_none = counts.get("none", 0)

    print("\nraw class counts")
    for k, v in counts.items():
        print(f"  {k:<10}: {v}")

    # -- keep all minorities
    df_minor = df[df["label"] != "none"].copy()
    minor_counts = df_minor["label"].value_counts().to_dict()
    if minor_counts:
        mean_minor = sum(minor_counts.values()) / len(minor_counts)
        min_target_per_class = int(mean_minor * 0.33)
    else:
        mean_minor = 0
        min_target_per_class = 0

    print(f"Mean non-'none' class count: {mean_minor:.2f}")
    print(f"Target minimum per class (33% of mean): {min_target_per_class}")

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
    for label, orig_count in minor_counts.items():
        df_label = df_minor[df_minor["label"] == label]
        new_count = orig_count
        action = "unchanged"
        # Upsample if needed
        if orig_count < min_target_per_class:
            df_label = resample(
                df_label,
                replace=True,
                n_samples=min_target_per_class,
                random_state=random_state,
            )
            new_count = min_target_per_class
            action = "upsampled"
        frames.append(df_label)
        print(f"  {label:<10}: {orig_count} → {new_count} ({action})")

    # -- concat & shuffle
    df_bal = pd.concat(frames).sample(frac=1, random_state=random_state)

    print(f"\nFinal balanced set: {len(df_bal)} rows "
          f"( none={len(df_none_down)}, minorities={len(df_bal)-len(df_none_down)} )")

    y_bal = df_bal["label"].tolist()
    x_bal = df_bal.drop("label", axis=1).to_dict(orient="records")
    return x_bal, y_bal



def train(limit: int | None = None, memory: bool = False) -> None:
    """
    Train and save a text classification model for recipe components.

    This function loads labeled data, extracts features, trains a model,
    evaluates its performance, and saves the model to disk.

    Parameters
    ----------
    limit : Optional[int], default=None
        Maximum number of elements to load for training.
    memory: [Optional[bool]], default=False
        Whether to track memory usage during training.
    """

    # Start tracking time and memory usage if requested
    start = time()
    if memory:
        import psutil
        import tracemalloc

        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB

        tracemalloc.start()
        start_memory = get_memory_usage()

    print("Loading labeled data...")
    X_features, y = load_labeled_blocks(limit=limit)
    print(f"Loaded {len(X_features)} blocks.")

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    logging.basicConfig(level=logging.INFO)

    X_train_bal, y_train_bal = balance_training_data(X_train, y_train)

    validate_data(X_train_bal, y_train_bal)

    print("Preprocessing data...")
    X_train_proc = preprocess_data(X_train_bal)
    X_test_proc = preprocess_data(X_test)

    print("Training model...")
    model = make_pipeline(
        build_transformer(),
        GradientBoostingClassifier(random_state=42)
    )

    model.fit(X_train_proc, y_train_bal)

    print("Evaluating...")
    y_pred = model.predict(X_test_proc)
    print(classification_report(y_test, y_pred))

    dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} with a size of {os.path.getsize(MODEL_PATH) / 1024 / 1024:.3f} MB")
    print(f"️Total time: {time() - start:.2f}s")


    if memory:
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        end_memory = get_memory_usage()
        print(f"Peak memory from tracemalloc: {peak / 1024 / 1024:.2f} MB")
        print(f"Memory usage from psutil: {end_memory:.2f} MB")
        print(f"Memory increase: {end_memory - start_memory:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a recipe component classifier.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of elements to load for training')
    parser.add_argument('--memory', action="store_true", help='Track memory usage during training')
    args = parser.parse_args()

    train(limit=args.limit, memory=args.memory)
