"""
Train an HTML block classifier using labeled JSON and HTML pairs.

This module handles the training pipeline for classifying HTML blocks
into recipe components (ingredients, directions, title, etc.) using
supervised learning.
"""

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

from feature_extraction import build_feature_pipeline, load_labeled_blocks, filter_valid_features
from config import LABEL_DIR, HTML_DIR, MODEL_PATH

from time import time
import logging
import numpy as np




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
    X_raw, y = load_labeled_blocks(LABEL_DIR, HTML_DIR)
    print(f"✅ Loaded {len(X_raw)} blocks.")
    #
    # print("🔧 Extracting features...")
    # X_features = [str(f) for f in X_raw]  # Ensure X_features is a list of strings

    X_raw, y = filter_valid_features(X_raw, y)

    print("🎯 Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"X_train length: {len(X_train)}")
    logging.info(f"y_train length: {len(y_train)}")
    validate_data(X_train, y_train)

    print("🧠 Training model...")
    model = make_pipeline(
        DictVectorizer(sparse=True),
        LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga')
    )

    model.fit(X_train, y_train)

    print("📊 Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"⏱️ Total time: {time() - start:.2f}s")

if __name__ == "__main__":
    train()
