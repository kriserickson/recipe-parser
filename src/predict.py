"""
Recipe content prediction and extraction module.

This module provides functionality to predict and extract structured content
from raw HTML recipe pages using a trained model.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

from joblib import load

from html_parser import parse_html
from config import MODEL_PATH
from feature_extraction import extract_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_features_and_text(features):
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
        return list(zip(features_wo_text, texts))
    else:
        features_wo_text, _ = split_features_and_text(features)
        return features_wo_text

# NOTE: Add this local version of ItemSelector to make sure model loading works
# Otherwise joblib.load() cannot resolve ItemSelector if not imported in predict.py
from sklearn.base import BaseEstimator, TransformerMixin

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

def extract_structured_data(html_path: Path) -> Dict[str, Optional[List[str]]]:
    """
    Extract structured recipe data from HTML file.
    """
    try:
        html = html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"HTML file not found: {html_path}")
        raise

    logger.info(f"Processing HTML file: {html_path}")
    elements = parse_html(html)
    all_features = []
    for idx, el in enumerate(elements):
        text_elem = el.get("text", "").strip()
        features = extract_features(el, text_elem, elements, idx)
        all_features.append(features)

    try:
        model = load(MODEL_PATH)
    except FileNotFoundError:
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise

    # --- Detect if model expects nlp tuple ---
    use_tuple = False
    if hasattr(model, 'steps') and hasattr(model.steps[0][1], 'transformer_list'):
        names = [name for name, _ in model.steps[0][1].transformer_list]
        if 'text' in names and 'structured' in names:
            use_tuple = True

    features_proc = preprocess_data(all_features, use_nlp_features=use_tuple)
    predictions = model.predict(features_proc)

    structured: Dict[str, Optional[List[str]]] = {
        "title": None,
        "ingredients": [],
        "directions": []
    }

    for el, label in zip(elements, predictions):
        if label == "title" and structured["title"] is None:
            structured["title"] = el["text"]
        elif label == "ingredient":
            structured["ingredients"].append(el["text"])
        elif label == "direction":
            structured["directions"].append(el["text"])

    logger.info("Successfully extracted structured data")
    return structured

def main() -> None:
    if len(sys.argv) != 2:
        logger.error("Invalid number of arguments")
        print("Usage: python predict.py path/to/recipe.html")
        sys.exit(1)

    try:
        result = extract_structured_data(Path(sys.argv[1]))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error processing recipe: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()