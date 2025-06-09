"""
Recipe content prediction and extraction module.

This module provides functionality to predict and extract structured content
from raw HTML recipe pages using a trained model.
"""

import json
import sys
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from joblib import load

from html_parser import parse_html
from config import MODEL_PATH
from feature_extraction import extract_features
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_features_and_text(features: list[dict]) -> Tuple[list[dict], list[str]]:
    """
    Split a list of feature dicts into two lists: one without the 'text' field, and one with just the text values.

    Parameters
    ----------
    features : list[dict]
        List of feature dictionaries, each containing a 'text' key.

    Returns
    -------
    Tuple[list[dict], list[str]]
        Tuple of (features without text, list of text values).
    """
    features_wo_text = []
    texts = []
    for feat in features:
        texts.append(feat['text'])
        f = feat.copy()
        del f['text']
        features_wo_text.append(f)
    return features_wo_text, texts

def clean_prediction(text: str, label: str) -> bool:
    """
    Determine if a predicted text block should be included in the structured output.

    This function applies a set of heuristics to filter out noisy, irrelevant, or misclassified predictions
    for recipe titles, ingredients, and directions. It is used to post-process model predictions before
    returning them to the user.

    Parameters
    ----------
    text : str
        The text content of the predicted block.
    label : str
        The predicted label for the block (e.g., 'title', 'ingredient', 'direction').

    Returns
    -------
    bool
        True if the prediction should be included in the output, False if it should be filtered out.
    """
    text_lower = text.lower().strip()

    # Skip empty or too short
    if not text or len(text) < 3:
        return False

    # Global stopwords for both directions and ingredients
    global_blocklist = {
        "by", "news", "trends", "or", "and", "text ingredients", "sponsored", "advertisement"
    }

    if text_lower in global_blocklist:
        return False

    # Ingredient-specific cleanup
    if label == "ingredient":
        # Reject headings accidentally classified as ingredients
        if re.fullmatch(r"(ingredients|yield|nutrition|directions|preparation|prep time|cook time|total time|servings)", text_lower):
            return False
        # Reject single words (likely noise)
        if len(text.split()) < 2:
            return False

    # Direction-specific cleanup
    if label == "direction":
        if re.fullmatch(r"(ingredients|yield|nutrition|prep time|cook time|total time|servings)", text_lower):
            return False
        # Reject single words
        if len(text.split()) < 3:
            return False

    # Title is tricky — don't filter aggressively
    # You can add special rules here if desired

    return True

def preprocess_data(features: list[dict], use_nlp_features: bool) -> Any:
    """
    Preprocess features for model prediction, optionally including NLP features.

    Parameters
    ----------
    features : list[dict]
        List of feature dictionaries.
    use_nlp_features : bool
        Whether to include NLP/text features in the output.

    Returns
    -------
    Any
        Preprocessed features, either as a list of dicts or list of (dict, text) tuples.
    """
    if use_nlp_features:
        features_wo_text, texts = split_features_and_text(features)
        return list(zip(features_wo_text, texts))
    else:
        features_wo_text, _ = split_features_and_text(features)
        return features_wo_text

# NOTE: Add this local version of ItemSelector to make sure model loading works
# Otherwise joblib.load() cannot resolve ItemSelector if not imported in predict.py

class ItemSelector(BaseEstimator, TransformerMixin):
    """
    For FeatureUnion to extract a specific item from a tuple or dict in the pipeline.

    Parameters
    ----------
    key : str
        The key to select ('structured' or 'text').
    """
    def __init__(self, key: str):
        self.key = key
    def fit(self, X: Any, y: Any = None) -> 'ItemSelector':
        return self
    def transform(self, X: Any) -> list:
        if self.key == 'structured':
            return [x[0] for x in X]  # structured features (dict)
        elif self.key == 'text':
            return [x[1] for x in X]  # text field
        else:
            raise ValueError(f"Unknown key: {self.key}")

def extract_structured_data(html_path: Path) -> Dict[str, Optional[List[str]]]:
    """
    Extract structured recipe data from an HTML file using a trained model.

    Parameters
    ----------
    html_path : Path
        Path to the HTML file to process.

    Returns
    -------
    Dict[str, Optional[List[str]]]
        Dictionary with keys 'title', 'ingredients', and 'directions'.
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

        text = el["text"].strip()
        if not clean_prediction(text, label):
            continue

        if label == "title" and structured["title"] is None:
            structured["title"] = text
        elif label == "ingredient":
            structured["ingredients"].append(text)
        elif label == "direction":
            structured["directions"].append(text)

    logger.info("Successfully extracted structured data")
    return structured

def main() -> None:
    """
    Main entry point for the prediction script. Handles command-line arguments and runs extraction.
    """
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