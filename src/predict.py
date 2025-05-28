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
from feature_extraction import extract_features
from config import MODEL_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_structured_data(html_path: Path) -> Dict[str, Optional[List[str]]]:
    """
    Extract structured recipe data from HTML file.

    Parameters
    ----------
    html_path : Path
        Path to HTML file containing recipe

    Returns
    -------
    Dict[str, Optional[List[str]]]
        Dictionary containing structured recipe data with keys:
        - 'title': Optional[str]
        - 'ingredients': List[str]
        - 'directions': List[str]

    Raises
    ------
    FileNotFoundError
        If HTML file doesn't exist
    ValueError
        If HTML content is invalid
    """
    try:
        html = html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"HTML file not found: {html_path}")
        raise

    logger.info(f"Processing HTML file: {html_path}")
    elements = parse_html(html)
    features = extract_features(elements)

    try:
        model = load(MODEL_PATH)
    except FileNotFoundError:
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise

    predictions = model.predict(features)

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
    """
    Main entry point for the recipe prediction script.
    
    Usage: python predict.py path/to/recipe.html
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

