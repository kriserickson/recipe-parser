# predict.py
# Predict and extract structured content from a new raw HTML recipe page

from pathlib import Path
from joblib import load
import sys
import json
from typing import Tuple

from html_parser import parse_html
from feature_extraction import extract_features

MODEL_PATH = Path("../models/model.joblib")

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
        texts.append(feat['raw'])
        f = feat.copy()
        del f['raw']
        features_wo_text.append(f)
    return features_wo_text, texts


def extract_structured_data(html_path):
    html = Path(html_path).read_text(encoding="utf-8")
    elements = parse_html(html)
    features = extract_features(elements)

    features_wo_text, texts = split_features_and_text(features)

    model = load(MODEL_PATH)
    predictions = model.predict(texts)

    structured = {"title": None, "ingredients": [], "directions": []}
    for el, label in zip(elements, predictions):
        if label == "title" and structured["title"] is None:
            structured["title"] = el["text"]
        elif label == "ingredient":
            structured["ingredients"].append(el["text"])
        elif label == "direction":
            structured["directions"].append(el["text"])

    return structured


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/recipe.html")
        sys.exit(1)

    result = extract_structured_data(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))

