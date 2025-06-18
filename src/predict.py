# predict.py
# Predict and extract structured content from a new raw HTML recipe page

import json
import sys
from pathlib import Path
from joblib import load

from feature_extraction import extract_features, preprocess_data
from html_parser import parse_html

MODEL_PATH = Path("../models/model.joblib")

def extract_structured_data(html_path: string):
    html = Path(html_path).read_text(encoding="utf-8")
    elements = parse_html(html)
    all_features = []
    for el in elements:
        features = extract_features(el)
        all_features.append(features)

    data = preprocess_data(all_features)

    model = load(MODEL_PATH)
    predictions = model.predict(data)

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

