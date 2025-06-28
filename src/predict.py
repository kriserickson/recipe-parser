# predict.py
# Predict and extract structured content from a new raw HTML recipe page

import json
import argparse
import re
from pathlib import Path
from time import time
from joblib import load

from config import MODEL_PATH
# Note, Densify is imported here but not used in the script but is used in model so it must be here.
from feature_extraction import extract_features, preprocess_data, get_section_header, Densify
from html_parser import parse_html


SECTION_HEADING_PATTERN = re.compile(
    r"^for the ",
    re.IGNORECASE,
)

NUTRITION_PATTERN = re.compile(
    r"^(calories|fat|saturated fat|carbohydrates|sugar|fiber|protein|sodium|cholesterol)[:\\-]\s*\d+",
    re.IGNORECASE,
)

def is_fake_ingredient(text: str) -> bool:
    """
    Determines if a given text string is a section heading or nutrition information
    that should not be treated as a real ingredient.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text is a fake ingredient, False otherwise.
    """
    text = text.strip()
    if SECTION_HEADING_PATTERN.match(text):
        return True
    if NUTRITION_PATTERN.match(text):
        return True
    return False

def extract_structured_data(html: str, model) -> dict:
    """
def extract_structured_data(html_path: str, model) -> dict:

    Args:
        html (str): Contents of the HTML file
        model (Any): The trained model used for prediction.

    Returns:
        dict: A dictionary with keys 'title', 'ingredients', and 'directions'.
    """

    elements = parse_html(html)
    all_features = []
    current_section_heading = None  # Track current section heading
    for idx, el in enumerate(elements):

        current_section_heading = get_section_header(current_section_heading, el)

        features = extract_features(el, idx, elements, current_section_heading)
        all_features.append(features)

    data = preprocess_data(all_features)

    predictions = model.predict(data)

    structured = {"title": None, "ingredients": [], "directions": []}
    for el, label in zip(elements, predictions):
        text = el["text"]
        if label == "title" and structured["title"] is None:
            structured["title"] = text
        elif label == "ingredient":
            if is_fake_ingredient(text):
                continue
            structured["ingredients"].append(text)
        elif label == "direction":
            structured["directions"].append(text)

    return structured


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Predict structured recipe content from HTML.")
    parser.add_argument("html_path", type=str, help="Path to the recipe HTML file")
    parser.add_argument("--memory", action="store_true", help="Track memory usage during prediction")
    args = parser.parse_args()

    start = time()
    if args.memory:
        import psutil
        import tracemalloc
        import os

        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB

        tracemalloc.start()
        start_memory = get_memory_usage()

    recipeModel = load(MODEL_PATH)
    html = Path(args.html_path).read_text(encoding="utf-8")

    result = extract_structured_data(html, recipeModel)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.memory:
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        end_memory = get_memory_usage()
        print(f"Start memory: {start_memory:.2f} MB")
        print(f"Total memory usage from psutil: {end_memory:.2f} MB")
        print(f"Memory increase: {end_memory - start_memory:.2f} MB")
        print(f"Peak memory from tracemalloc: {peak / 1024 / 1024:.2f} MB")
        print(f"️Total time: {time() - start:.2f}s")
