# predict.py
# Predict and extract structured content from a new raw HTML recipe page

import json
import argparse
from pathlib import Path
from time import time

from joblib import load

from config import MODEL_PATH
from feature_extraction import extract_features, preprocess_data, get_section_header
from html_parser import parse_html

def extract_structured_data(html_path: str) -> dict:


    html = Path(html_path).read_text(encoding="utf-8")
    elements = parse_html(html)
    all_features = []
    current_section_heading = None  # Track current section heading
    for idx, el in enumerate(elements):

        current_section_heading = get_section_header(current_section_heading, el)

        features = extract_features(el, idx, elements, current_section_heading)
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

    result = extract_structured_data(args.html_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.memory:
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        end_memory = get_memory_usage()
        print(f"Peak memory from tracemalloc: {peak / 1024 / 1024:.2f} MB")
        print(f"Total memory usage from psutil: {end_memory:.2f} MB")
        print(f"Memory increase: {end_memory - start_memory:.2f} MB")

