"""
Feature extraction module for recipe parsing.

This module provides functions to extract features from HTML elements
and build feature processing pipelines for machine learning models.
"""

import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Tuple
from typing import List

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from html_parser import parse_html

# Constants
MAX_FEATURES: int = 1000
NGRAM_RANGE: tuple[int, int] = (1, 2)

# Keywords for common roles
ing_keywords = ["ingr", "ingredient"]
dir_keywords = ["instr", "direction", "step", "method"]
title_keywords = ["title", "name", "headline"]
img_keywords = ["img", "image", "photo", "picture"]

# Measurement units for English recipes
units = [
    "teaspoon", "teaspoons", "tsp", "tablespoon", "tablespoons", "tbsp",
    "cup", "cups", "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs",
    "gram", "grams", "g", "kilogram", "kilograms", "kg", "liter", "liters", "l",
    "ml", "milliliter", "milliliters", "pinch", "clove", "cloves", "slice", "slices"
]

def label_element(text: str, label_data: Dict[str, Any]) -> str:
    """
    Determine the label for a given block of text using JSON label data.

    Parameters
    ----------
    text : str
        The text block to be labeled.
    label_data : Dict[str, Any]
        Dictionary containing recipe data with keys 'ingredients', 'directions', 'title'.

    Returns
    -------
    str
        One of: 'ingredient', 'direction', 'title', or 'none'.
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

def process_file_pair(json_file: Path, html_dir: Path) -> list[tuple[dict, str]]:
    """
    Process a single JSON/HTML file pair and extract features and labels for each HTML element.

    Parameters
    ----------
    json_file : Path
        Path to the JSON file containing recipe labels.
    html_dir : Path
        Directory containing the corresponding HTML file.

    Returns
    -------
    list[tuple[dict, str]]
        A list of (features, label) tuples for each element in the HTML file.
    """
    features_labels = []
    base = json_file.stem
    html_file = html_dir / f"{base}.html"
    if not html_file.exists():
        return features_labels  # empty

    try:
        label_data = json.loads(json_file.read_text(encoding="utf-8"))
        html = html_file.read_text(encoding="utf-8")
        elements = parse_html(html)

        for idx, el in enumerate(elements):
            tag = el.get("tag", "").lower()
            elem_text = el.get("text", "").strip()

            section_heading_text = None
            # If this element is a heading → update current section
            if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                if any(k in elem_text.lower() for k in ing_keywords):
                    section_heading_text = "ingredient"
                elif any(k in elem_text.lower() for k in dir_keywords):
                    section_heading_text = "direction"
                elif any(k in elem_text.lower() for k in title_keywords):
                    section_heading_text = "title"
                else:
                    section_heading_text = None  # unknown heading

            label = label_element(elem_text, label_data)
            features = extract_features(el, elem_text, elements, idx, current_section_heading = section_heading_text)
            features_labels.append((features, label))

    except Exception as e:
        print(f"⚠️ Error processing {json_file.name}: {e}")

    return features_labels

def load_labeled_blocks(labels_dir: Path, html_dir: Path, limit: int | None = None) -> tuple[list, list]:
    """
    Load and parse labeled HTML blocks from files in parallel.

    This function reads JSON label files and their corresponding HTML files, extracts features and labels for each HTML element,
    and returns two lists: one of feature dicts and one of labels. Processing is parallelized for efficiency.

    Parameters
    ----------
    labels_dir : Path
        Path to the directory containing JSON label files (one per recipe).
    html_dir : Path
        Path to the directory containing HTML files (one per recipe).
    limit : int | None, optional
        Optional maximum number of files to process. If None, all files are processed.

    Returns
    -------
    tuple[list, list]
        A tuple (features_list, labels_list) where:
            - features_list: list of feature dicts for all elements in all recipes
            - labels_list: list of corresponding labels for each element
    """
    features_list, labels_list = [], []
    json_files = sorted(labels_dir.glob("recipe_*.json"))
    total = len(json_files)

    if limit:
        json_files = json_files[:limit]
    total = len(json_files)  # recalculate after applying limit

    report_size = max(10, int(total / 100))

    # Use ProcessPoolExecutor for parallelism
    with ProcessPoolExecutor() as executor:
        # Submit all jobs
        futures = {executor.submit(process_file_pair, json_file, html_dir): json_file for json_file in json_files}

        # Collect results as they complete
        for idx, future in enumerate(as_completed(futures)):
            results = future.result()
            for features, label in results:
                features_list.append(features)
                labels_list.append(label)

            if (idx + 1) % report_size == 0 or (idx + 1) == total:
                percent = ((idx + 1) / total) * 100
                print(f"📦 Processed {idx + 1}/{total} files ({percent:.1f}%)")

    return features_list, labels_list

def find_nearest_section_heading(elements: list[dict], idx: int) -> str | None:
    """
    Find the nearest section heading above the given element index.

    Parameters
    ----------
    elements : list[dict]
        List of parsed HTML elements, each as a dictionary.
    idx : int
        Index of the current element in the elements list.

    Returns
    -------
    str or None
        Returns 'ingredient', 'direction', 'title', or None based on the nearest heading tag above the element.
    """
    for i in range(idx - 1, -1, -1):
        tag = elements[i].get("tag", "").lower()
        text = elements[i].get("text", "").strip().lower()
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            if any(k in text for k in ing_keywords):
                return "ingredient"
            if any(k in text for k in dir_keywords):
                return "direction"
            if any(k in text for k in title_keywords):
                return "title"
            return None
    return None


def compute_distance_to_nearest_heading(
    elements: list[dict], idx: int, heading_tags: tuple = ("h1", "h2", "h3", "h4", "h5", "h6")
) -> int:
    """
    Compute the distance from the current element to the nearest heading element.

    Parameters
    ----------
    elements : list[dict]
        List of parsed HTML elements.
    idx : int
        Index of the current element.
    heading_tags : tuple, optional
        Tuple of heading tag names to consider.

    Returns
    -------
    int
        Distance to the nearest heading element, or 9999 if none found.
    """
    # Search backward
    backward_dist = None
    for i in range(idx - 1, -1, -1):
        tag = elements[i].get("tag", "").lower()
        if tag in heading_tags:
            backward_dist = idx - i
            break

    # Search forward
    forward_dist = None
    for i in range(idx + 1, len(elements)):
        tag = elements[i].get("tag", "").lower()
        if tag in heading_tags:
            forward_dist = i - idx
            break

    distances = [d for d in [backward_dist, forward_dist] if d is not None]
    if not distances:
        return 9999  # no heading found nearby
    return min(distances)


def extract_features(
    el: dict, elem_text: str, elements: list[dict], idx: int, current_section_heading: str=None
) -> dict[str, Any]:
    """
    Extract features from a single HTML element for ML models.

    Parameters
    ----------
    el : dict
        The HTML element dictionary.
    elem_text : str
        The text content of the element.
    elements : list[dict]
        List of all parsed HTML elements.
    idx : int
        Index of the current element in the list.

    Returns
    -------
    dict[str, Any]
        Dictionary of extracted features for the element.
    """
    elem_text_lower = elem_text.lower()
    features: dict[str, Any] = {
        "tag": el.get("tag", "None"),
        "depth": el.get("depth", 0),
        "parent_tag": el.get("parent_tag", "None"),
        "block_index": idx,
        "position_ratio": idx / max(1, len(elements) - 1),
        "text": elem_text,
        "text_length": len(elem_text),
        "num_digits": sum(ch.isdigit() for ch in elem_text),
        "starts_with_number": int(bool(re.match(r"^\d", elem_text))),
        "contains_quantity_number": int(bool(re.search(r"\d+|\d+/\d+|½|¼|¾|⅓|⅔|\bone\b|\btwo\b|\bthree\b|\bfour\b|\bfive\b", elem_text))),
        "contains_unit": int(any(re.search(r"\b" + re.escape(unit) + r"\b", elem_text_lower) for unit in units)),
        "comma_count": elem_text.count(","),
        "dot_count": elem_text.count("."),
        "is_heading": int(el.get("tag", "").lower() in ["h1", "h2", "h3", "h4", "h5", "h6"]),
        "is_list_item": int(el.get("tag", "").lower() == "li"),
        "distance_to_nearest_heading": compute_distance_to_nearest_heading(elements, idx),
        "is_under_current_ingredient_section": int(current_section_heading == "ingredient"),
        "is_under_current_direction_section": int(current_section_heading == "direction"),
        "is_under_current_title_section": int(current_section_heading == "title")
    }

    # Class/id keywords
    class_id_str = " ".join(str(x) for x in el.get("class", [])) + " " + str(el.get("id", ""))
    class_id_str = class_id_str.lower()
    features["has_ing_keyword"] = int(any(k in class_id_str for k in ing_keywords))
    features["has_dir_keyword"] = int(any(k in class_id_str for k in dir_keywords))
    features["has_title_keyword"] = int(any(k in class_id_str for k in title_keywords))
    features["has_img_keyword"] = int(any(k in class_id_str for k in img_keywords))

    # Microdata
    itemprop = el.get("itemprop", "")
    itemprop = itemprop.lower() if itemprop else ""
    features["itemprop_name"] = int(itemprop == "name")
    features["itemprop_ingredient"] = int(itemprop == "recipeingredient")
    features["itemprop_instructions"] = int(itemprop == "recipeinstructions")
    features["itemprop_image"] = int(itemprop == "image")

    # Ancestor context
    ancestor_classes = el.get("ancestor_classes", [])
    ancestor_classes_flat = " ".join(ancestor_classes).lower()
    features["ancestor_has_ingredient_keyword"] = int(any(k in ancestor_classes_flat for k in ing_keywords))
    features["ancestor_has_direction_keyword"] = int(any(k in ancestor_classes_flat for k in dir_keywords))
    features["ancestor_has_title_keyword"] = int(any(k in ancestor_classes_flat for k in title_keywords))

    # Section context
    nearest_section = find_nearest_section_heading(elements, idx)
    features["is_under_ingredient_section"] = int(nearest_section == "ingredient")
    features["is_under_direction_section"] = int(nearest_section == "direction")

    # Title Hueristics
    features["is_possible_title_heading"] = int(el.get("tag", "").lower() in ["h1", "h2", "h3"])
    features["is_possible_title_short"] = int(len(elem_text.split()) <= 8)
    features["is_first_5_blocks"] = int(idx < 5)
    features["is_first_10_blocks"] = int(idx < 10)

    return features


def build_feature_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline for feature processing.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline with DictVectorizer and LogisticRegression.
    """
    return Pipeline([
        ('dict_vect', DictVectorizer(sparse=True)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga'))
    ])

def is_valid_feature(feature: Dict[str, Any]) -> bool:
    """
    Check if a feature dict is valid for training.

    Parameters
    ----------
    feature : Dict[str, Any]
        Feature dictionary.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    required_keys = [
        "tag", "depth", "parent_tag", "block_index", "position_ratio",
        "text", "text_length", "num_digits", "starts_with_number",
        "contains_unit", "comma_count", "dot_count",
        "has_ing_keyword", "has_dir_keyword", "has_title_keyword", "has_img_keyword",
        "itemprop_name", "itemprop_ingredient", "itemprop_instructions", "itemprop_image"
    ]
    # Ensure all required keys are present and not None
    result = all(key in feature and feature[key] is not None for key in required_keys)
    if not result:
        print(f"Invalid feature: {feature}")

    return result

def filter_valid_features(
    features: List[Dict[str, Any]], labels: List[str]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build a scikit-learn Pipeline for structured feature processing.
    Filter out invalid feature dicts and their corresponding labels.

    Parameters
    ----------
    features : List[Dict[str, Any]]
        List of feature dicts.
    labels : List[str]
        List of labels.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[str]]
        Filtered features and labels.
    """
    valid_indices = [i for i, feat in enumerate(features) if is_valid_feature(feat)]
    filtered_features = [features[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    return filtered_features, filtered_labels