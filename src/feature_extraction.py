"""
Feature extraction module for recipe parsing.

This module provides functions to extract features from HTML elements
and build feature processing pipelines for machine learning models.
"""

from typing import List, Dict, Any, Union

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import re
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
    label_data : dict
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

def load_labeled_blocks(labels_dir: Path, html_dir: Path, limit: int | None = None) -> tuple[list, list]:
    """
    Load and parse labeled HTML blocks from files.

    Args:
        labels_dir: Path to directory containing JSON label files
        html_dir: Path to directory containing HTML files
        limit: Optional maximum number of files to process

    Returns:
        tuple: (features_list, labels_list) containing the training data
    """
    features_list, labels_list = [], []
    json_files = sorted(labels_dir.glob("recipe_*.json"))
    total = len(json_files)

    if limit:
        total = min(total, limit)

    report_size = max(10, int(total / 100))  # Report every 1% of total files

    for i, json_file in enumerate(json_files):
        if limit and i >= limit:
            break

        base = json_file.stem
        html_file = html_dir / f"{base}.html"
        if not html_file.exists():
            continue

        label_data = json.loads(json_file.read_text(encoding="utf-8"))
        html = html_file.read_text(encoding="utf-8")
        elements = parse_html(html)

        for idx, el in  enumerate(elements):
            elem_text = el.get("text", "").strip()

            label = label_element(elem_text, label_data)

            features = extract_features(el, elem_text, elements, idx)

            features_list.append(features)
            labels_list.append(label)

        if (i + 1) % report_size == 0 or (i + 1) == total:
            percent = ((i + 1) / total) * 100
            print(f"📦 Processed {i + 1}/{total} files ({percent:.1f}%)")

    return features_list, labels_list


def extract_features(el, elem_text, elements, idx):
    features = {}
    # Tag features
    features["tag"] = el.get("tag", "None")
    features["depth"] = el.get("depth", 0)
    # Parent tag (if available)
    features["parent_tag"] = el.get("parent_tag", "None")
    # Position in document
    features["block_index"] = idx
    features["position_ratio"] = idx / max(1, len(elements) - 1)
    # Text-based features
    features["text"] = elem_text  # Keep for vectorizer
    features["text_length"] = len(elem_text)
    features["num_digits"] = sum(ch.isdigit() for ch in elem_text)
    features["starts_with_number"] = 1 if re.match(r"^\\d", elem_text) else 0
    elem_text_lower = elem_text.lower()
    features["contains_unit"] = int(
        any(re.search(r'\\b' + re.escape(unit) + r'\\b', elem_text_lower) for unit in units))
    features["comma_count"] = elem_text.count(",")
    features["dot_count"] = elem_text.count(".")
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
    return features


def build_feature_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline for feature processing.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline with TF-IDF vectorizer

    """
    return Pipeline([
        ('dict_vect', DictVectorizer(sparse=True)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga'))
    ])

def is_valid_feature(feature: Dict[str, Any]) -> bool:
    """
    Check if a feature dict is valid for training.

    Args:
        feature: Feature dictionary.

    Returns:
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

    Args:
        features: List of feature dicts.
        labels: List of labels.

    Returns:
         Scikit-learn pipeline with DictVectorizer and LogisticRegression.
    """
    valid_indices = [i for i, feat in enumerate(features) if is_valid_feature(feat)]
    filtered_features = [features[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    return filtered_features, filtered_labels