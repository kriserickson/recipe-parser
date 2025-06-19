import re
from typing import Any, Dict, List, Tuple, Union

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Measurement units for English recipes
units = [
    "teaspoon", "teaspoons", "tsp", "tablespoon", "tablespoons", "tbsp",
    "cup", "cups", "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs",
    "gram", "grams", "g", "kilogram", "kilograms", "kg", "liter", "liters", "l",
    "ml", "milliliter", "milliliters", "pinch", "clove", "cloves", "slice", "slices"
]

ing_keywords = ["ingredient", "component", "element", "material"]
dir_keywords = ["instruct", "direction", "step", "method", "preparation", "procedure", "technique"]
title_keywords = ["title", "name", "headline"]


class ItemSelector(BaseEstimator, TransformerMixin):
    """
    For FeatureUnion to extract a specific item from a tuple or dict in the pipeline.
    """
    def __init__(self, key: str):
        """
        Initialize the ItemSelector.

        Args:
            key (str): The key to select ('structured' or 'text').
        """
        self.key = key

    def fit(self, X: List[Any], y: Any = None) -> 'ItemSelector':
        """
        Fit method (does nothing, present for compatibility).

        Args:
            X (List[Any]): Input data.
            y (Any, optional): Target values (ignored).

        Returns:
            ItemSelector: self
        """
        return self

    def transform(self, X: List[Tuple[Dict[str, Any], str]]) -> List[Union[Dict[str, Any], str]]:
        """
        Transform the input data by selecting the specified key.

        Args:
            X (List[Tuple[Dict[str, Any], str]]): Input data as list of tuples.

        Returns:
            List[Union[Dict[str, Any], str]]: Selected data based on key.
        """
        if self.key == 'structured':
            return [x[0] for x in X]  # structured features (dict)
        elif self.key == 'text':
            return [x[1] for x in X]  # text field
        else:
            raise ValueError(f"Unknown key: {self.key}")


def get_section_header(current_section_heading, el):
    elem_text = el["text"]
    tag = el.get("tag", "").lower()
    elem_text_lower = elem_text.lower()
    # If this element is a heading → update current section
    if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        # Check if it contains keywords for ingredient or direction sections
        if any(k in elem_text_lower for k in ing_keywords):
            current_section_heading = "ingredient"
        elif any(k in elem_text_lower for k in dir_keywords):
            current_section_heading = "direction"
        else:
            current_section_heading = None  # unknown heading
    return current_section_heading

def split_features_and_text(features: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Splits features into two lists: one with dicts (without 'raw'), and one with the 'raw' text.

    Args:
        features (List[Dict[str, Any]]): List of feature dictionaries.

    Returns:
        Tuple[List[Dict[str, Any]], List[str]]: Tuple of (features without 'raw', list of raw texts).
    """
    features_wo_text = []
    texts = []
    for feat in features:
        texts.append(feat['raw'])
        f = feat.copy()
        del f['raw']
        features_wo_text.append(f)
    return features_wo_text, texts

def preprocess_data(features: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
    """
    Prepares data for feature extraction by splitting and zipping features and texts.

    Args:
        features (List[Dict[str, Any]]): List of feature dictionaries.

    Returns:
        List[Tuple[Dict[str, Any], str]]: List of tuples (features, text).
    """
    features_wo_text, texts = split_features_and_text(features)
    # FeatureUnion requires parallel lists, passed as tuples
    combined = list(zip(features_wo_text, texts))
    return combined


def extract_features(el: Dict[str, Any], idx: int, elements: list[Dict[str, Any]], current_section_heading: str | None = None) -> Dict[str, Any]:
    """
    Extracts structured features from a list of elements.

    Args:
        el (Dict[str, Any]): Element dictionary containing 'tag', 'depth', and 'text'.
        idx (int): Index of the element in the list.
        elements (List[Dict[str, Any]]): List of elements, each with 'tag', 'depth', and 'text'.
        current_section_heading (str | None): Current section heading, if applicable.

    Returns:
        List[Dict[str, Any]]: List of feature dictionaries with extracted features.

    """

    # Class/id keywords
    class_id_str = " ".join(str(x) for x in el.get("class", [])) + " " + str(el.get("id", ""))
    class_id_str = class_id_str.lower()

    # Microdata
    itemprop = el.get("itemprop", "")
    itemprop = itemprop.lower() if itemprop else ""

    features = {
        "tag": el["tag"],
        "depth": el["depth"],
        "text_len": len(elem_text := el.get("text", "")),
        "starts_with_digit": elem_text[0].isdigit(),
        "parent_tag": el.get("parent_tag", "None"),
        "raw": elem_text,
        "num_digits": sum(ch.isdigit() for ch in elem_text),
        "contains_unit": int(any(re.search(r"\b" + re.escape(unit) + r"\b", elem_text.lower()) for unit in units)),
        "comma_count": elem_text.count(","),
        "dot_count": elem_text.count("."),
        "contains_quantity_number": int(bool(re.search(r"\d+|\d+/\d+|½|¼|¾|⅓|⅔\bone\b|\btwo\b|\bthree\b|\bfour\b|\bfive\b", elem_text))),
        "element_index": idx,
        "position_ratio": idx / max(1, len(elements) - 1),
        "is_under_current_ingredient_section": int(current_section_heading == "ingredient"),
        "is_under_current_direction_section": int(current_section_heading == "direction"),
        "class_has_ing_keyword": int(any(k in class_id_str for k in ing_keywords)),
        "class_has_dir_keyword": int(any(k in class_id_str for k in dir_keywords)),
        "class_has_title_keyword": int(any(k in class_id_str for k in title_keywords)),
        "itemprop_name": int(itemprop == "name"),
        "itemprop_ingredient": int(itemprop == "recipeingredient"),
        "itemprop_instructions": int(itemprop == "recipeinstructions"),
    }


    return features

def build_transformer() -> FeatureUnion:
    """
    Builds a FeatureUnion transformer for structured and text features.

    Returns:
        FeatureUnion: A scikit-learn FeatureUnion object combining structured and text pipelines.
    """
    dict_vect = DictVectorizer(sparse=True)
    transformer = FeatureUnion([
        ("structured", make_pipeline(ItemSelector('structured'), dict_vect)),
        ("text", make_pipeline(ItemSelector('text'), TfidfVectorizer(max_features=500, ngram_range=(1, 2))))
    ])

    return transformer

