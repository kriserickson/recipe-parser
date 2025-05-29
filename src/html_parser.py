"""
HTML parsing module for recipe extraction.

This module provides functionality to parse HTML content and extract
relevant text elements while maintaining structural information.
"""

from typing import List, Dict, Any, Union
from bs4 import BeautifulSoup, Comment, Tag, NavigableString
import re

# Constants for HTML parsing
EXCLUDED_TAGS: List[str] = [
    "script", "style", "noscript", "footer",
    "nav", "link", "meta", "button"
]

def parse_html(html: str) -> List[Dict[str, Any]]:
    """
    Parse HTML content and extract text elements with metadata.

    Parameters
    ----------
    html : str
        Raw HTML content to parse

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing element data with keys:
        - 'text': str, extracted text content
        - 'tag': str, HTML tag name of parent element
        - 'depth': int, nesting depth in document tree

    Raises
    ------
    ValueError
        If input HTML is empty or invalid
    """
    if not html or not html.strip():
        raise ValueError("HTML content cannot be empty")

    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(EXCLUDED_TAGS):
        tag.decompose()

    elements: List[Dict[str, Any]] = []

    def is_ingredient_li(tag: Tag) -> bool:
        # Simple check: is it a <li> in a list with a parent having 'ingredient' in class/id?
        parent = tag.parent
        if parent and parent.name in ['ul', 'ol']:
            parent_classes = " ".join(parent.get('class', [])).lower()
            parent_id = parent.get('id', '').lower()
            return 'ingredient' in parent_classes or 'ingredient' in parent_id
        return False

    def recurse(element: Union[Tag, NavigableString], depth: int = 0) -> None:
        if isinstance(element, Comment):
            return
        # SPECIAL CASE: Ingredient <li>
        if isinstance(element, Tag) and element.name == "li" and is_ingredient_li(element):
            combined = element.get_text(separator=" ", strip=True)
            if combined:
                elements.append({
                    'text': combined,
                    'tag': element.name,
                    'depth': depth,
                    'itemprop': element.get('itemprop', ''),
                    'class': element.get('class', []),
                    'id': element.get('id', ''),
                })
            return  # Do not emit children of this <li>
        if element.name is not None:
            for child in element.children:
                recurse(child, depth + 1)
        elif element.string and re.search(r'\S', element.string):
            cleaned_text = element.string.replace("\u200b", "").strip()
            if cleaned_text:
                parent_tag = element.parent.name if element.parent else 'unknown'
                elements.append({
                    'text': cleaned_text,
                    'tag': parent_tag,
                    'depth': depth,
                    'itemprop': element.parent.get('itemprop', '') if element.parent else '',
                    'class': element.parent.get('class', []) if element.parent else [],
                    'id': element.parent.get('id', '') if element.parent else '',
                })

    recurse(soup.body if soup.body else soup)
    return elements