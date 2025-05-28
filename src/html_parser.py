"""
HTML parsing module for recipe extraction.

This module provides functionality to parse HTML content and extract
relevant text elements while maintaining structural information.
"""

from typing import List, Dict, Any, Union
from bs4 import BeautifulSoup, Comment, Tag, NavigableString

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

    # Remove unwanted elements
    for tag in soup(EXCLUDED_TAGS):
        tag.decompose()

    elements: List[Dict[str, Any]] = []

    def recurse(element: Union[Tag, NavigableString], depth: int = 0) -> None:
        """
        Recursively process HTML elements and extract text content.

        Parameters
        ----------
        element : Union[Tag, NavigableString]
            BeautifulSoup element to process
        depth : int, optional
            Current nesting depth, by default 0
        """
        if isinstance(element, Comment):
            return

        if element.name is not None:
            for child in element.children:
                recurse(child, depth + 1)
        elif element.string and element.string.strip():
            parent_tag = element.parent.name if element.parent else 'unknown'
            elements.append({
                'text': element.string.strip(),
                'tag': parent_tag,
                'depth': depth
            })

    recurse(soup.body if soup.body else soup)
    return elements