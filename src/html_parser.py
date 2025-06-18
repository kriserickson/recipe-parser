from typing import List
from bs4 import BeautifulSoup, Comment, Tag

EXCLUDED_TAGS: List[str] = [
    "script", "style", "noscript", "footer",
    "nav", "link", "meta", "button"
]

def parse_html(html: str) -> list[dict[str, str]]:
    """
    Parses the HTML and returns a list of elements with metadata.
    Each element is a dict with keys like: 'text', 'tag', 'depth', etc.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Remove unwanted tags
    for tag in soup(EXCLUDED_TAGS):
        tag.decompose()

    elements = []

    def recurse(element, depth=0):

        if isinstance(element, Comment):
            return

        # If the element is a <li> tag, combine its text and skip its children
        if isinstance(element, Tag) and element.name == "li":
            combined = element.get_text(separator=" ", strip=True)
            if combined:
                elements.append({
                    'text': combined,
                    'tag': element.name,
                    'depth': depth,
                    'itemprop': element.get('itemprop', ''),
                    'class': element.get('class', []),
                    'id': element.get('id', '')
                })
            return  # Do not emit children of this <li>

        if element.name is not None:
            for child in element.children:
                recurse(child, depth + 1)
        elif element.string and element.string.strip():
            parent_tag = element.parent.name if element.parent else 'unknown'
            elements.append({
                'text': element.string.strip(),
                'tag': parent_tag,
                'depth': depth,
                'itemprop': element.parent.get('itemprop', ''),
                'class': element.parent.get('class', []),
                'id': element.parent.get('id', '')
            })

    recurse(soup.body if soup.body else soup)
    return elements