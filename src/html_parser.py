from bs4 import BeautifulSoup, Comment


def parse_html(html):
    """
    Parses the HTML and returns a list of elements with metadata.
    Each element is a dict with keys like: 'text', 'tag', 'depth', etc.
    """
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(["script", "style", "noscript", "footer", "nav", 'link', 'meta', 'button']):
        tag.decompose()

    elements = []

    def recurse(element, depth=0):

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