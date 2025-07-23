from bs4 import BeautifulSoup
import os


def get_html_path(filename: str) -> str:
    """
    Returns the full path to the HTML file. If no directory is specified,
    defaults to /data/html_pages.
    """
    if os.path.dirname(filename):
        return filename
    return os.path.join(os.path.dirname(__file__), "..", "data", "html_pages", filename)

def clean_html(html: str) -> str:
    """
    Removes non-visible elements from HTML: <head>, <script>, <style>, <link>, <img>, <svg> tags and their contents.
    Also removes all empty tags.
    Returns cleaned, visible HTML as a string.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove <head>
    if soup.head:
        soup.head.decompose()
    # Remove specific tags and their contents
    for tag in soup(["script", "style", "link", "img", "svg"]):
        tag.decompose()
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.Comment))):
        comment.decompose()

    for tag in soup.find_all():
    # Remove all empty tags (tags with no content and no attributes)
        if not tag.contents or not ''.join(str(c).strip() for c in tag.contents).strip():
        # Remove if tag has no contents (ignoring whitespace) and no attributes
            tag.decompose()
    return str(soup)