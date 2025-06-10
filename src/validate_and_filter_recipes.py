import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
import tldextract
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
POTENTIAL_LABELS_DIR = DATA_DIR / "potential_labels"
LABELS_DIR = DATA_DIR / "labels"
HTML_DIR = DATA_DIR / "html_pages"
STATE_FILE = DATA_DIR / "processing_state.json"

LABELS_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)

# === Settings ===
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/113.0 Safari/537.36"
    )
}
THROTTLE_DELAY = 60  # seconds per domain
PROCESS_FAILED = "--process-failed-files" in sys.argv

# === State Tracking ===
if STATE_FILE.exists():
    processing_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    SELENIUM_WHITELIST = set(processing_state.get("selenium_whitelist", []))
else:
    processing_state = {
        "processed_files": [],
        "failed_files": {},
        "output_index": 0,
        "selenium_whitelist": ["allrecipes.com"]
    }
    SELENIUM_WHITELIST = set(processing_state["selenium_whitelist"])

last_domain_access = {}


def save_state() -> None:
    """
    Save the current processing state and Selenium whitelist to the state file.
    """
    processing_state["selenium_whitelist"] = sorted(SELENIUM_WHITELIST)
    STATE_FILE.write_text(json.dumps(processing_state, indent=2), encoding="utf-8")


def throttle_domain(domain: str) -> bool:
    """
    Throttle requests to a domain to avoid hitting rate limits.

    Parameters
    ----------
    domain : str
        The domain to check for throttling.

    Returns
    -------
    bool
        True if the domain is ready for a new request, False if it should be throttled.
    """
    now = time.time()
    last_time = last_domain_access.get(domain, 0)
    wait = THROTTLE_DELAY - (now - last_time)
    if wait > 0:
        return False  # not ready yet
    last_domain_access[domain] = now
    return True


def is_valid_page(html: str, data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that the HTML page contains the expected recipe content.

    Parameters
    ----------
    html : str
        The HTML content of the page.
    data : Dict[str, Any]
        The expected recipe data (title, ingredients, directions).

    Returns
    -------
    Tuple[bool, str]
        (True, '') if valid, otherwise (False, reason for failure).
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ").lower()

    title_ok = data.get("title") and data["title"].lower() in text

    def cleaned_strings(items):
        return [s.strip().lower() for s in items if s.strip() and not s.strip().isdigit()]

    ingredients = cleaned_strings(data.get("ingredients", []))
    directions = cleaned_strings(data.get("directions", []))

    ingredients_ok = any(item in text for item in ingredients)
    directions_ok = any(step in text for step in directions)

    reasons = []
    if not title_ok:
        reasons.append("title missing")
    if not ingredients_ok:
        reasons.append("no ingredients matched")
    if not directions_ok:
        reasons.append("no directions matched")

    return title_ok or (ingredients_ok or directions_ok), ", ".join(reasons)

def fetch_with_requests(href: str) -> str:
    """
    Fetch HTML content from a URL using the requests library.

    Parameters
    ----------
    href : str
        The URL to fetch.

    Returns
    -------
    str
        The HTML content as a string, or an error message if the request fails.
    """
    try:
        resp = requests.get(href, headers=HEADERS, timeout=15)
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
            return resp.text
        return f"HTTP {resp.status_code}"
    except requests.RequestException as e:
        return f"RequestException: {e}"


def fetch_with_selenium(href: str) -> str:
    """
    Fetch HTML content from a URL using Selenium WebDriver.

    Parameters
    ----------
    href : str
        The URL to fetch.

    Returns
    -------
    str
        The HTML content as a string, or an error message if Selenium fails.
    """
    try:
        options = Options()
        options.headless = True
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        driver.get(href)
        html = driver.page_source
        driver.quit()
        return html
    except WebDriverException as e:
        print(f"Selenium failed for {href}: {e}")
        return f"SeleniumException: {e}"


def process_file(json_path: Path) -> bool:
    """
    Process a single potential label JSON file: fetch the HTML, validate, and save if valid.

    Parameters
    ----------
    json_path : Path
        Path to the JSON file containing potential recipe labels and metadata.

    Returns
    -------
    bool
        True if the file was processed (success or permanent failure), False if it should be retried later.
    """
    if json_path.name in processing_state["processed_files"]:
        print(f"⏩ Already processed: {json_path.name}")
        return True
    if not PROCESS_FAILED and json_path.name in processing_state["failed_files"]:
        print(f"⏩ Skipping known failed: {json_path.name}")
        return True

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        processing_state["failed_files"][json_path.name] = f"Invalid JSON: {e}"
        return True

    href = data.get("href", "").strip()
    if not href:
        processing_state["failed_files"][json_path.name] = "Missing href"
        return True

    domain = tldextract.extract(href).top_domain_under_public_suffix
    if not throttle_domain(domain):
        return False  # not ready, requeue this item

    print(f"🔍 Checking {href}")

    if domain in SELENIUM_WHITELIST:
        html = fetch_with_selenium(href)
    else:
        html = fetch_with_requests(href)
        if isinstance(html, str) and ("not acceptable" in html.lower() or "error 405" in html.lower()):
            print(f"⚠️  Falling back to Selenium for: {href}")
            html = fetch_with_selenium(href)

    if not isinstance(html, str) or html.strip() == "":
        processing_state["failed_files"][json_path.name] = {
            "url": href,
            "reason": "Empty or invalid HTML from both methods"
        }
        SELENIUM_WHITELIST.add(domain)
        print(f"🔁 Added {domain} to Selenium whitelist")
        return True

    out_index = processing_state["output_index"]
    base_name = f"recipe_{out_index:05d}"

    valid, fail_reason = is_valid_page(html, data)
    if not valid:
        print(f"❌ Failed: {base_name}")
        processing_state["failed_files"][json_path.name] = {
            "url": href,
            "reason": fail_reason or "Page content did not match expectations"
        }
        return True

    LABELS_DIR.joinpath(f"{base_name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    HTML_DIR.joinpath(f"{base_name}.html").write_text(html, encoding="utf-8")

    processing_state["processed_files"].append(json_path.name)
    processing_state["output_index"] += 1
    print(f"✅ Saved: {base_name}")
    return True


def main() -> None:
    """
    Main entry point for validating and filtering recipe pages.
    Iterates through potential label files, processes each, and saves state after each attempt.
    """
    json_files = deque(sorted(POTENTIAL_LABELS_DIR.glob("recipe_*.json")))
    while json_files:
        json_file = json_files.popleft()
        result = process_file(json_file)
        if not result:
            json_files.append(json_file)  # requeue if throttled
        save_state()

if __name__ == "__main__":
    main()
