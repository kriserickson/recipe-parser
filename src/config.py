from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
HTML_DIR = DATA_DIR / "html_pages"
LABEL_DIR = DATA_DIR / "labels"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
CACHE_DIR = DATA_DIR / "cache"