"""
Configuration settings for the Recipe Parser application.

This module contains path configurations and other settings used throughout the application.
"""
from pathlib import Path
from typing import Final

# Base paths
DATA_DIR: Final[Path] = Path("..") / "data"
HTML_DIR: Final[Path] = DATA_DIR / "html_pages"
LABEL_DIR: Final[Path] = DATA_DIR / "labels"
MODEL_DIR: Final[Path] = Path("..") / "models"
MODEL_PATH: Final[Path] = MODEL_DIR / "model.joblib"

# Validate critical paths exist
def validate_paths() -> None:
    """
    Validate that all required directories exist.
    
    Raises
    ------
    FileNotFoundError
        If any required directory is missing
    """
    for path in [DATA_DIR, HTML_DIR, LABEL_DIR, MODEL_PATH.parent]:
        if not path.exists():
            raise FileNotFoundError(f"Required directory not found: {path}")

# Validate paths when module is imported
validate_paths()
