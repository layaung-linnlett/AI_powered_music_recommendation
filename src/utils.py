"""Shared constants, helpers, and logging setup for the music mood classifier."""

# ==== Standard Library Imports ====
import logging
import os
from pathlib import Path
from typing import List

# ==== Project Root ====
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ==== Directory Paths ====
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
SRC_DIR: Path = PROJECT_ROOT / "src"

# ==== Data Constants ====
RANDOM_SEED: int = 42
TARGET_COLUMN: str = "track_genre"
INDEX_COLUMN: str = "Unnamed: 0"

# Columns that are identifiers or free text and must be dropped before modelling.
DROP_COLUMNS: List[str] = ["track_id", "artists", "album_name", "track_name"]

# The boolean column that needs to be cast to int before scaling.
BOOL_COLUMN: str = "explicit"

# Numeric feature columns used for modelling.
NUMERIC_FEATURES: List[str] = [
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

# ==== Split Ratios ====
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ==== Model Persistence ====
PREPROCESSOR_FILENAME: str = "preprocessor.pkl"
FINAL_MODEL_FILENAME: str = "final_model.pkl"

# ==== Cross-Validation ====
CV_FOLDS: int = 5

# ==== Minimum Target Accuracy ====
MIN_TARGET_ACCURACY: float = 0.80

# ==== Logging Format ====
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ==== Helpers ====

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default INFO).

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dirs() -> None:
    """Create all required output directories if they do not already exist."""
    for directory in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# ==== Entry Point ====

if __name__ == "__main__":
    ensure_dirs()
    log = get_logger(__name__)
    log.info("Project root: %s", PROJECT_ROOT)
    log.info("All output directories are ready.")
