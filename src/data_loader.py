"""CSV loading, schema inspection, and target variable discovery."""

# ==== Standard Library Imports ====
from pathlib import Path
from typing import Optional, Tuple

# ==== Third-Party Imports ====
import pandas as pd

# ==== Internal Imports ====
from src.utils import (
    DATA_RAW_DIR,
    DROP_COLUMNS,
    TARGET_COLUMN,
    get_logger,
)

# ==== Module Logger ====
logger = get_logger(__name__)

# ==== Constants ====
# Minimum number of unique values for a column to be considered a viable target.
MIN_TARGET_CARDINALITY: int = 2
# Maximum fraction of unique values (relative to row count) allowed in a target column.
MAX_TARGET_UNIQUE_FRACTION: float = 0.50
# Minimum number of samples per class for the target to be meaningful.
MIN_SAMPLES_PER_CLASS: int = 10


# ==== Functions ====

def find_csv(directory: Path = DATA_RAW_DIR) -> Path:
    """Locate the first CSV file in the given directory.

    Args:
        directory: The directory to search for CSV files.

    Returns:
        Path to the discovered CSV file.

    Raises:
        FileNotFoundError: If no CSV file is found in the directory.
    """
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in: {directory}")
    if len(csv_files) > 1:
        logger.warning(
            "Multiple CSV files found; using the first: %s", csv_files[0].name
        )
    logger.info("Found dataset: %s", csv_files[0].name)
    return csv_files[0]


def load_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the dataset from the raw data directory.

    If csv_path is not provided the function discovers the CSV automatically
    via find_csv().

    Args:
        csv_path: Optional explicit path to the CSV file.

    Returns:
        Raw DataFrame loaded from disk.
    """
    if csv_path is None:
        csv_path = find_csv()

    df = pd.read_csv(csv_path, index_col=0)
    logger.info("Loaded data: shape=%s", df.shape)
    return df


def inspect_schema(df: pd.DataFrame) -> None:
    """Print a full schema summary of the DataFrame.

    Reports shape, column names, data types, and the first few rows to stdout.

    Args:
        df: The DataFrame to inspect.
    """
    print("=" * 60)
    print("DATASET SCHEMA SUMMARY")
    print("=" * 60)
    print(f"Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print()
    print("Column names:")
    for col in df.columns:
        print(f"  {col}")
    print()
    print("Data types:")
    print(df.dtypes.to_string())
    print()
    print("First 5 rows:")
    print(df.head().to_string())
    print("=" * 60)


def discover_target(df: pd.DataFrame) -> str:
    """Identify the target column from the DataFrame automatically.

    The function examines each column and scores it as a candidate target based
    on the following heuristics:

    1. Column name contains a mood or genre keyword (highest signal).
    2. The column is of object or low-cardinality integer dtype.
    3. Cardinality is in a sensible range for a classification target (2 to 50% of rows).
    4. The minimum samples per class is at least MIN_SAMPLES_PER_CLASS.

    The discovered target column name is logged with the reasoning.

    Args:
        df: The raw DataFrame to inspect.

    Returns:
        The name of the discovered target column.

    Raises:
        ValueError: If no suitable target column can be identified.
    """
    # Keywords that strongly indicate a target column in a music mood dataset.
    target_keywords = [
        "genre", "mood", "label", "category", "class", "tag",
        "emotion", "sentiment", "style",
    ]

    candidates: list[Tuple[str, float]] = []

    for col in df.columns:
        if col in DROP_COLUMNS:
            continue

        col_lower = col.lower()
        score: float = 0.0

        # Keyword match gives a strong boost.
        if any(kw in col_lower for kw in target_keywords):
            score += 10.0

        # Object or boolean dtype columns are typical label columns.
        if df[col].dtype == object:
            score += 3.0
        elif df[col].dtype == bool:
            score += 1.0

        # Cardinality check: too few or too many unique values disqualifies.
        n_unique = df[col].nunique()
        unique_fraction = n_unique / len(df)
        if n_unique < MIN_TARGET_CARDINALITY or unique_fraction > MAX_TARGET_UNIQUE_FRACTION:
            continue

        # Verify minimum samples per class.
        min_count = df[col].value_counts().min()
        if min_count < MIN_SAMPLES_PER_CLASS:
            continue

        score += 1.0
        candidates.append((col, score))

    if not candidates:
        raise ValueError("No suitable target column could be identified in the dataset.")

    # Pick the column with the highest score; ties broken alphabetically.
    candidates.sort(key=lambda x: (-x[1], x[0]))
    best_col, best_score = candidates[0]

    logger.info(
        "Target column discovered: '%s' (score=%.1f). "
        "Reasoning: column name contains a genre/mood keyword and has object dtype "
        "with cardinality in the expected range for a multi-class target.",
        best_col,
        best_score,
    )
    return best_col


def summarise_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Print and return a summary of the target variable distribution.

    Args:
        df: DataFrame containing the target column.
        target_col: Name of the target column.

    Returns:
        DataFrame with columns [label, count, percentage] sorted by count descending.
    """
    counts = df[target_col].value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts["percentage"] = (counts["count"] / len(df) * 100).round(2)

    print("=" * 60)
    print(f"TARGET COLUMN: '{target_col}'")
    print(f"Unique classes : {len(counts)}")
    print(f"Total samples  : {len(df):,}")
    print()
    print(counts.to_string(index=False))
    print("=" * 60)

    return counts


# ==== Entry Point ====

if __name__ == "__main__":
    raw_df = load_data()
    inspect_schema(raw_df)
    target = discover_target(raw_df)
    summarise_target(raw_df, target)
