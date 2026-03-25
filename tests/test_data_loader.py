"""Tests for src/data_loader.py.

Verifies that the CSV is found automatically, columns are as expected,
the target column is correctly identified, and the DataFrame is non-empty.
"""

# ==== Standard Library Imports ====
import sys
from pathlib import Path

# ==== Third-Party Imports ====
import pandas as pd
import pytest

# Ensure the project root is on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ==== Internal Imports ====
from src.data_loader import discover_target, find_csv, load_data, summarise_target
from src.utils import DROP_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN

# ==== Constants ====
EXPECTED_MIN_ROWS: int = 1000
EXPECTED_TARGET_COLUMN: str = "track_genre"
EXPECTED_NUMERIC_FEATURES: list = [
    "popularity", "duration_ms", "danceability", "energy",
    "loudness", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]


# ==== Fixtures ====

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Load the raw dataset once for all tests in this module."""
    return load_data()


# ==== Tests ====

class TestFindCsv:
    """Tests for the find_csv() helper."""

    def test_csv_file_found(self) -> None:
        """A CSV file must be discoverable in data/raw/."""
        path = find_csv()
        assert path.exists(), f"CSV not found at expected path: {path}"

    def test_csv_is_file(self) -> None:
        """The discovered path must point to a file, not a directory."""
        path = find_csv()
        assert path.is_file()

    def test_csv_extension(self) -> None:
        """The discovered file must have a .csv extension."""
        path = find_csv()
        assert path.suffix == ".csv"


class TestLoadData:
    """Tests for the load_data() function."""

    def test_returns_dataframe(self, raw_df: pd.DataFrame) -> None:
        """load_data() must return a pandas DataFrame."""
        assert isinstance(raw_df, pd.DataFrame)

    def test_dataframe_not_empty(self, raw_df: pd.DataFrame) -> None:
        """The loaded DataFrame must have at least EXPECTED_MIN_ROWS rows."""
        assert len(raw_df) >= EXPECTED_MIN_ROWS, (
            f"DataFrame has only {len(raw_df)} rows; expected at least {EXPECTED_MIN_ROWS}."
        )

    def test_target_column_present(self, raw_df: pd.DataFrame) -> None:
        """The target column must be present in the loaded data."""
        assert EXPECTED_TARGET_COLUMN in raw_df.columns, (
            f"Column '{EXPECTED_TARGET_COLUMN}' not found. "
            f"Available columns: {raw_df.columns.tolist()}"
        )

    def test_numeric_features_present(self, raw_df: pd.DataFrame) -> None:
        """Core numeric feature columns must all be present."""
        missing = [c for c in EXPECTED_NUMERIC_FEATURES if c not in raw_df.columns]
        assert not missing, f"Missing feature columns: {missing}"

    def test_no_all_null_columns(self, raw_df: pd.DataFrame) -> None:
        """No column should be entirely null."""
        all_null = [c for c in raw_df.columns if raw_df[c].isnull().all()]
        assert not all_null, f"Columns that are entirely null: {all_null}"


class TestDiscoverTarget:
    """Tests for the discover_target() function."""

    def test_returns_string(self, raw_df: pd.DataFrame) -> None:
        """discover_target() must return a string column name."""
        target = discover_target(raw_df)
        assert isinstance(target, str)

    def test_identifies_correct_target(self, raw_df: pd.DataFrame) -> None:
        """The discovered target must be 'track_genre'."""
        target = discover_target(raw_df)
        assert target == EXPECTED_TARGET_COLUMN, (
            f"Expected target '{EXPECTED_TARGET_COLUMN}', got '{target}'."
        )

    def test_target_column_exists_in_dataframe(self, raw_df: pd.DataFrame) -> None:
        """The discovered target column must actually exist in the DataFrame."""
        target = discover_target(raw_df)
        assert target in raw_df.columns

    def test_target_has_multiple_classes(self, raw_df: pd.DataFrame) -> None:
        """The target column must contain more than one unique class."""
        target = discover_target(raw_df)
        n_unique = raw_df[target].nunique()
        assert n_unique > 1, f"Target has only {n_unique} unique value(s)."


class TestSummariseTarget:
    """Tests for the summarise_target() function."""

    def test_returns_dataframe(self, raw_df: pd.DataFrame) -> None:
        """summarise_target() must return a DataFrame."""
        summary = summarise_target(raw_df, EXPECTED_TARGET_COLUMN)
        assert isinstance(summary, pd.DataFrame)

    def test_summary_columns(self, raw_df: pd.DataFrame) -> None:
        """Summary must contain label, count, and percentage columns."""
        summary = summarise_target(raw_df, EXPECTED_TARGET_COLUMN)
        for col in ("label", "count", "percentage"):
            assert col in summary.columns

    def test_percentages_sum_to_100(self, raw_df: pd.DataFrame) -> None:
        """Percentages must sum to approximately 100.

        Tolerance is 1.0 to allow for rounding error in per-class percentages.
        With 114 classes each rounded to 2 decimal places, accumulated rounding
        error can reach up to 0.57 percentage points.
        """
        summary = summarise_target(raw_df, EXPECTED_TARGET_COLUMN)
        total = summary["percentage"].sum()
        assert abs(total - 100.0) < 1.0, f"Percentages sum to {total:.2f}, expected 100."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
