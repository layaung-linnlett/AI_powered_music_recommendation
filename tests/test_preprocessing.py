"""Tests for src/preprocessing.py.

Verifies that the pipeline produces a null-free output with the correct shape,
that train/test splits are stratified, and that the label encoder is consistent.
"""

# ==== Standard Library Imports ====
import sys
from pathlib import Path

# ==== Third-Party Imports ====
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ==== Internal Imports ====
from src.data_loader import discover_target, load_data
from src.preprocessing import (
    build_preprocessor_pipeline,
    detect_and_clip_outliers,
    encode_features,
    encode_target,
    fit_and_transform,
    handle_missing_values,
    load_pipeline,
    prepare_features_target,
    split_data,
)
from src.utils import MODELS_DIR, NUMERIC_FEATURES, RANDOM_SEED

# ==== Constants ====
EXPECTED_FEATURE_COUNT: int = 42  # 15 original + 27 engineered
EXPECTED_TRAIN_FRACTION_MIN: float = 0.68
EXPECTED_TRAIN_FRACTION_MAX: float = 0.72
EXPECTED_TEST_FRACTION_MIN: float = 0.13
EXPECTED_TEST_FRACTION_MAX: float = 0.17
STRATIFY_TOLERANCE: float = 0.03


# ==== Fixtures ====

@pytest.fixture(scope="module")
def prepared_data():
    """Run the full preprocessing chain and return (X_train, X_val, X_test, y_train, y_val, y_test, le)."""
    raw_df = load_data()
    target_col = discover_target(raw_df)
    df = handle_missing_values(raw_df)
    df = encode_features(df)
    df = detect_and_clip_outliers(df)
    X, y_raw = prepare_features_target(df, target_col)
    y, le = encode_target(y_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    pipeline = build_preprocessor_pipeline()
    X_train_sc, X_val_sc, X_test_sc = fit_and_transform(pipeline, X_train, X_val, X_test)
    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, le


# ==== Tests ====

class TestPipelineOutput:
    """Tests for the output of the full preprocessing pipeline."""

    def test_no_nulls_in_train(self, prepared_data) -> None:
        """The scaled training matrix must contain no NaN values."""
        X_train_sc = prepared_data[0]
        assert not np.isnan(X_train_sc).any(), "NaN values found in X_train_scaled."

    def test_no_nulls_in_val(self, prepared_data) -> None:
        """The scaled validation matrix must contain no NaN values."""
        X_val_sc = prepared_data[1]
        assert not np.isnan(X_val_sc).any(), "NaN values found in X_val_scaled."

    def test_no_nulls_in_test(self, prepared_data) -> None:
        """The scaled test matrix must contain no NaN values."""
        X_test_sc = prepared_data[2]
        assert not np.isnan(X_test_sc).any(), "NaN values found in X_test_scaled."

    def test_feature_count(self, prepared_data) -> None:
        """Each split must have exactly EXPECTED_FEATURE_COUNT columns."""
        for split_name, split in zip(("train", "val", "test"), prepared_data[:3]):
            assert split.shape[1] == EXPECTED_FEATURE_COUNT, (
                f"X_{split_name} has {split.shape[1]} features; "
                f"expected {EXPECTED_FEATURE_COUNT}."
            )

    def test_no_infinite_values(self, prepared_data) -> None:
        """No split should contain infinite values after scaling."""
        for split in prepared_data[:3]:
            assert not np.isinf(split).any(), "Infinite values found in a scaled split."


class TestSplitRatios:
    """Tests that split sizes match the configured ratios."""

    def test_train_size_fraction(self, prepared_data) -> None:
        """Training set fraction must be within the expected range."""
        X_tr, X_v, X_te = prepared_data[:3]
        total = len(X_tr) + len(X_v) + len(X_te)
        frac = len(X_tr) / total
        assert EXPECTED_TRAIN_FRACTION_MIN <= frac <= EXPECTED_TRAIN_FRACTION_MAX, (
            f"Train fraction {frac:.3f} outside [{EXPECTED_TRAIN_FRACTION_MIN}, "
            f"{EXPECTED_TRAIN_FRACTION_MAX}]."
        )

    def test_test_size_fraction(self, prepared_data) -> None:
        """Test set fraction must be within the expected range."""
        X_tr, X_v, X_te = prepared_data[:3]
        total = len(X_tr) + len(X_v) + len(X_te)
        frac = len(X_te) / total
        assert EXPECTED_TEST_FRACTION_MIN <= frac <= EXPECTED_TEST_FRACTION_MAX, (
            f"Test fraction {frac:.3f} outside [{EXPECTED_TEST_FRACTION_MIN}, "
            f"{EXPECTED_TEST_FRACTION_MAX}]."
        )


class TestStratification:
    """Tests that splits are stratified correctly."""

    def test_all_classes_in_train(self, prepared_data) -> None:
        """All target classes must appear in the training set."""
        _, _, _, y_train, _, _, le = prepared_data
        missing = set(range(len(le.classes_))) - set(y_train)
        assert not missing, f"{len(missing)} classes missing from training set."

    def test_all_classes_in_test(self, prepared_data) -> None:
        """All target classes must appear in the test set."""
        _, _, _, _, _, y_test, le = prepared_data
        missing = set(range(len(le.classes_))) - set(y_test)
        assert not missing, f"{len(missing)} classes missing from test set."

    def test_class_proportions_consistent(self, prepared_data) -> None:
        """Class proportions in train and test must be within STRATIFY_TOLERANCE."""
        _, _, _, y_train, _, y_test, le = prepared_data
        n_classes = len(le.classes_)
        for cls in range(n_classes):
            train_prop = np.mean(y_train == cls)
            test_prop = np.mean(y_test == cls)
            diff = abs(train_prop - test_prop)
            assert diff <= STRATIFY_TOLERANCE, (
                f"Class {le.classes_[cls]} proportion differs by {diff:.4f} "
                f"(train={train_prop:.4f}, test={test_prop:.4f})."
            )


class TestSavedPipeline:
    """Tests for the persisted preprocessor pipeline."""

    def test_pipeline_file_exists(self) -> None:
        """The serialised preprocessor pipeline file must exist."""
        pipeline_path = MODELS_DIR / "preprocessor.pkl"
        assert pipeline_path.exists(), f"preprocessor.pkl not found at {pipeline_path}"

    def test_pipeline_loads(self) -> None:
        """The saved pipeline must be loadable."""
        pipeline = load_pipeline()
        assert pipeline is not None

    def test_pipeline_is_sklearn_pipeline(self) -> None:
        """The loaded object must be an sklearn Pipeline."""
        pipeline = load_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_two_steps(self) -> None:
        """The pipeline must have exactly two steps: engineer and scaler."""
        pipeline = load_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert len(step_names) == 2, f"Expected 2 steps, got: {step_names}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
