"""Tests for model loading and inference.

Verifies that the final model can be loaded from disk, that predict() returns
the correct shape, and that all returned labels belong to the discovered class set.
"""

# ==== Standard Library Imports ====
import sys
from pathlib import Path

# ==== Third-Party Imports ====
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ==== Internal Imports ====
from src.model_training import load_model
from src.preprocessing import load_pipeline
from src.utils import MODELS_DIR

# ==== Constants ====
# Number of test samples to run inference on.
N_INFERENCE_SAMPLES: int = 50
# Expected number of engineered features.
EXPECTED_N_FEATURES: int = 42


# ==== Fixtures ====

@pytest.fixture(scope="module")
def model():
    """Load the final trained model from disk."""
    return load_model()


@pytest.fixture(scope="module")
def pipeline():
    """Load the saved preprocessor pipeline from disk."""
    return load_pipeline()


@pytest.fixture(scope="module")
def label_encoder():
    """Load the saved LabelEncoder from disk."""
    import pickle
    le_path = MODELS_DIR / "label_encoder.pkl"
    with open(le_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def sample_features(pipeline):
    """Return a small numpy array of scaled feature vectors for inference."""
    rng = np.random.default_rng(42)
    # Generate random raw features in plausible ranges.
    n = N_INFERENCE_SAMPLES
    raw = {
        "popularity":        rng.integers(0, 100, size=n).astype(float),
        "duration_ms":       rng.integers(60000, 400000, size=n).astype(float),
        "explicit":          rng.integers(0, 2, size=n).astype(float),
        "danceability":      rng.uniform(0, 1, size=n),
        "energy":            rng.uniform(0, 1, size=n),
        "key":               rng.integers(0, 12, size=n).astype(float),
        "loudness":          rng.uniform(-30, 0, size=n),
        "mode":              rng.integers(0, 2, size=n).astype(float),
        "speechiness":       rng.uniform(0, 0.5, size=n),
        "acousticness":      rng.uniform(0, 1, size=n),
        "instrumentalness":  rng.uniform(0, 1, size=n),
        "liveness":          rng.uniform(0, 1, size=n),
        "valence":           rng.uniform(0, 1, size=n),
        "tempo":             rng.uniform(60, 200, size=n),
        "time_signature":    rng.integers(3, 6, size=n).astype(float),
    }
    import pandas as pd
    df_raw = pd.DataFrame(raw)
    return pipeline.transform(df_raw)


# ==== Tests ====

class TestModelLoading:
    """Tests for model persistence."""

    def test_model_file_exists(self) -> None:
        """The final_model.pkl file must exist in the models directory."""
        model_path = MODELS_DIR / "final_model.pkl"
        assert model_path.exists(), f"final_model.pkl not found at {model_path}"

    def test_model_loads_without_error(self, model) -> None:
        """Loading the model must not raise any exception."""
        assert model is not None

    def test_model_has_predict(self, model) -> None:
        """The loaded model must expose a predict() method."""
        assert hasattr(model, "predict"), "Model has no predict() method."

    def test_label_encoder_file_exists(self) -> None:
        """The label_encoder.pkl file must exist."""
        le_path = MODELS_DIR / "label_encoder.pkl"
        assert le_path.exists(), f"label_encoder.pkl not found at {le_path}"


class TestPredictShape:
    """Tests for the shape and type of predict() output."""

    def test_predict_returns_array(self, model, sample_features) -> None:
        """predict() must return a numpy array or list."""
        y_pred = model.predict(sample_features)
        assert hasattr(y_pred, "__len__"), "predict() output has no length."

    def test_predict_correct_length(self, model, sample_features) -> None:
        """predict() must return one label per input sample."""
        y_pred = model.predict(sample_features)
        assert len(y_pred) == N_INFERENCE_SAMPLES, (
            f"Expected {N_INFERENCE_SAMPLES} predictions, got {len(y_pred)}."
        )

    def test_predict_no_nans(self, model, sample_features) -> None:
        """predict() must not return any NaN values."""
        y_pred = model.predict(sample_features)
        arr = np.array(y_pred)
        assert not np.isnan(arr.astype(float)).any(), "predict() returned NaN labels."


class TestPredictLabels:
    """Tests that predicted labels are valid class indices."""

    def test_all_labels_in_class_set(self, model, sample_features, label_encoder) -> None:
        """All predicted integer labels must correspond to a known class."""
        y_pred = model.predict(sample_features)
        valid_labels = set(range(len(label_encoder.classes_)))
        invalid = set(y_pred) - valid_labels
        assert not invalid, (
            f"Model returned labels outside the known class set: {invalid}"
        )

    def test_decoded_labels_are_strings(self, model, sample_features, label_encoder) -> None:
        """Decoding predicted integer labels via LabelEncoder must yield strings."""
        y_pred = model.predict(sample_features)
        decoded = label_encoder.inverse_transform(y_pred)
        assert all(isinstance(lbl, str) for lbl in decoded), (
            "Some decoded labels are not strings."
        )

    def test_decoded_labels_in_known_genres(self, model, sample_features, label_encoder) -> None:
        """All decoded labels must belong to the set of discovered genre names."""
        y_pred = model.predict(sample_features)
        decoded = set(label_encoder.inverse_transform(y_pred))
        known = set(label_encoder.classes_)
        unknown = decoded - known
        assert not unknown, f"Unknown genre labels returned: {unknown}"


class TestPredictProba:
    """Tests for probabilistic output (if supported by the model)."""

    def test_proba_shape(self, model, sample_features, label_encoder) -> None:
        """predict_proba() must return shape (n_samples, n_classes) when available."""
        if not hasattr(model, "predict_proba"):
            pytest.skip("Model does not support predict_proba.")
        proba = model.predict_proba(sample_features)
        n_classes = len(label_encoder.classes_)
        assert proba.shape == (N_INFERENCE_SAMPLES, n_classes), (
            f"predict_proba shape {proba.shape} != expected ({N_INFERENCE_SAMPLES}, {n_classes})."
        )

    def test_proba_sums_to_one(self, model, sample_features) -> None:
        """Each row of predict_proba() must sum to approximately 1."""
        if not hasattr(model, "predict_proba"):
            pytest.skip("Model does not support predict_proba.")
        proba = model.predict_proba(sample_features)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"predict_proba rows do not sum to 1. Min={row_sums.min():.6f}, Max={row_sums.max():.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
