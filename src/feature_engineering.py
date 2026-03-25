"""Feature engineering transformer for the music mood classifier.

All engineered features are derived from domain knowledge about how Spotify
audio attributes relate to music genre. The transformer is implemented as a
sklearn-compatible BaseEstimator so it can be embedded in a Pipeline and
serialised cleanly with the rest of the preprocessor.
"""

# ==== Standard Library Imports ====
from typing import List, Optional

# ==== Third-Party Imports ====
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ==== Internal Imports ====
from src.utils import get_logger

# ==== Module Logger ====
logger = get_logger(__name__)

# ==== Constants ====
# Small epsilon added to denominators to avoid division by zero.
EPSILON: float = 1e-6

# Features that are heavily right-skewed and benefit from a log1p transform.
LOG_TRANSFORM_FEATURES: List[str] = [
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
]

# Milliseconds to minutes conversion factor.
MS_TO_MIN: float = 60_000.0

# Maximum realistic tempo (BPM) used to normalise the tempo feature.
MAX_TEMPO: float = 250.0

# Maximum loudness magnitude used to normalise (loudness is negative in dB).
MAX_LOUDNESS_ABS: float = 60.0


# ==== Transformer ====

class MusicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that adds domain-informed engineered features.

    All new features are appended as additional columns to the input DataFrame
    or numpy array. The transformer is stateless (fit() is a no-op) so it is
    safe to use in a Pipeline before a stateful scaler.

    Engineered feature groups
    -------------------------
    1. Log-transformed skewed features (speechiness, acousticness, etc.).
    2. Duration converted to minutes and log-scaled.
    3. Absolute loudness (genres like metal are loud; classical is quiet).
    4. Pairwise audio interaction terms that capture genre-discriminating
       joint properties (e.g. high energy AND low acousticness = rock/metal).
    5. Squared terms for non-linear relationships (tempo, popularity).
    6. Key-mode interaction (musical key context).
    7. Normalised tempo (divided by MAX_TEMPO to linearise the scale).
    """

    # Names of features this transformer expects as input, in order.
    INPUT_FEATURES: List[str] = [
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

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MusicFeatureEngineer":
        """No-op fit: the transformer has no learned parameters.

        Args:
            X: Input feature matrix (ignored).
            y: Target array (ignored).

        Returns:
            self
        """
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute and append engineered features to the input matrix.

        Args:
            X: Input feature matrix with columns in the order defined by
               INPUT_FEATURES. Accepts both numpy arrays and pandas DataFrames.

        Returns:
            numpy array with the original columns followed by all engineered
            feature columns.
        """
        if isinstance(X, pd.DataFrame):
            df = X[self.INPUT_FEATURES].copy().astype(float)
        else:
            df = pd.DataFrame(X, columns=self.INPUT_FEATURES).astype(float)

        engineered: dict = {}

        # ==== Group 1: Log transforms of right-skewed features ====
        for col in LOG_TRANSFORM_FEATURES:
            engineered[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

        # ==== Group 2: Duration engineering ====
        engineered["duration_min"] = df["duration_ms"] / MS_TO_MIN
        engineered["log_duration_ms"] = np.log1p(df["duration_ms"].clip(lower=0))

        # ==== Group 3: Loudness (genres like metal are loud; acoustic is quiet) ====
        engineered["abs_loudness"] = df["loudness"].abs()
        engineered["loudness_norm"] = df["loudness"].abs() / MAX_LOUDNESS_ABS

        # ==== Group 4: Pairwise interaction terms ====
        # Electric signal: high energy with low acousticness (rock, metal, EDM).
        engineered["energy_x_not_acoustic"] = df["energy"] * (1.0 - df["acousticness"])

        # Dance energy: captures dance/EDM vs slow/acoustic music.
        engineered["dance_x_energy"] = df["danceability"] * df["energy"]

        # Mood valence combined with energy: happy-energetic vs sad-calm.
        engineered["valence_x_energy"] = df["valence"] * df["energy"]

        # Happy dance: high valence AND high danceability (party, disco, pop).
        engineered["valence_x_dance"] = df["valence"] * df["danceability"]

        # Speech over acoustic: distinguishes spoken word from instrumental.
        engineered["speech_x_not_acoustic"] = (
            df["speechiness"] * (1.0 - df["acousticness"])
        )

        # Instrumental loudness: captures orchestral vs ambient instrumental.
        engineered["instrumental_x_energy"] = df["instrumentalness"] * df["energy"]

        # Loudness per unit energy (compression/dynamics proxy).
        engineered["loudness_per_energy"] = (
            df["loudness"].abs() / (df["energy"] + EPSILON)
        )

        # Popularity times danceability: mainstream dance tracks.
        engineered["pop_x_dance"] = (df["popularity"] / 100.0) * df["danceability"]

        # Acoustic valence: gentle acoustic music (folk, singer-songwriter).
        engineered["acoustic_x_valence"] = df["acousticness"] * df["valence"]

        # ==== Group 5: Squared terms (non-linear relationships) ====
        engineered["tempo_sq"] = (df["tempo"] / MAX_TEMPO) ** 2
        engineered["popularity_sq"] = (df["popularity"] / 100.0) ** 2
        engineered["energy_sq"] = df["energy"] ** 2
        engineered["acousticness_sq"] = df["acousticness"] ** 2
        engineered["instrumentalness_sq"] = df["instrumentalness"] ** 2

        # ==== Group 6: Tempo normalisation and binning ====
        engineered["tempo_norm"] = df["tempo"] / MAX_TEMPO
        # Tempo range indicator: slow (<90), mid (90-140), fast (>140).
        engineered["tempo_slow"] = (df["tempo"] < 90).astype(float)
        engineered["tempo_fast"] = (df["tempo"] > 140).astype(float)

        # ==== Group 7: Key-mode interaction ====
        # Encodes whether the track is in a major or minor key.
        engineered["key_x_mode"] = df["key"] * df["mode"]

        # ==== Group 8: Liveness ratio ====
        # Live recordings are common in jazz, classical, and folk.
        engineered["liveness_ratio"] = df["liveness"] / (df["energy"] + EPSILON)

        # ==== Assemble output ====
        engineered_df = pd.DataFrame(engineered, index=df.index)
        result = np.hstack([df.values, engineered_df.values])
        return result

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return the full ordered list of output feature names.

        Args:
            input_features: Ignored; present for sklearn API compatibility.

        Returns:
            List of feature name strings.
        """
        engineered_names = [
            "log1p_speechiness",
            "log1p_acousticness",
            "log1p_instrumentalness",
            "log1p_liveness",
            "duration_min",
            "log_duration_ms",
            "abs_loudness",
            "loudness_norm",
            "energy_x_not_acoustic",
            "dance_x_energy",
            "valence_x_energy",
            "valence_x_dance",
            "speech_x_not_acoustic",
            "instrumental_x_energy",
            "loudness_per_energy",
            "pop_x_dance",
            "acoustic_x_valence",
            "tempo_sq",
            "popularity_sq",
            "energy_sq",
            "acousticness_sq",
            "instrumentalness_sq",
            "tempo_norm",
            "tempo_slow",
            "tempo_fast",
            "key_x_mode",
            "liveness_ratio",
        ]
        return self.INPUT_FEATURES + engineered_names


# ==== Entry Point ====

if __name__ == "__main__":
    from src.data_loader import load_data, discover_target
    from src.preprocessing import (
        handle_missing_values,
        encode_features,
        detect_and_clip_outliers,
        prepare_features_target,
    )

    raw_df = load_data()
    target_col = discover_target(raw_df)
    df = handle_missing_values(raw_df)
    df = encode_features(df)
    df = detect_and_clip_outliers(df)
    X, _ = prepare_features_target(df, target_col)

    eng = MusicFeatureEngineer()
    X_out = eng.transform(X)

    feature_names = eng.get_feature_names_out()
    print(f"Input features  : {X.shape[1]}")
    print(f"Output features : {X_out.shape[1]}")
    print(f"Feature names   : {feature_names}")
