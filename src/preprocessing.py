"""All cleaning, encoding, scaling, and splitting logic."""

# ==== Standard Library Imports ====
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

# ==== Third-Party Imports ====
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.feature_engineering import MusicFeatureEngineer

# ==== Internal Imports ====
from src.utils import (
    BOOL_COLUMN,
    DROP_COLUMNS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PREPROCESSOR_FILENAME,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    ensure_dirs,
    get_logger,
)

# ==== Module Logger ====
logger = get_logger(__name__)

# ==== Constants ====
# IQR multiplier used to define the outlier fence.
IQR_MULTIPLIER: float = 3.0
# Features where outlier clipping is appropriate (continuous audio features).
CLIP_FEATURES: List[str] = [
    "duration_ms",
    "tempo",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
]


# ==== Data Cleaning ====

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset.

    Strategy:
    - Text/metadata columns (artists, album_name, track_name) are dropped
      entirely before modelling, so their missing values are irrelevant.
    - Numeric feature columns contain no missing values in this dataset.
    - Any row where the target column is null is dropped to avoid silent errors.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with missing values addressed.
    """
    initial_rows = len(df)

    # Drop rows with a null target.
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    dropped = initial_rows - len(df)
    if dropped:
        logger.warning("Dropped %d rows with a null target value.", dropped)
    else:
        logger.info("No rows with a null target value detected.")

    # Apply genre taxonomy mapping: collapse 114 sub-genres into 22 acoustically
    # distinct super-genres so that near-duplicate labels (e.g. 'punk' and
    # 'punk-rock', 'indie' and 'indie-pop') no longer confuse the classifier.
    from src.genre_mapping import apply_genre_mapping
    original_n = df[TARGET_COLUMN].nunique()
    df[TARGET_COLUMN] = apply_genre_mapping(df[TARGET_COLUMN])
    new_n = df[TARGET_COLUMN].nunique()
    logger.info(
        "Genre mapping applied: %d original labels collapsed to %d super-genres.",
        original_n, new_n,
    )

    # Report on text column missing values (informational only).
    for col in DROP_COLUMNS:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing:
                logger.info(
                    "Column '%s' has %d missing value(s); this column will be dropped "
                    "before modelling so no imputation is needed.",
                    col,
                    n_missing,
                )

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode and cast feature columns to numeric types suitable for modelling.

    Encoding decisions:
    - `explicit` (boolean): cast to int (0 or 1). This is a binary ordinal
      variable; integer encoding preserves the natural ordering.
    - All other feature columns are already numeric and require no encoding.

    Args:
        df: DataFrame after missing-value handling.

    Returns:
        DataFrame with the explicit column cast to int.
    """
    df = df.copy()
    if BOOL_COLUMN in df.columns:
        df[BOOL_COLUMN] = df[BOOL_COLUMN].astype(int)
        logger.info("Cast '%s' from bool to int.", BOOL_COLUMN)
    return df


def detect_and_clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip extreme outliers in selected continuous features using the IQR method.

    Method: For each selected feature, values beyond
    Q1 - IQR_MULTIPLIER * IQR and Q3 + IQR_MULTIPLIER * IQR are clipped to
    those fence values.

    Justification: Clipping (Winsorisation) rather than deletion is chosen
    because (a) the dataset is large and losing rows is unnecessary, and (b)
    tree-based and distance-based models can be sensitive to extreme outliers
    that distort feature scales. A multiplier of 3.0 is used to be conservative,
    preserving genuine variation while removing only the most extreme values.

    Args:
        df: DataFrame with numeric features already encoded.

    Returns:
        DataFrame with outliers clipped in the selected columns.
    """
    df = df.copy()
    for col in CLIP_FEATURES:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_clipped:
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(
                "Clipped %d outlier(s) in '%s' [%.4f, %.4f].",
                n_clipped, col, lower, upper,
            )
    return df


def prepare_features_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract the feature matrix and target series from the DataFrame.

    Drops identifier and text columns that cannot be used as model features.

    Args:
        df: Preprocessed DataFrame.
        target_col: Name of the target column.

    Returns:
        A tuple of (X, y) where X is the feature DataFrame and y is the target Series.
    """
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    feature_df = df.drop(columns=cols_to_drop + [target_col], errors="ignore")

    # Keep only the expected numeric features.
    available = [c for c in NUMERIC_FEATURES if c in feature_df.columns]
    X = feature_df[available].copy()
    y = df[target_col].copy()

    logger.info("Feature matrix shape: %s | Target shape: %s", X.shape, y.shape)
    return X, y


def encode_target(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """Encode the target series to integer labels.

    Args:
        y: Target series with string class labels.

    Returns:
        A tuple of (y_encoded, label_encoder) where y_encoded is a numpy array
        of integer codes and label_encoder is a fitted LabelEncoder.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    logger.info("Encoded %d classes.", len(le.classes_))
    return y_encoded, le


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split into train, validation, and test sets.

    Split ratios are defined in utils.py (TRAIN_RATIO, VAL_RATIO, TEST_RATIO).
    Stratified sampling ensures each split has the same class distribution as
    the full dataset.

    Args:
        X: Feature matrix.
        y: Integer-encoded target array.

    Returns:
        A tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # First split off the test set.
    test_fraction = TEST_RATIO
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_fraction,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Then split the remaining data into train and validation.
    val_fraction_of_trainval = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor_pipeline() -> Pipeline:
    """Build a reusable sklearn preprocessing Pipeline.

    The pipeline contains two steps:

    1. MusicFeatureEngineer: a stateless transformer that appends 27
       domain-informed engineered features (log transforms, interaction
       terms, squared terms, tempo bins) to the original 15 audio features,
       producing a 42-dimensional feature matrix.

    2. StandardScaler: scales all 42 features to zero mean and unit variance.
       This is required for distance-based models (SVM, KNN, MLP) and does
       not harm tree-based models (LightGBM, Random Forest).

    Returns:
        An unfitted sklearn Pipeline with feature engineering and scaling steps.
    """
    pipeline = Pipeline(steps=[
        ("engineer", MusicFeatureEngineer()),
        ("scaler", StandardScaler()),
    ])
    logger.info("Preprocessor pipeline built: %s", [s[0] for s in pipeline.steps])
    return pipeline


def fit_and_transform(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the pipeline on the training set and transform all three splits.

    Args:
        pipeline: Unfitted sklearn Pipeline.
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        X_test: Test feature matrix.

    Returns:
        A tuple of (X_train_scaled, X_val_scaled, X_test_scaled).
    """
    X_train_scaled = pipeline.fit_transform(X_train)
    X_val_scaled = pipeline.transform(X_val)
    X_test_scaled = pipeline.transform(X_test)
    logger.info("Pipeline fitted on training data and applied to all splits.")
    return X_train_scaled, X_val_scaled, X_test_scaled


def save_pipeline(pipeline: Pipeline, path: Optional[Path] = None) -> Path:
    """Serialise the fitted pipeline to disk using pickle.

    Args:
        pipeline: Fitted sklearn Pipeline to save.
        path: Optional file path. Defaults to models/preprocessor.pkl.

    Returns:
        Path where the pipeline was saved.
    """
    ensure_dirs()
    if path is None:
        path = MODELS_DIR / PREPROCESSOR_FILENAME
    with open(path, "wb") as fh:
        pickle.dump(pipeline, fh)
    logger.info("Preprocessor pipeline saved to: %s", path)
    return path


def load_pipeline(path: Optional[Path] = None) -> Pipeline:
    """Load a serialised pipeline from disk.

    Args:
        path: Optional file path. Defaults to models/preprocessor.pkl.

    Returns:
        The loaded sklearn Pipeline.
    """
    if path is None:
        path = MODELS_DIR / PREPROCESSOR_FILENAME
    with open(path, "rb") as fh:
        pipeline = pickle.load(fh)
    logger.info("Preprocessor pipeline loaded from: %s", path)
    return pipeline


# ==== Entry Point ====

if __name__ == "__main__":
    from src.data_loader import load_data, discover_target

    raw_df = load_data()
    target_col = discover_target(raw_df)

    df_clean = handle_missing_values(raw_df)
    df_encoded = encode_features(df_clean)
    df_clipped = detect_and_clip_outliers(df_encoded)

    X, y_raw = prepare_features_target(df_clipped, target_col)
    y, le = encode_target(y_raw)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    pipeline = build_preprocessor_pipeline()
    X_train_sc, X_val_sc, X_test_sc = fit_and_transform(pipeline, X_train, X_val, X_test)
    save_pipeline(pipeline)

    print("Preprocessing complete.")
    print(f"X_train_scaled shape: {X_train_sc.shape}")
    print(f"X_val_scaled shape  : {X_val_sc.shape}")
    print(f"X_test_scaled shape : {X_test_sc.shape}")
