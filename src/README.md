# Source Code (`src/`)

This package contains all Python source modules for the Music Mood Classifier.

## Module Overview

| Module | Purpose |
|--------|---------|
| `utils.py` | Shared constants, directory paths, logger factory, `ensure_dirs()` |
| `data_loader.py` | CSV discovery, DataFrame loading, schema inspection, target discovery |
| `feature_engineering.py` | `MusicFeatureEngineer` sklearn transformer (adds 27 engineered features) |
| `preprocessing.py` | Missing-value handling, encoding, outlier clipping, splits, pipeline |
| `eda.py` | EDA plots and the written `eda_summary.md` report |
| `model_training.py` | Candidate model definitions, cross-validation, GridSearchCV tuning, save/load |
| `evaluation.py` | Test-set metrics, confusion matrix, ROC curves, evaluation report |
| `predict.py` | Inference pipeline for new inputs (single record or batch CSV) |

## Data Flow

```
data/raw/dataset.csv
        |
        v
data_loader.load_data()          # load raw DataFrame
        |
        v
preprocessing (clean, encode, clip)
        |
        v
preprocessing.split_data()       # stratified 70/15/15 split
        |
        v
Pipeline([MusicFeatureEngineer, StandardScaler]).fit_transform()
        |
        v
model_training.cross_validate_models()   # 3-fold CV on 10k subsample
        |
        v
model_training.tune_model_grid()         # GridSearchCV (16 combos, 3-fold)
        |
        v
model_training.train_final_model()       # fit on full 79,800-sample train set
        |
        v
evaluation.compute_metrics()             # evaluate on held-out test set
        |
        v
predict.predict()                        # inference on new data
```

## Module Dependencies

```
utils  <--- all modules
data_loader  <--- preprocessing, eda
feature_engineering  <--- preprocessing
preprocessing  <--- model_training, evaluation, predict
model_training  <--- evaluation
evaluation  <--- (standalone, run after training)
predict  <--- (standalone, uses saved pkl files)
```

## Running Individual Modules

Each module has an `if __name__ == "__main__":` guard. You can run them in order:

```bash
python -m src.data_loader       # inspect the dataset
python -m src.eda               # generate EDA figures and report
python -m src.preprocessing     # build pipeline and splits
python -m src.model_training    # CV comparison, tuning, and final training
python -m src.evaluation        # test-set evaluation and plots
python -m src.predict           # demo inference on a few samples
```
