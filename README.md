# Music Mood Classifier

A machine learning pipeline that classifies Spotify tracks into six broad mood/style categories from audio features alone, with a Streamlit UI on top.

---

## Key Findings

- Final model (LightGBM) reaches **69.03% test accuracy**, weighted F1 of **0.6845**, macro F1 of **0.6444**, and macro ROC-AUC (one-vs-rest) of **0.9045** on the held-out test set.
- Starting from the raw 114 Spotify genre labels, the model only reached **~30-35% accuracy** — genres like "punk" and "punk-rock" are acoustically indistinguishable, so no feature-based model can separate them. Collapsing them into 6 acoustically distinct super-genres, through four rounds of per-class accuracy analysis, took accuracy from ~31% to 69.03% (roughly a 2x improvement).
- The macro ROC-AUC of 0.90 is well above the 69% accuracy figure. That gap is expected, not a bug: ROC-AUC rewards the model for ranking the correct class highly even when the top-1 hard prediction is wrong, which happens a lot at the boundary between classes like `alternative`/`heavy` or `dance`/`electronic` that share overlapping audio signatures.
- Full breakdown of what was tried to push past 69% and why 80% wasn't reachable with these features is in [`reports/improvement_log.md`](reports/improvement_log.md).

| Metric | Value |
|--------|-------|
| Test accuracy | 69.03% |
| Weighted F1 | 0.6845 |
| Macro F1 | 0.6444 |
| Macro ROC-AUC (OVR) | 0.9045 |

Confusion matrix and ROC curves are saved to `reports/figures/`.

---

## Screenshots

*(Not yet added — the Streamlit apps under `ui/` and `ui-mood/` need screenshots or a short demo GIF here. See "Remaining issues" in the audit notes.)*

---

## Tech Stack

- **Language:** Python 3.11+
- **Modelling:** scikit-learn, LightGBM, Optuna (hyperparameter tuning)
- **Data handling:** pandas, NumPy
- **Visualisation:** matplotlib, seaborn
- **Testing:** pytest
- **UI:** Streamlit (two separate apps — see below)

Full pinned versions are in [`requirements.txt`](requirements.txt).

---

## Methodology

### Dataset

The dataset contains 114,000 Spotify tracks with 15 audio features per track (danceability, energy, acousticness, speechiness, tempo, etc.) and one of 114 original genre labels (`track_genre`). It's perfectly balanced — 1,000 samples per genre. Three rows have a missing `artists`/`album_name`/`track_name` value, but those text columns are dropped before modelling anyway, so it doesn't matter.

| Property | Value |
|----------|-------|
| Rows | 114,000 |
| Original genre labels | 114 |
| Audio features | 15 |
| Source | Spotify track metadata via Kaggle (`track_genre` column) |

### Genre taxonomy: 114 labels to 6 categories

The 114 original Spotify sub-genre labels are collapsed into 6 broad categories. I went through four rounds of per-class accuracy analysis, merging categories that were consistently confused with each other, until each remaining class had at least one clearly distinguishing audio axis.

| Category | Sub-genres included | Key audio signature |
|----------|---------------------|---------------------|
| **acoustic** | folk, classical, ambient, blues, jazz, romance, sleep, study | Very high acousticness, low energy |
| **alternative** | indie, grunge, rock, alt-rock, psych-rock | Moderate-high energy, guitar-driven, low acousticness |
| **dance** | latin, pop, dance, R&B, soul, reggae, k-pop, j-pop, world music | Very high danceability, moderate-high valence |
| **electronic** | EDM, house, techno, trance, drum-and-bass, dubstep | Very high energy, very low acousticness, high instrumentalness |
| **heavy** | metal, punk, hardcore, emo, goth | Very high energy, maximum loudness, low valence |
| **vocal** | hip-hop, rap, children, comedy | Very high speechiness |

The full reasoning for each merge is documented in `src/genre_mapping.py`.

### Feature engineering

`MusicFeatureEngineer` (a custom sklearn transformer) expands the 15 raw features to 42:

- Log-transformed features: speechiness, acousticness, instrumentalness, liveness
- Duration conversions: `duration_min`, `log_duration_ms`
- Loudness transformations: `abs_loudness`, `loudness_norm`
- Interaction terms: `energy x danceability`, `valence x energy`, etc.
- Squared terms: tempo, popularity, energy, acousticness, instrumentalness
- Tempo bins: `tempo_slow`, `tempo_fast`, `tempo_norm`
- Key-mode interaction: `key_x_mode`

### Model

| Component | Detail |
|-----------|--------|
| Algorithm | LightGBM (`LGBMClassifier`) |
| Preprocessing | `StandardScaler` fitted on train set only |
| Hyperparameters | `n_estimators=1000`, `num_leaves=511`, `learning_rate=0.05` |
| Train/val/test split | 70% / 15% / 15%, stratified |
| Final training | Trained on combined train+val (96,900 samples) |

LightGBM was picked over Random Forest even though Random Forest had a marginally higher CV F1 at the initial 114-class stage — the gap (0.0078) was too small to be meaningful at that many classes, and LightGBM had more headroom to improve once classes were reduced and tuning was applied. That held up: LightGBM ended up ahead after tuning. Full comparison table and rationale in [`reports/model_selection.md`](reports/model_selection.md).

---

## Project Structure

```
Music_Mood_Classifier/
├── data/
│   ├── raw/               Dataset CSV expected here for the src/ pipeline (not tracked in git)
│   └── README.md          Column reference and dataset notes
├── models/                Serialised artefacts (tracked, except final_model.pkl — see note below)
│   ├── final_model.pkl    Trained LightGBM classifier (~330 MB, gitignored — regenerate with model_training.py)
│   ├── preprocessor.pkl   Fitted sklearn Pipeline
│   ├── label_encoder.pkl  Fitted LabelEncoder
│   ├── cv_results.pkl     Cached cross-validation results
│   └── README.md
├── notebooks/              Exploratory / walkthrough notebooks (run in order 01-05)
│   ├── 01_data_and_eda.ipynb
│   ├── 02_preprocessing_and_features.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_ui_notebook_demo.ipynb
│   └── dataset.csv         Self-contained copy of the dataset used by the notebooks
├── reports/                Generated reports and figures (tracked)
│   ├── figures/            Confusion matrix, ROC curves, EDA plots
│   ├── eda_summary.md
│   ├── evaluation_report.md
│   ├── model_selection.md
│   ├── improvement_log.md
│   └── README.md
├── src/                    Source code
│   ├── data_loader.py      CSV auto-discovery and schema inspection
│   ├── eda.py               Exploratory data analysis and figures
│   ├── feature_engineering.py  Custom sklearn transformer (42 features)
│   ├── genre_mapping.py    114-genre to 6-class taxonomy
│   ├── model_training.py   CV comparison, tuning, and final training
│   ├── predict.py           Inference pipeline
│   ├── preprocessing.py    Cleaning, encoding, scaling, splitting
│   ├── evaluation.py        Metrics and visualisation
│   ├── utils.py             Shared constants and logger
│   └── README.md
├── tests/                   pytest test suite
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── ui/
│   ├── app.py               Streamlit app: manual slider entry + batch CSV upload
│   └── README.md
├── ui-mood/
│   └── app.py                Streamlit app: mood text input, quick-mood buttons, and playlist recommendations, plus the same slider/batch modes as ui/app.py
├── CONTRIBUTING.md          Contribution guidelines
├── README.md                This file
└── requirements.txt         Python dependencies
```

**Note on the dataset:** the `src/` pipeline (via `data_loader.find_csv()`) looks for a CSV in `data/raw/`, which isn't tracked in git — you need to place the file there yourself. The notebooks, on the other hand, load a separate, self-contained copy committed at `notebooks/dataset.csv`. Both are the same underlying data; they're just loaded two different ways depending on whether you're running the `src/` pipeline or a notebook.

---

## How to Run

### Prerequisites

- Python 3.11+
- 8 GB RAM minimum (16 GB recommended for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/layaung-linnlett/Music_Mood_Classifier.git
cd Music_Mood_Classifier

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Dataset

For the `src/` pipeline: place the Spotify dataset CSV in `data/raw/`. It's auto-discovered at runtime by scanning that folder for any `.csv` file — no hardcoded file path.

For the notebooks: `notebooks/dataset.csv` is already included.

### Run the full training pipeline

```bash
# Step 1: Exploratory data analysis
python -m src.eda

# Step 2: Preprocessing (cleans data, builds pipeline, saves splits)
python -m src.preprocessing

# Step 3: Model selection and training
python -m src.model_training

# Step 4: Evaluation
python -m src.evaluation
```

### Run individual modules

```bash
python -m src.data_loader      # Inspect the dataset
python -m src.genre_mapping    # Show genre distribution after mapping
python -m src.predict          # Run inference on example tracks
```

### Launch a Streamlit UI

```bash
streamlit run ui/app.py        # manual sliders + batch CSV upload
streamlit run ui-mood/app.py   # mood text input + playlist recommendations
```

Both apps open at `http://localhost:8501` and load `models/final_model.pkl`, `models/preprocessor.pkl`, and `models/label_encoder.pkl` — run the training pipeline first if these don't exist locally.

### Run tests

```bash
pytest tests/ -v
```

---

## Limitations & Future Work

The model plateaus at 69.03% accuracy. Reasons and what I tried are documented in detail in [`reports/improvement_log.md`](reports/improvement_log.md); short version:

- Spotify's 15 audio features are perceptual summaries, not raw audio. Only about four or five axes (acousticness, energy+loudness, danceability, speechiness) actually separate genres — feature engineering can't create new signal that isn't in the original data, it can only reshape what's there.
- Tuning the model further (class weighting, more trees, different leaf counts) moved accuracy by about ±1%, not the several points that would be needed to reach 80%.
- What would likely get closer to 80%: raw audio features (MFCCs, spectral centroid) instead of Spotify's summary stats, lyrics as a second input (hip-hop vs. folk vs. comedy is obvious from the words), or fewer than 6 classes (at that point the classifier is less useful, so I didn't pursue it).
- `vocal` (recall 0.45) and `alternative` (recall 0.45) are the weakest classes in the per-class report — see `reports/evaluation_report.md`.
- `ui-mood/app.py`'s mood-to-genre mapping is a hand-written keyword matcher, not a learned model — it's a simple layer on top of the trained classifier, not a separate ML component.

---

## Contact

GitHub: [github.com/layaung-linnlett/Music_Mood_Classifier](https://github.com/layaung-linnlett/Music_Mood_Classifier)

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, commit message style, and code standards if you want to contribute.
