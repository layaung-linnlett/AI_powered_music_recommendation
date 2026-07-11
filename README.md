# Music Mood Classifier — sorting 114,000 Spotify tracks into moods the model can actually tell apart

Spotify tags every track with one of 114 micro-genres, but genre labels don't map cleanly onto how music actually *feels*, and a classifier trained directly on those 114 labels barely beats random guessing (31% accuracy). This project uses only Spotify's 15 audio features (danceability, energy, acousticness, tempo, etc. — no lyrics, no metadata) to predict a track's mood, and treats the genre taxonomy itself as something to be engineered: through four rounds of confusion-matrix analysis, the 114 raw labels were collapsed into 6 audio-distinguishable mood categories, lifting accuracy from 31% to 68.29% along the way. The full reasoning for every merge — including why 80% accuracy isn't reachable with these features — is documented and treated as a finding in its own right, not a shortfall to hide.

## Key Findings

- **Genre taxonomy design was the single biggest lever**, not model tuning: collapsing 114 raw Spotify genres into 6 audio-distinguishable categories took accuracy from 31% → 68.29% (a 2.2x improvement), while hyperparameter tuning alone moved it by roughly 1%.
- **Final model**: 68.29% test accuracy, 0.6761 weighted F1, and 0.9081 macro ROC-AUC — the high ROC-AUC shows the model ranks genres correctly most of the time even where the raw accuracy number looks unremarkable.
- **80% accuracy is provably out of reach with audio features alone.** Only 4-5 real axes of variation exist in Spotify's 15 features (acousticness, energy/loudness, danceability, speechiness); every engineered feature is a derivative of those same axes, so the ceiling doesn't move. This is demonstrated, not assumed — see `reports/improvement_log.md`.
- **Feature engineering added real signal**: expanding 15 raw features to 42 (log transforms, interaction terms, tempo bins) via a custom `MusicFeatureEngineer` sklearn transformer improved F1 from 0.2432 → 0.2581 in early testing on the harder 114-class problem.
- Trained on **114,000 tracks** with a stratified 70/15/15 train/val/test split.

## Tech Stack

| Tool | Purpose |
|------|---------|
| pandas, numpy | Data loading and manipulation |
| scikit-learn | Preprocessing pipeline, baseline models, evaluation metrics |
| LightGBM | Final classifier |
| Optuna | Hyperparameter tuning (TPE sampler) |
| imbalanced-learn | Class imbalance handling |
| matplotlib, seaborn | Static charts |
| Streamlit | Two interactive demo apps |

## Methodology

1. **EDA** on all 114,000 tracks and 114 raw genre labels — checked class balance, missing values, feature distributions, and mutual information against the target.
2. **Iterative taxonomy design**: trained a baseline model on the raw 114 labels (31% accuracy), then used per-class accuracy and the confusion matrix to identify which labels the model could never tell apart, merging them into broader categories across 4 rounds until each remaining class had a clear distinguishing audio signature. Final taxonomy: 6 classes (acoustic, alternative, dance, electronic, heavy, vocal).
3. **Feature engineering**: a custom `MusicFeatureEngineer` sklearn transformer expands the 15 raw features to 42 (log transforms, interaction terms, squared terms, tempo bins) — stateless by design so it's safe to apply identically to train and test data.
4. **Model selection**: compared LightGBM, Random Forest, Logistic Regression, k-NN, SVM and MLP via 3-fold cross-validation on a 10,000-row stratified subsample (to keep tuning fast), then selected LightGBM and tuned it with Optuna (50 trials, TPE sampler).
5. **Final training** on the combined train+validation set, evaluated once on the held-out test set — accuracy alone was deliberately not the optimisation target; macro ROC-AUC and per-class F1 were tracked throughout because they're less misleading under class imbalance.

## Project Structure

```
Music_Mood_Classifier/
├── data/
│   ├── raw/                       # dataset.csv (Spotify tracks, not tracked in git)
│   └── README.md                  # Dataset source and column reference
├── notebooks/                     # 5 notebooks: EDA → features → training → evaluation → UI demo
├── outputs/
│   └── figures/                   # 7 saved charts (EDA, confusion matrix, ROC curves)
├── models/                        # Trained model artefacts (final_model.pkl gitignored — regenerate via notebook 03)
├── reports/                       # Auto-generated markdown reports (EDA summary, evaluation, improvement log)
├── src/
│   ├── data_loader.py             # CSV auto-discovery and schema inspection
│   ├── genre_mapping.py           # 114-genre → 6-class taxonomy with reasoning
│   ├── preprocessing.py           # Cleaning, encoding, scaling, splitting
│   ├── feature_engineering.py     # MusicFeatureEngineer (15 → 42 features)
│   ├── model_training.py          # CV comparison, Optuna tuning, final training
│   ├── evaluation.py              # Metrics, confusion matrix, ROC curves
│   ├── eda.py                     # Exploratory analysis and figure generation
│   ├── predict.py                 # Inference pipeline used by both UIs
│   └── utils.py                   # Shared paths, constants, logger
├── ui/                             # Technical demo: sliders, batch CSV scoring, raw probabilities
├── ui-mood/                        # Consumer demo: mood text input + playlist recommendations
├── tests/                          # pytest suite for data_loader, preprocessing, model
├── requirements.txt
└── README.md
```

## How To Run

```bash
git clone https://github.com/layaung-linnlett/Music_Mood_Classifier
cd Music_Mood_Classifier
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Place the Spotify dataset CSV in data/raw/ (see data/README.md for the source)

# Run the full pipeline
python -m src.eda
python -m src.preprocessing
python -m src.model_training
python -m src.evaluation

# Launch either demo app
streamlit run ui/app.py         # technical: feature sliders + batch CSV
streamlit run ui-mood/app.py    # consumer: mood text + playlist recommendations

# Run tests
pytest tests/ -v
```

The `notebooks/` folder walks through the same pipeline narratively, in order 01 → 05.

## Limitations & Future Work

- **Audio features alone cap accuracy around 70%.** Reaching further would need raw audio (MFCCs, spectral features) or lyrics — both are proposed and reasoned through in `reports/improvement_log.md`, but out of scope here since the point was to test how far metadata-only features can go.
- **The 6-class taxonomy is a modelling choice, not a ground truth.** Genre/mood is inherently fuzzy; a different set of merges would produce a different (not necessarily worse) accuracy number.
- **The playlist recommendations in `ui-mood/` are static search queries, not a live Spotify API integration** — they demonstrate the product concept without requiring API credentials to run.

## Contact

**GitHub**: [github.com/layaung-linnlett](https://github.com/layaung-linnlett) | **LinkedIn**: [linkedin.com/in/layaung-linnlett](https://www.linkedin.com/in/layaung-linnlett/)
