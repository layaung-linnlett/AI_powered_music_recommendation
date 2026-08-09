# MoodTunes AI — Music Mood Classifier

**Sorting 114,000 Spotify tracks into moods a model can actually tell apart.**

Spotify tags every track with one of 114 micro-genres, but those labels don't map onto how music actually *feels* — and a classifier trained directly on all 114 barely beats guessing. This project uses only Spotify's 15 audio features (danceability, energy, acousticness, tempo, speechiness — no lyrics, no metadata) and treats the genre taxonomy itself as something to be engineered, collapsing 114 raw labels into 6 audio-distinguishable mood categories: acoustic, alternative, dance, electronic, heavy, vocal.

Built as a final-year group project on the AI module (UFCE3P-30-3) at UWE Bristol.

---

## Project Context and My Role

This was a **five-person group project**, submitted April 2026. This repository is my personal portfolio copy — it contains the team's integrated work, published with credit to the people who built each part.

**My role was Feature Engineer.** I owned:

- **Data cleaning and preparation** — IQR-based Winsorisation (multiplier 3.0), boolean encoding, duplicate removal
- **The initial genre taxonomy** — the first collapse of 114 Spotify labels into 4 broad classes, which formed the structural basis for the final 6-class taxonomy that a teammate later extended
- **`MusicFeatureEngineer`** — a custom stateless scikit-learn transformer expanding 15 raw audio features to 42 through log transforms, interaction terms, squared terms and tempo bins
- **The Random Forest baseline**, one of six models in the cross-validation comparison

**What I did not build.** The 4→6 taxonomy extension, the data leakage fix, the LightGBM model and Optuna tuning, the ROC-AUC evaluation, the exploratory analysis, the cosine-similarity recommender, and the Streamlit applications were built by other members of the team. Full credits, by name and pipeline stage, are in [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

**What my contribution actually bought.** LightGBM feature-importance analysis confirmed engineered features among the top 25 most important — including the interaction terms `dance_x_energy`, `energy_x_not_acoustic` and `speech_x_not_acoustic`. And the taxonomy work mattered more than any modelling choice: reframing the target from 114 near-identical labels into separable classes was the single biggest driver of performance in this project. Tuning, by comparison, moved the needle very little.

---

## Results

| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.9081 |
| Test accuracy | 68.29% |
| Weighted F1 | 0.6761 |
| Test set | 17,100 tracks (15%, stratified) |

The gap between accuracy and ROC-AUC is the interesting part. ROC-AUC is threshold-independent and measures ranking quality: the model ranks the true class above the false ones about 90% of the time, even where it commits to the wrong final label. With `dance` making up about 40% of the test set after taxonomy mapping and `vocal` only 6%, accuracy alone would have been a misleading headline — so macro ROC-AUC was used as the primary metric.

**Per-class recall** shows where the boundaries blur: `dance` (0.81) and `electronic` (0.72) separate cleanly, while `alternative` (0.42, F1 0.49) and `vocal` (0.45) overlap heavily with their neighbours. `alternative` shares energy characteristics with both acoustic and dance; `vocal` overlaps with dance on speechiness. `vocal` is the interesting case — precision is high (0.74) but recall is low, so when the model calls something `vocal` it is usually right, it just misses more than half of them.

**Just under 70% is the ceiling here, and that's a finding rather than a shortfall.** Spotify's 15 features contain only 4–5 genuine axes of variation (acousticness, energy/loudness, danceability, speechiness). Every engineered feature is a derivative of those same axes, so no amount of feature work moves the ceiling. Getting past it would need raw audio (MFCCs, spectral features) or lyrics.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| pandas, numpy | Data loading and manipulation |
| scikit-learn | Preprocessing pipeline, baseline models, evaluation metrics |
| LightGBM | Final classifier |
| Optuna | Hyperparameter tuning (TPE sampler) |
| matplotlib, seaborn | Static charts |
| Streamlit | Interactive demo applications |
| pytest | Test suite (41 tests) |

*No resampling was applied. The raw 114-genre dataset is perfectly balanced (1,000 tracks per genre), but collapsing it into 6 mood classes leaves an uneven spread — `dance` ends up at roughly 40% of tracks and `vocal` at 6%. `class_weight='balanced'` was tested and didn't improve results, so the model trains on the natural distribution.*

---

## Methodology

1. **EDA** across 114,000 tracks and 114 raw genre labels — class balance, missing values, feature distributions, Pearson correlation and mutual information against the target.
2. **Data cleaning** — IQR Winsorisation at multiplier 3.0, boolean encoding, duplicate removal.
3. **Iterative taxonomy design** — a baseline on the raw 114 labels reached roughly 31% accuracy. Per-class accuracy and confusion-matrix analysis identified labels the model could never separate, merging them across four rounds. An initial 4-class taxonomy proved insufficiently discriminative at the acoustic–vocal boundary and was extended to the final 6.
4. **Feature engineering** — `MusicFeatureEngineer` expands 15 raw features to 42. Stateless by design, so it applies identically to train and test data with no leakage risk.
5. **Model selection** — LightGBM, Random Forest, Logistic Regression, k-NN, LinearSVC and MLP compared via 3-fold stratified cross-validation on a 10,000-row subsample. LightGBM selected and tuned with Optuna TPE.
6. **Final training** on train+validation, evaluated once on the held-out test set, with macro ROC-AUC and per-class F1 tracked throughout.

**A data leakage bug was caught in peer code review** — `StandardScaler` was being fitted before the train/test split, which would have inflated every reported metric. It was found and fixed by our Lead ML Engineer, not by me, but it's worth recording: it's the clearest illustration in this project that a review process caught something no individual working alone did.

---

## Project Structure

```
Music_Mood_Classifier/
├── data/
│   ├── raw/                    # dataset.csv (not tracked in git)
│   └── README.md               # Dataset source and column reference
├── notebooks/                  # 01 EDA → 02 features → 03 training → 04 evaluation → 05 UI demo
├── src/
│   ├── data_loader.py          # CSV auto-discovery and schema inspection
│   ├── genre_mapping.py        # 114-genre → 6-class taxonomy, with reasoning
│   ├── preprocessing.py        # Cleaning, encoding, scaling, splitting
│   ├── feature_engineering.py  # MusicFeatureEngineer (15 → 42 features)
│   ├── model_training.py       # CV comparison, Optuna tuning, final training
│   ├── evaluation.py           # Metrics, confusion matrix, ROC curves
│   ├── eda.py                  # Exploratory analysis and figure generation
│   ├── predict.py              # Inference pipeline shared by both UIs
│   └── utils.py                # Paths, constants, logger
├── models/                     # Trained artefacts (final_model.pkl gitignored)
├── outputs/figures/            # Saved charts (EDA, confusion matrix, ROC curves)
├── reports/                    # eda_summary, model_selection, evaluation_report, improvement_log
├── tests/                      # pytest suite
├── ui/                         # Technical demo: feature sliders, batch CSV scoring
├── ui-mood/                    # Consumer demo: mood input + song recommendations
├── CONTRIBUTORS.md             # Team credits
├── requirements.txt
└── README.md
```

---

## How To Run

**Requires Python 3.12 or later** — the pinned `numpy` and `scipy` versions don't publish wheels for anything older.

```bash
git clone https://github.com/layaung-linnlett/Music_Mood_Classifier
cd Music_Mood_Classifier

python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

On macOS, LightGBM also needs OpenMP, which isn't bundled with the wheel: `brew install libomp`.

Place the Spotify Tracks Dataset CSV in `data/raw/` — see `data/README.md` for the source and expected schema.

```bash
# Run the pipeline in order
python -m src.eda              # figures → outputs/figures/
python -m src.preprocessing    # fits and saves models/preprocessor.pkl
python -m src.model_training   # CV comparison, tuning → models/final_model.pkl
python -m src.evaluation       # metrics → reports/evaluation_report.md

# Launch either demo app
streamlit run ui/app.py
streamlit run ui-mood/app.py

# Run tests
pytest tests/ -v
```

The test suite exercises the real pipeline, so tests that need the dataset or a trained model skip with an explanatory reason until both are present. On a fresh clone you'll see 5 passed and 36 skipped; once the dataset is in place and `src.model_training` has run, all 41 execute.

---

## Limitations

- **Audio features cap accuracy around 70%.** Lyrics (e.g. BERT embeddings) or raw audio would be needed to go further.
- **The 6-class taxonomy is a modelling choice, not ground truth.** Genre and mood are fuzzy; a different set of merges would give a different, not necessarily worse, result.
- **Class imbalance after mapping** — `dance` dominates, which is why weighted F1 sits well below ROC-AUC.
- **No user feedback loop** — recommendation quality can't improve without interaction data.

---

## Acknowledgements

Dataset: Spotify Tracks Dataset (Pandya, Kaggle). Built for UFCE3P-30-3, Essentials and Applications of Artificial Intelligence, UWE Bristol — module leaders Dr Mahmoud Elbattah and Dr Sondess Missaoui.

**Contact:** [GitHub](https://github.com/layaung-linnlett) · [LinkedIn](https://www.linkedin.com/in/layaung-linnlett/)
