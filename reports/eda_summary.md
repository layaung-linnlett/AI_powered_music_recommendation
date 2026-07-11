# EDA Summary: Music Mood Classifier

## Dataset Overview

| Property | Value |
|----------|-------|
| Rows | 114,000 |
| Columns | 20 |
| Target column | `track_genre` |
| Unique classes | 114 |
| Missing values | 3 |

## Target Variable Discovery

The column `track_genre` was selected as the target because:

- The column name contains the keyword **genre**, which directly signals
  a classification target in a music dataset.
- The dtype is `object`, consistent with categorical labels.
- Cardinality is 114 unique values, well within the range expected
  for a music genre classification task.
- Every class contains at least 1000 samples.

## Discovered Mood / Genre Categories

The dataset contains **114 genre labels**.

| Label | Count | Percentage |
|-------|-------|------------|
| acoustic | 1,000 | 0.88% |
| afrobeat | 1,000 | 0.88% |
| alt-rock | 1,000 | 0.88% |
| alternative | 1,000 | 0.88% |
| ambient | 1,000 | 0.88% |
| anime | 1,000 | 0.88% |
| black-metal | 1,000 | 0.88% |
| bluegrass | 1,000 | 0.88% |
| blues | 1,000 | 0.88% |
| brazil | 1,000 | 0.88% |
| breakbeat | 1,000 | 0.88% |
| british | 1,000 | 0.88% |
| cantopop | 1,000 | 0.88% |
| chicago-house | 1,000 | 0.88% |
| children | 1,000 | 0.88% |
| chill | 1,000 | 0.88% |
| classical | 1,000 | 0.88% |
| club | 1,000 | 0.88% |
| comedy | 1,000 | 0.88% |
| country | 1,000 | 0.88% |
| dance | 1,000 | 0.88% |
| dancehall | 1,000 | 0.88% |
| death-metal | 1,000 | 0.88% |
| deep-house | 1,000 | 0.88% |
| detroit-techno | 1,000 | 0.88% |
| disco | 1,000 | 0.88% |
| disney | 1,000 | 0.88% |
| drum-and-bass | 1,000 | 0.88% |
| dub | 1,000 | 0.88% |
| dubstep | 1,000 | 0.88% |
| edm | 1,000 | 0.88% |
| electro | 1,000 | 0.88% |
| electronic | 1,000 | 0.88% |
| emo | 1,000 | 0.88% |
| folk | 1,000 | 0.88% |
| forro | 1,000 | 0.88% |
| french | 1,000 | 0.88% |
| funk | 1,000 | 0.88% |
| garage | 1,000 | 0.88% |
| german | 1,000 | 0.88% |
| gospel | 1,000 | 0.88% |
| goth | 1,000 | 0.88% |
| grindcore | 1,000 | 0.88% |
| groove | 1,000 | 0.88% |
| grunge | 1,000 | 0.88% |
| guitar | 1,000 | 0.88% |
| happy | 1,000 | 0.88% |
| hard-rock | 1,000 | 0.88% |
| hardcore | 1,000 | 0.88% |
| hardstyle | 1,000 | 0.88% |
| heavy-metal | 1,000 | 0.88% |
| hip-hop | 1,000 | 0.88% |
| honky-tonk | 1,000 | 0.88% |
| house | 1,000 | 0.88% |
| idm | 1,000 | 0.88% |
| indian | 1,000 | 0.88% |
| indie-pop | 1,000 | 0.88% |
| indie | 1,000 | 0.88% |
| industrial | 1,000 | 0.88% |
| iranian | 1,000 | 0.88% |
| j-dance | 1,000 | 0.88% |
| j-idol | 1,000 | 0.88% |
| j-pop | 1,000 | 0.88% |
| j-rock | 1,000 | 0.88% |
| jazz | 1,000 | 0.88% |
| k-pop | 1,000 | 0.88% |
| kids | 1,000 | 0.88% |
| latin | 1,000 | 0.88% |
| latino | 1,000 | 0.88% |
| malay | 1,000 | 0.88% |
| mandopop | 1,000 | 0.88% |
| metal | 1,000 | 0.88% |
| metalcore | 1,000 | 0.88% |
| minimal-techno | 1,000 | 0.88% |
| mpb | 1,000 | 0.88% |
| new-age | 1,000 | 0.88% |
| opera | 1,000 | 0.88% |
| pagode | 1,000 | 0.88% |
| party | 1,000 | 0.88% |
| piano | 1,000 | 0.88% |
| pop-film | 1,000 | 0.88% |
| pop | 1,000 | 0.88% |
| power-pop | 1,000 | 0.88% |
| progressive-house | 1,000 | 0.88% |
| psych-rock | 1,000 | 0.88% |
| punk-rock | 1,000 | 0.88% |
| punk | 1,000 | 0.88% |
| r-n-b | 1,000 | 0.88% |
| reggae | 1,000 | 0.88% |
| reggaeton | 1,000 | 0.88% |
| rock-n-roll | 1,000 | 0.88% |
| rock | 1,000 | 0.88% |
| rockabilly | 1,000 | 0.88% |
| romance | 1,000 | 0.88% |
| sad | 1,000 | 0.88% |
| salsa | 1,000 | 0.88% |
| samba | 1,000 | 0.88% |
| sertanejo | 1,000 | 0.88% |
| show-tunes | 1,000 | 0.88% |
| singer-songwriter | 1,000 | 0.88% |
| ska | 1,000 | 0.88% |
| sleep | 1,000 | 0.88% |
| songwriter | 1,000 | 0.88% |
| soul | 1,000 | 0.88% |
| spanish | 1,000 | 0.88% |
| study | 1,000 | 0.88% |
| swedish | 1,000 | 0.88% |
| synth-pop | 1,000 | 0.88% |
| tango | 1,000 | 0.88% |
| techno | 1,000 | 0.88% |
| trance | 1,000 | 0.88% |
| trip-hop | 1,000 | 0.88% |
| turkish | 1,000 | 0.88% |
| world-music | 1,000 | 0.88% |

## Class Balance Assessment

The dataset is **perfectly balanced** (all classes have the same number of samples).

- Minimum samples per class: 1,000
- Maximum samples per class: 1,000

Because the dataset is balanced, no resampling (SMOTE or class weighting) was required.

## Missing Values

| Column | Missing Count |
|--------|---------------|
| artists | 1 |
| album_name | 1 |
| track_name | 1 |

## Feature Statistics

| Feature | Mean | Std | Min | Max | Skew | Kurtosis |
|---------|------|-----|-----|-----|------|----------|
| popularity | 33.239 | 22.305 | 0.000 | 100.000 | 0.046 | -0.928 |
| duration_ms | 228029.153 | 107297.713 | 0.000 | 5237295.000 | 11.195 | 354.952 |
| danceability | 0.567 | 0.174 | 0.000 | 0.985 | -0.399 | -0.185 |
| energy | 0.641 | 0.252 | 0.000 | 1.000 | -0.597 | -0.526 |
| key | 5.309 | 3.560 | 0.000 | 11.000 | -0.009 | -1.277 |
| loudness | -8.259 | 5.029 | -49.531 | 4.532 | -2.007 | 5.896 |
| mode | 0.638 | 0.481 | 0.000 | 1.000 | -0.572 | -1.673 |
| speechiness | 0.085 | 0.106 | 0.000 | 0.965 | 4.648 | 28.824 |
| acousticness | 0.315 | 0.333 | 0.000 | 0.996 | 0.727 | -0.950 |
| instrumentalness | 0.156 | 0.310 | 0.000 | 1.000 | 1.734 | 1.271 |
| liveness | 0.214 | 0.190 | 0.000 | 1.000 | 2.106 | 4.378 |
| valence | 0.474 | 0.259 | 0.000 | 0.995 | 0.115 | -1.027 |
| tempo | 122.148 | 29.978 | 0.000 | 243.372 | 0.232 | -0.109 |
| time_signature | 3.904 | 0.433 | 0.000 | 5.000 | -4.098 | 26.013 |

## Feature Importance (Mutual Information with Target)

Mutual information quantifies the dependency between each feature and the target.
Higher values indicate stronger association.

| Rank | Feature | Mutual Information Score |
|------|---------|--------------------------|
| 1 | popularity | 0.8879 |
| 2 | acousticness | 0.6292 |
| 3 | tempo | 0.5939 |
| 4 | loudness | 0.5392 |
| 5 | duration_ms | 0.5279 |
| 6 | energy | 0.4941 |
| 7 | instrumentalness | 0.4157 |
| 8 | danceability | 0.4103 |
| 9 | valence | 0.3593 |
| 10 | speechiness | 0.3219 |
| 11 | liveness | 0.2183 |
| 12 | time_signature | 0.0527 |
| 13 | key | 0.0371 |
| 14 | mode | 0.0365 |

## Key Observations

1. The dataset has 114 genre classes, each with exactly 1,000 samples, making it
   perfectly balanced. No resampling is needed.
2. The text columns (track_id, artists, album_name, track_name) are dropped before
   modelling as they are identifiers, not audio features.
3. `instrumentalness` and `speechiness` are heavily right-skewed, indicating most
   tracks are neither instrumental nor speech-heavy.
4. `acousticness` shows high variance and skew, reflecting the diversity of genres.
5. `popularity` ranges from 0 to 100 with a mean around 33, suggesting many obscure
   or niche tracks in the dataset.
6. `energy` and `loudness` are positively correlated, as expected for music.
7. `valence` (musical positiveness) is roughly uniformly distributed, covering
   both sad and happy sounding music.

## Figures

All figures are saved to `reports/figures/`:

- `class_distribution.png`: Bar chart of all 114 class counts.
- `feature_distributions.png`: Histograms with KDE for all numeric features.
- `correlation_heatmap.png`: Lower-triangular Pearson correlation matrix.
- `pairplot_top_features.png`: Pairplot of the top 4 features by mutual information.
- `missing_values.png`: Missing value counts (only generated if missing values exist).