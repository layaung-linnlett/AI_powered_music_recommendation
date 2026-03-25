# Data Directory

## Overview

This directory contains the raw input dataset for the Music Mood Classifier project.
The raw data is **not committed to version control** (see `.gitignore`), but the
expected file location and format are documented here.

## Expected File Location

```
data/
raw/
    dataset.csv    # place the raw dataset here
```

## Dataset Description

| Property | Value |
|----------|-------|
| Format | CSV |
| Rows | 114,000 |
| Columns | 20 (including target) |
| Target column | `track_genre` |
| Unique classes | 114 |
| Class balance | Perfectly balanced (1,000 samples per class) |
| Source | Spotify track metadata and audio features |

## Column Reference

| Column | Type | Description |
|--------|------|-------------|
| track_id | string | Spotify track URI (dropped before modelling) |
| artists | string | Artist name(s) (dropped before modelling) |
| album_name | string | Album title (dropped before modelling) |
| track_name | string | Track title (dropped before modelling) |
| popularity | int | Spotify popularity score (0 to 100) |
| duration_ms | int | Track duration in milliseconds |
| explicit | bool | Whether the track has explicit content |
| danceability | float | How suitable for dancing (0.0 to 1.0) |
| energy | float | Perceptual measure of intensity (0.0 to 1.0) |
| key | int | Musical key (0=C, 1=C#, ..., 11=B) |
| loudness | float | Average loudness in dB (typically -60 to 0) |
| mode | int | Modality: 1=major, 0=minor |
| speechiness | float | Presence of spoken words (0.0 to 1.0) |
| acousticness | float | Confidence that track is acoustic (0.0 to 1.0) |
| instrumentalness | float | Predicts whether track has no vocals (0.0 to 1.0) |
| liveness | float | Probability of live performance (0.0 to 1.0) |
| valence | float | Musical positiveness (0.0 to 1.0) |
| tempo | float | Estimated beats per minute |
| time_signature | int | Estimated overall time signature (3 to 7) |
| track_genre | string | **Target**: music genre label |

## Missing Values

The raw file contains 1 missing value each in `artists`, `album_name`, and `track_name`.
These columns are dropped entirely before modelling, so no imputation is required.
All numeric feature columns have zero missing values.

## Notes

- Identifer and text columns (`track_id`, `artists`, `album_name`, `track_name`) are
  dropped during preprocessing.
- The `explicit` boolean is cast to integer (0 or 1) before scaling.
- Outliers in `duration_ms`, `loudness`, `speechiness`, `instrumentalness`, and
  `liveness` are clipped to the IQR fence (multiplier 3.0) before modelling.
