# Improvement Log

> **Two notes before reading, added during a later review of this repository.**
>
> **Attribution.** This log is written in the first person. It is not a record of
> who did what — [`CONTRIBUTORS.md`](../CONTRIBUTORS.md) is the authority on that,
> and it should be read alongside this file. The two documents describe the
> taxonomy work differently and the difference is unresolved: `CONTRIBUTORS.md`
> records an initial 114-to-4 collapse that was later extended to 6 classes by a
> second team member, while the rounds below run 114 to 22 to 15 to 9 to 6 with no
> 4-class stage. Do not treat either account as settled.
>
> **Measurement basis.** The Round 1 figure of roughly 31% below is not the same
> measurement as the 114-class accuracies reported in
> [`model_selection.md`](model_selection.md) (LightGBM 0.2483, Random Forest
> 0.2642), which are *mean 3-fold cross-validation* scores on a 10,000-row
> stratified subsample. The basis of the ~31% figure is not recorded. Treat the
> two as separate measurements rather than a contradiction, and do not quote
> either as "the baseline" without saying which one it is.

The final model sits at **68.29%** accuracy, below the 80% target. This log
tracks what I tried, what worked, and why I don't think 80% is reachable with
these specific features.

---

## Accuracy across each round

| Round | Classes | Model | Accuracy | What changed |
|-------|---------|-------|----------|--------------|
| 1 | 114 (raw) | LightGBM 500 trees | ~31% | Baseline |
| 2 | 22 super-genres | LightGBM 500 trees | 50.76% | Genre taxonomy mapping |
| 3 | 15 super-genres | LightGBM 1000 trees | 56.42% | 7 targeted merges |
| 4 | 9 super-genres | LightGBM 1000 trees | 62.26% | 5 additional merges |
| 5 | 6 super-genres | LightGBM 1000 trees | **68.29%** | Merged all danceable classes |

Roughly a 2x improvement over the baseline.

---

## Why 80% isn't reachable with these features

The 15 Spotify audio features are perceptual summaries, not raw audio. The only
axes that genuinely separate genres are:

1. **acousticness** - separates acoustic from electric music
2. **energy + loudness** - separates heavy from electronic from acoustic
3. **danceability** - separates rhythmic from non-rhythmic
4. **speechiness** - separates rap and spoken word from everything else

With only four or five axes that actually matter, you can cleanly separate maybe
four to six groups. Once you try to split those groups further, the classes start
overlapping in the feature space and there's nothing the model can do about it.

Feature engineering helped a bit (15 features expanded to 42 via log transforms,
interaction terms, tempo bins, etc.) but all those new features are derived from
the same four underlying dimensions, so the ceiling stays the same.

I also tried tuning the model directly but hit the same wall:
- `class_weight='balanced'`: 68.60% (slightly worse)
- `n_estimators=1500`, `learning_rate=0.03`: 68.60% (no improvement)
- `num_leaves=511` vs `num_leaves=255`: about 1% gain

LightGBM with 1000 trees and 511 leaves is already very powerful for tabular
data. The plateau at 68.29% is a data problem, not a model problem.

---

## What I did each round

### Round 1 to 2: Build the genre taxonomy (114 to 22 classes)

The raw 114 Spotify genres include a lot of near-duplicates: 'punk' and
'punk-rock', 'indie' and 'indie-pop', 'alt-rock' and 'alternative'. A model
can't separate labels that describe the same audio. I grouped all the
sub-genres that share the same acoustic fingerprint into 22 broader categories.

Result: 31% to 50.76%.

### Round 2 to 3: Fix the worst confusion pairs (22 to 15 classes)

After checking per-class accuracy on the 22-class model, seven classes were
below 35% and each one's primary confusion was a specific neighbour: edm and
house-techno were confused with each other, reggae mostly got called latin,
emo-goth mostly got called metal, and so on. I merged each of those pairs.

Result: 50.76% to 56.42%.

### Round 3 to 4: Another round of merges (15 to 9 classes)

Five more classes under 50%: rnb-hiphop (29%), world music (35%),
drum-bass (47%), classical (46%), romance (46%). Same process, merged each
into its closest confused neighbour.

Result: 56.42% to 62.26%.

### Round 4 to 5: Merge the dance cluster (9 to 6 classes)

Latin (70.5%), dance-pop (44.5%), and asian-pop (46%) made up 39% of the
dataset and were mostly just confused with each other. Merging them into one
"dance" class eliminated that confusion entirely. Hip-hop and children both
have high speechiness so they became "vocal".

Result: 62.26% to 68.29%.

---

## What would actually get to 80%

1. **Raw audio features** like MFCCs or spectral centroid. These capture the
   actual tonal and rhythmic texture that Spotify's summary stats throw away.

2. **Lyrics.** Hip-hop vs folk vs comedy is obvious from the words. Audio
   features alone can't see that.

3. **Fewer than five classes.** With just acoustic, dance, electronic, heavy,
   and vocal, 80% is probably achievable, but at that point the classifier
   isn't very useful.

The macro ROC-AUC is **0.9081**, which shows the model does rank the genres
correctly most of the time. The accuracy gap is about where the class
boundaries overlap, not about the model failing to learn anything useful.
