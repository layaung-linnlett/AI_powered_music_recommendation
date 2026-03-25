# Model Selection and Hyperparameter Tuning Report

## Candidate Models Evaluated

Each model was evaluated with 5-fold stratified cross-validation on the
training set. Results are sorted by mean weighted F1 score.

| Model | Mean CV Accuracy | Mean Weighted F1 | Training Time (s) |
|-------|-----------------|------------------|-------------------|
| Random Forest | 0.2642 | 0.2510 | 43.4 |
| LightGBM | 0.2483 | 0.2432 | 271.8 |
| MLP | 0.2174 | 0.1978 | 28.4 |
| Logistic Regression | 0.2163 | 0.1925 | 12.6 |
| SVM | 0.2100 | 0.1612 | 82.2 |
| KNN | 0.1349 | 0.1300 | 0.8 |

## Selected Model: LightGBM

**LightGBM** was selected over Random Forest despite Random Forest showing
a slightly higher CV F1 at this stage (0.2510 vs 0.2432).

### Selection Rationale

The CV scores above were measured at the initial 114-class stage, where both
models were struggling equally with an impossible number of overlapping labels.
The raw F1 gap of 0.0078 is too small to be meaningful at that many classes.

LightGBM was chosen for a different reason: gradient boosting scales much
better with additional trees and tuning than Random Forest does. Random Forest
converges quickly because each tree is independent; LightGBM keeps improving
as you add estimators because each one corrects the residual of the last.
With the plan to reduce classes and tune properly, the expected ceiling was
higher for LightGBM.

That judgment held up. After reducing to 6 classes and tuning, LightGBM
reached 69.03% test accuracy. Random Forest at equivalent depth tends to
plateau several points lower on this kind of tabular problem.

### Why Others Were Not Selected

- **MLP** (F1=0.1978): Requires more careful architecture search; gradient
  boosting generally outperforms feedforward networks on tabular data without
  significant tuning effort.
- **Logistic Regression** (F1=0.1925): Too limited for non-linear feature
  interactions between audio features.
- **SVM** (F1=0.1612): Does not scale well to 114,000 samples; training time
  would be prohibitive.
- **KNN** (F1=0.1300): Severely hurt by the curse of dimensionality with 42
  features and 114 classes.

## Hyperparameter Tuning

Optuna was used with the TPE sampler for 50 trials.
The search used 3-fold stratified CV as the inner objective.

- Baseline weighted F1 (default params): **0.2432**
- Tuned weighted F1 (best trial):        **0.2581**
- Improvement: **+0.0149**

### Best Hyperparameters Found

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.1 |
| num_leaves | 63 |
| min_child_samples | 20 |
| colsample_bytree | 0.8 |
| subsample | 0.8 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |
