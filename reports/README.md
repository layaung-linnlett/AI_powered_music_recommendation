# Reports Directory

This directory contains all generated reports and figures from the analysis pipeline.

## Reports

| File | Contents |
|------|----------|
| `eda_summary.md` | Full exploratory data analysis: class distribution, missing values, feature statistics, mutual information ranking, and key observations. |
| `model_selection.md` | Cross-validation comparison table for all candidate models, selection rationale, GridSearchCV tuning results, and the best hyperparameters found. |
| `evaluation_report.md` | Final test-set evaluation: accuracy, weighted and macro F1, per-class classification report, and links to figure files. |
| `improvement_log.md` | Only present if test accuracy fell below 80%. Documents each improvement attempt, what was tried, and the outcome. |

## Figures (`../outputs/figures/`)

Figures live in `outputs/figures/` at the project root, not inside `reports/`,
so they follow the same convention as every other project in this portfolio.

| Figure | Description |
|--------|-------------|
| `class_distribution.png` | Bar chart showing the count of each of the 114 raw genre labels (pre-taxonomy-collapse). |
| `feature_distributions.png` | Histograms with KDE overlays for all 15 numeric audio features. |
| `correlation_heatmap.png` | Lower-triangular Pearson correlation matrix of all numeric features. |
| `pairplot_top_features.png` | Pairplot of the 4 features with the highest mutual information with the target, coloured by a random sample of 10 classes. |
| `missing_values.png` | Bar chart of missing-value counts per column (only generated if missing values exist). |
| `confusion_matrix.png` | Normalised confusion matrix on the held-out test set, for the final 6-class model. |
| `roc_auc_curves_01.png` | One-vs-rest ROC curves for the final 6 mood classes. |
