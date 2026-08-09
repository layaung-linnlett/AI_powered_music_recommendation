# Contributors

MoodTunes AI was built by a team of five students (Group 3) for UFCE3P-30-3
(Essentials and Applications of Artificial Intelligence), BSc Computer Science,
UWE Bristol. Submitted 21 April 2026.

Work was divided into five pipeline-ownership roles, with each member holding
clear primary responsibility for their stage. All members additionally shared
weekly meetings, peer code review, both presentations, and the final submission
check.

| Stage | Owner | Role | Contribution |
|---|---|---|---|
| Data | Htet Htet Wint | Data Engineer | Sourced the 114,000-track Spotify dataset, data loading and schema inspection, missing-value analysis, Logistic Regression baseline |
| Features | La Yaung Linn Lett | Feature Engineer | IQR Winsorisation and dataset cleaning, initial 114→4 genre taxonomy, `MusicFeatureEngineer` (15→42 features), Random Forest baseline |
| EDA | Zach | Data Analyst and Project Manager | Formed the group and acted as primary liaison, 12 visualisations, Pearson correlation, mutual information, trained the MLP model, ran weekly meetings |
| Training | Zulfiqar Khan | Lead ML Engineer | Extended the taxonomy 4→6 classes, identified and fixed the data leakage bug, KNN baseline, LightGBM + Optuna TPE tuning, ROC-AUC evaluation, initial Streamlit UI, repository documentation |
| Product | Hein Htet Phyo | AI Product Developer | Original project concept, XGBoost baseline, cosine-similarity top-5 song recommender, final Streamlit application, repository maintenance |

Shared across all members: Presentation 1 and 2, peer code review, weekly
meetings, the final report, and the final submission check.

## About this repository

This is a personal portfolio copy of the group project, published by
La Yaung Linn Lett. It reflects the integrated final state of the work rather
than the full development history, so the commit log is not a reliable guide to
who wrote what — the table above is.

My own contribution is the Features row. The remaining work was authored by the
teammates credited here and is included so the project can be understood and run
as a whole, not claimed as my own.

Team members are credited by the names and roles recorded in the group's
submitted reflective report. If you are a member of this team and would like
your name shown differently, amended, or removed, contact me and I will action
it.
