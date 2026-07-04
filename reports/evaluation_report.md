# Test Set Evaluation Report

## Summary Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.6829 (68.29%) |
| Weighted F1 | 0.6761 |
| Macro F1 | 0.6361 |
| Macro ROC-AUC (OVR) | 0.9081 |
| 80% accuracy target met | No |

## Per-Class Classification Report

```
              precision    recall  f1-score   support

    acoustic       0.71      0.66      0.68      3150
 alternative       0.57      0.42      0.49      1800
       dance       0.68      0.81      0.74      6750
  electronic       0.72      0.72      0.72      2550
       heavy       0.67      0.59      0.63      1800
       vocal       0.74      0.45      0.56      1050

    accuracy                           0.68     17100
   macro avg       0.68      0.61      0.64     17100
weighted avg       0.68      0.68      0.68     17100

```

## Figures

- `outputs/figures/confusion_matrix.png`: Normalised confusion matrix.
- `outputs/figures/roc_auc_curves_*.png`: One-vs-rest ROC curves.