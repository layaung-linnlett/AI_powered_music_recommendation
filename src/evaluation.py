"""Metrics, classification report, and evaluation plots."""

# ==== Standard Library Imports ====
from pathlib import Path
from typing import List, Optional, Tuple

# ==== Third-Party Imports ====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

# ==== Internal Imports ====
from src.utils import (
    FIGURES_DIR,
    MIN_TARGET_ACCURACY,
    REPORTS_DIR,
    ensure_dirs,
    get_logger,
)

# ==== Module Logger ====
logger = get_logger(__name__)

# ==== Constants ====
FIGURE_DPI: int = 150
# Confusion matrix figure width per class (inches).
CM_WIDTH_PER_CLASS: float = 0.18
CM_HEIGHT_PER_CLASS: float = 0.18
CM_MIN_SIZE: float = 14.0
# Number of ROC curves to plot per figure.
ROC_CURVES_PER_FIGURE: int = 30
# ROC figure size.
ROC_FIGSIZE: Tuple[int, int] = (12, 8)


# ==== Helpers ====

def _save_fig(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib Figure to the figures directory.

    Args:
        fig: The Figure to save.
        filename: Filename including extension (e.g. 'confusion_matrix.png').
    """
    ensure_dirs()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path.name)


# ==== Evaluation Functions ====

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, float, float, str]:
    """Compute and print all evaluation metrics.

    Args:
        y_true: True integer-encoded labels.
        y_pred: Predicted integer-encoded labels.
        class_names: Optional list of class names for the report.

    Returns:
        A tuple of (accuracy, weighted_f1, macro_f1, report_str).
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    print("=" * 60)
    print("EVALUATION METRICS (TEST SET)")
    print("=" * 60)
    print(f"Overall Accuracy    : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted F1 Score   : {weighted_f1:.4f}")
    print(f"Macro F1 Score      : {macro_f1:.4f}")
    print()
    print("Per-Class Classification Report:")
    print(report_str)

    if accuracy >= MIN_TARGET_ACCURACY:
        print(f"Target accuracy of {MIN_TARGET_ACCURACY:.0%} ACHIEVED.")
    else:
        print(
            f"Target accuracy of {MIN_TARGET_ACCURACY:.0%} NOT reached "
            f"(gap: {MIN_TARGET_ACCURACY - accuracy:.4f}). "
            "See reports/improvement_log.md for remediation steps."
        )
    print("=" * 60)

    return accuracy, weighted_f1, macro_f1, report_str


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> None:
    """Save a confusion matrix heatmap.

    For datasets with many classes the matrix is plotted without individual
    cell annotations to remain readable.

    Args:
        y_true: True integer-encoded labels.
        y_pred: Predicted integer-encoded labels.
        class_names: Optional list of class names for axis labels.
    """
    n_classes = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred)

    fig_size = max(CM_MIN_SIZE, n_classes * CM_WIDTH_PER_CLASS * 5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    # Normalise so each row sums to 1 for readability.
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        ax=ax,
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
        cbar=True,
        linewidths=0.0,
        annot=False,
    )
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    ax.set_title("Normalised Confusion Matrix", fontsize=11)
    if class_names and n_classes <= 30:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fig.tight_layout()
    _save_fig(fig, "confusion_matrix.png")


def plot_roc_auc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Optional[float]:
    """Compute and save one-vs-rest ROC-AUC curves.

    Because there are many classes the curves are split across multiple figures,
    each containing ROC_CURVES_PER_FIGURE classes.

    Args:
        y_true: True integer-encoded labels.
        y_proba: Predicted probability matrix of shape (n_samples, n_classes).
        class_names: List of string class names.

    Returns:
        Macro-averaged ROC-AUC score, or None if y_proba is not available.
    """
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    n_classes = len(class_names)

    if y_proba is None or y_proba.shape[1] != n_classes:
        logger.warning("Probability scores not available; skipping ROC-AUC curves.")
        return None

    try:
        roc_auc = float(
            roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        )
    except ValueError as exc:
        logger.warning("Could not compute ROC-AUC: %s", exc)
        return None

    logger.info("Macro ROC-AUC (OVR): %.4f", roc_auc)

    # Split into multiple figures for readability.
    chunks = [
        class_names[i : i + ROC_CURVES_PER_FIGURE]
        for i in range(0, n_classes, ROC_CURVES_PER_FIGURE)
    ]

    from sklearn.metrics import roc_curve

    for fig_idx, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=ROC_FIGSIZE)
        for class_name in chunk:
            class_idx = class_names.index(class_name)
            fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_proba[:, class_idx])
            auc_val = roc_auc_score(y_bin[:, class_idx], y_proba[:, class_idx])
            ax.plot(fpr, tpr, lw=0.8, label=f"{class_name} (AUC={auc_val:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(
            f"ROC Curves (One-vs-Rest) - Figure {fig_idx + 1} of {len(chunks)}"
        )
        ax.legend(loc="lower right", fontsize=6, ncol=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        fig.tight_layout()
        _save_fig(fig, f"roc_auc_curves_{fig_idx + 1:02d}.png")

    return roc_auc


def save_evaluation_report(
    accuracy: float,
    weighted_f1: float,
    macro_f1: float,
    report_str: str,
    roc_auc: Optional[float],
) -> None:
    """Write evaluation metrics to a markdown report file.

    Args:
        accuracy: Overall test-set accuracy.
        weighted_f1: Weighted F1 score.
        macro_f1: Macro F1 score.
        report_str: Full per-class classification report string.
        roc_auc: Macro ROC-AUC score, or None if not available.
    """
    ensure_dirs()
    roc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A (probabilities unavailable)"
    achieved = accuracy >= MIN_TARGET_ACCURACY

    lines = [
        "# Test Set Evaluation Report",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overall Accuracy | {accuracy:.4f} ({accuracy * 100:.2f}%) |",
        f"| Weighted F1 | {weighted_f1:.4f} |",
        f"| Macro F1 | {macro_f1:.4f} |",
        f"| Macro ROC-AUC (OVR) | {roc_str} |",
        f"| 80% accuracy target met | {'Yes' if achieved else 'No'} |",
        "",
        "## Per-Class Classification Report",
        "",
        "```",
        report_str,
        "```",
        "",
        "## Figures",
        "",
        "- `outputs/figures/confusion_matrix.png`: Normalised confusion matrix.",
        "- `outputs/figures/roc_auc_curves_*.png`: One-vs-rest ROC curves.",
    ]

    output_path = REPORTS_DIR / "evaluation_report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Evaluation report written to: %s", output_path)


def write_improvement_log(
    accuracy: float,
    attempts: Optional[List[str]] = None,
) -> None:
    """Write an improvement log if the target accuracy was not achieved.

    Does nothing if `reports/improvement_log.md` already exists: that file is
    meant to hold a hand-written record of what was tried and why (see the
    shipped version), and re-running this module should not silently replace
    that analysis with a generic placeholder.

    Args:
        accuracy: Achieved test-set accuracy.
        attempts: Optional list of strings describing improvement attempts made.
    """
    if accuracy >= MIN_TARGET_ACCURACY:
        return

    output_path = REPORTS_DIR / "improvement_log.md"
    if output_path.exists():
        logger.info(
            "Improvement log already exists at %s -- leaving the existing "
            "analysis in place instead of overwriting it.",
            output_path,
        )
        return

    ensure_dirs()
    gap = MIN_TARGET_ACCURACY - accuracy
    lines = [
        "# Improvement Log",
        "",
        f"The initial model achieved a test-set accuracy of **{accuracy:.4f}** "
        f"({accuracy * 100:.2f}%), which is below the target of "
        f"{MIN_TARGET_ACCURACY:.0%} (gap: {gap:.4f}).",
        "",
        "## Improvement Attempts",
        "",
    ]

    if attempts:
        for attempt in attempts:
            lines.append(f"- {attempt}")
    else:
        lines.append("No improvement attempts recorded yet.")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Improvement log written to: %s", output_path)


# ==== Entry Point ====

if __name__ == "__main__":
    from src.data_loader import load_data, discover_target
    from src.preprocessing import (
        handle_missing_values,
        encode_features,
        detect_and_clip_outliers,
        prepare_features_target,
        encode_target,
        split_data,
        build_preprocessor_pipeline,
        fit_and_transform,
        load_pipeline,
    )
    from src.model_training import load_model

    raw_df = load_data()
    target_col = discover_target(raw_df)
    df_clean = handle_missing_values(raw_df)
    df_enc = encode_features(df_clean)
    df_clip = detect_and_clip_outliers(df_enc)
    X, y_raw = prepare_features_target(df_clip, target_col)
    y, le = encode_target(y_raw)
    _, _, X_test, _, _, y_test = split_data(X, y)
    pipeline = load_pipeline()
    X_test_sc = pipeline.transform(X_test)

    model = load_model()
    y_pred = model.predict(X_test_sc)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_sc)

    accuracy, weighted_f1, macro_f1, report_str = compute_metrics(
        y_test, y_pred, class_names=le.classes_.tolist()
    )
    plot_confusion_matrix(y_test, y_pred, class_names=le.classes_.tolist())
    roc_auc = plot_roc_auc_curves(y_test, y_proba, class_names=le.classes_.tolist())
    save_evaluation_report(accuracy, weighted_f1, macro_f1, report_str, roc_auc)
    write_improvement_log(accuracy)
