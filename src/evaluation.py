"""
Shared metric computation and plotting utilities used by both the
classification (src/classification.py) and dose-response regression
(src/dose_response.py) modules, so every experiment reports results in a
consistent, comparable format.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def evaluate_classification(y_true, y_pred, y_proba, pos_label):
    y_true_bin = (y_true == pos_label).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "roc_auc": roc_auc_score(y_true_bin, y_proba),
        "pr_auc": average_precision_score(y_true_bin, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(model_results, pos_label, title, path):
    """model_results: dict of {model_name: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y_true, y_proba) in model_results.items():
        y_true_bin = (y_true == pos_label).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_proba)
        auc = roc_auc_score(y_true_bin, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(model_results, pos_label, title, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y_true, y_proba) in model_results.items():
        y_true_bin = (y_true == pos_label).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_bin, y_proba)
        ap = average_precision_score(y_true_bin, y_proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(model_results, pos_label, title, path, n_bins=5):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y_true, y_proba) in model_results.items():
        y_true_bin = (y_true == pos_label).astype(int)
        frac_pos, mean_pred = calibration_curve(y_true_bin, y_proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison_bar(df, metric_cols, group_col, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.8 / len(metric_cols)
    for i, m in enumerate(metric_cols):
        ax.bar(x + i * width, df[m], width, label=m)
    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(df[group_col], rotation=20, ha="right")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances, names, title, path, top_n=20):
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(order))))
    ax.barh(np.array(names)[order][::-1], np.array(importances)[order][::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def evaluate_regression(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pear_r, _ = pearsonr(y_true, y_pred)
    spear_r, _ = spearmanr(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "R2": r2_score(y_true, y_pred),
        "pearson_r": pear_r,
        "spearman_r": spear_r,
    }


def plot_predicted_vs_actual(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "k--", alpha=0.6, label="y = x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title, path):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, s=15)
    ax.axhline(0, color="k", linestyle="--", alpha=0.6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
