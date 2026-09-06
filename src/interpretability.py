"""
Model interpretability for both the classification and dose-response modules.

Methods: permutation importance (model-agnostic) and SHAP (TreeExplainer, for
tree-based models). Random Forest models are refit here with fixed, reasonable
hyperparameters specifically for interpretability (independent of the
hyperparameter-tuned models scored in classification.py / dose_response.py),
since interpretability analyses are conventionally run on one representative
model rather than every tuned variant.

IMPORTANT CAVEAT (also in docs/LIMITATIONS.md and printed at the end of this
script): feature importance / SHAP values indicate that a gene's expression is
STATISTICALLY ASSOCIATED with the model's prediction. They are NOT evidence
that the gene CAUSES the biological outcome (MSI status, drug sensitivity, or
survival risk). Establishing causality would require independent experimental
(wet-lab) validation, which is outside the scope of this computational project.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import split_gdsc

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
RANDOM_SEED = 42

CAVEAT = (
    "Feature importance and SHAP values reflect ASSOCIATION with the model's "
    "prediction on this dataset, not proof of biological causality. See "
    "docs/LIMITATIONS.md."
)


def classification_interpretability(df, gene_cols):
    sub = df[df["msi_label"].notna()].reset_index(drop=True)
    X, y = sub[gene_cols], (sub["msi_label"] == "MSI-H").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=5, class_weight="balanced", random_state=RANDOM_SEED))])
    pipe.fit(X_train, y_train)

    perm = permutation_importance(pipe, X_test, y_test, n_repeats=30, random_state=RANDOM_SEED, scoring="roc_auc")
    perm_df = pd.DataFrame({"gene": gene_cols, "perm_importance_mean": perm.importances_mean,
                             "perm_importance_std": perm.importances_std}).sort_values(
        "perm_importance_mean", ascending=False)
    perm_df.to_csv(RESULTS_DIR / "feature_importance_classification_permutation.csv", index=False)

    X_test_scaled = pipe.named_steps["scaler"].transform(X_test)
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_test_scaled)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({"gene": gene_cols, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False)
    shap_df.to_csv(RESULTS_DIR / "feature_importance_classification_shap.csv", index=False)

    fig = plt.figure(figsize=(8, 8))
    shap.summary_plot(sv, X_test, feature_names=gene_cols, show=False)
    plt.title("SHAP summary - MSI classification (Random Forest)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary_classification.png", dpi=150)
    plt.close(fig)

    fig2 = plt.figure()
    plt.barh(perm_df["gene"].head(15)[::-1], perm_df["perm_importance_mean"].head(15)[::-1])
    plt.xlabel("Permutation importance (ROC-AUC drop)")
    plt.title("Top 15 ferroptosis genes - MSI classification")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_classification.png", dpi=150)
    plt.close(fig2)

    print("Top 10 genes by permutation importance (MSI classification):")
    print(perm_df.head(10).to_string(index=False))
    return perm_df, shap_df


def dose_response_interpretability(df, gene_cols):
    feature_cols = gene_cols  # ferroptosis genes only, for a gene-level SHAP story
    split = split_gdsc(df, feature_cols, "LN_IC50")
    X_train, y_train = split["X_train"], split["y_train"]
    X_test = split["X_test"]

    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(
        n_estimators=300, max_depth=6, random_state=RANDOM_SEED))])
    pipe.fit(X_train, y_train)

    perm = permutation_importance(pipe, X_test, split["y_test"], n_repeats=30,
                                   random_state=RANDOM_SEED, scoring="neg_mean_absolute_error")
    perm_df = pd.DataFrame({"gene": feature_cols, "perm_importance_mean": perm.importances_mean,
                             "perm_importance_std": perm.importances_std}).sort_values(
        "perm_importance_mean", ascending=False)
    perm_df.to_csv(RESULTS_DIR / "feature_importance_regression_permutation.csv", index=False)

    X_test_scaled = pipe.named_steps["scaler"].transform(X_test)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_test_scaled)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"gene": feature_cols, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False)
    shap_df.to_csv(RESULTS_DIR / "feature_importance_regression_shap.csv", index=False)

    fig = plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title("SHAP summary - LN_IC50 dose-response (Random Forest, ferroptosis genes only)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary_dose_response.png", dpi=150)
    plt.close(fig)

    fig2 = plt.figure()
    plt.barh(perm_df["gene"].head(15)[::-1], perm_df["perm_importance_mean"].head(15)[::-1])
    plt.xlabel("Permutation importance (MAE increase)")
    plt.title("Top 15 ferroptosis genes - LN_IC50 prediction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_regression.png", dpi=150)
    plt.close(fig2)

    print("\nTop 10 genes by permutation importance (LN_IC50 regression):")
    print(perm_df.head(10).to_string(index=False))
    return perm_df, shap_df


def main():
    tcga = pd.read_csv(PROCESSED_DIR / "tcga_coadread_features.csv")
    with open(PROCESSED_DIR / "tcga_preprocessing_metadata.json") as f:
        tcga_meta = json.load(f)
    gdsc = pd.read_csv(PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv")
    with open(PROCESSED_DIR / "gdsc_preprocessing_metadata.json") as f:
        gdsc_meta = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Classification interpretability (MSI status) ===")
    classification_interpretability(tcga, tcga_meta["ferroptosis_genes_used"])

    print("\n=== Dose-response interpretability (LN_IC50) ===")
    dose_response_interpretability(gdsc, gdsc_meta["ferroptosis_genes_used"])

    print(f"\n{CAVEAT}")


if __name__ == "__main__":
    main()
