"""
Whole-transcriptome baseline classifier for MSI status (Experiment 4),
directly comparable to the 35-gene ferroptosis-panel classifier
(Experiment 1, src/classification.py) -- answers "how much of the panel's
classification signal is ferroptosis-specific vs recoverable from an
unconstrained feature set?" (docs/FUTURE_WORK.md item 2).

Uses ~19,200 protein-coding genes (src/fetch_whole_transcriptome.py) for the
SAME TCGA COADREAD samples and the SAME MSI-H/MSS-MSI-L target as
Experiment 1. Because the training set (~470 samples) is far smaller than
the feature count (~19,200 genes), PCA is used for dimensionality
reduction -- fit strictly INSIDE the sklearn Pipeline on the training fold
only (never on the full dataset, never using the target), so there is no
supervised feature-selection leakage. This mirrors, and is the honest
analogue of, the same PCA idea already used for the (2-feature) engineered
ferroptosis signature score in Experiment 2.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import evaluation as ev
from config import RANDOM_SEED

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "tcga"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

N_PCA_COMPONENTS = 50
EXPERIMENT_NAME = "exp4_msi_whole_transcriptome"


def _model_grid():
    def pipe(clf):
        # ~12% of genes have sparse missingness (a real cBioPortal RNA-seq V2
        # profile coverage gap across this many genes, not a bug -- see
        # docs/DATASET.md). Median imputation is fit on the training fold
        # only (it is a Pipeline step), so no test-fold value ever
        # influences a training-time imputation value.
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_SEED)),
            ("clf", clf),
        ])

    return {
        "LogisticRegression": (
            pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            pipe(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED)),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5, None]},
        ),
        "SVM": (
            pipe(SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED)),
            {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["linear", "rbf"]},
        ),
        "XGBoost": (
            pipe(XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=1)),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.05, 0.1]},
        ),
    }


def build_whole_transcriptome_matrix() -> pd.DataFrame:
    long = pd.read_csv(RAW_DIR / "whole_transcriptome_expression_long.csv")
    gene_list = pd.read_csv(RAW_DIR / "whole_transcriptome_gene_list.csv")[["entrezGeneId", "hugoGeneSymbol"]]
    long = long.merge(gene_list, on="entrezGeneId", how="left")
    wide = long.pivot_table(index="sampleId", columns="hugoGeneSymbol", values="value", aggfunc="first")
    wide.columns.name = None
    return wide.reset_index()


def main():
    print("Loading whole-transcriptome expression matrix ...")
    wt = build_whole_transcriptome_matrix()
    gene_cols = [c for c in wt.columns if c != "sampleId"]
    print(f"  {wt.shape[0]} samples x {len(gene_cols)} genes")

    for g in gene_cols:
        wt[g] = np.log2(wt[g].clip(lower=0) + 1)

    tcga = pd.read_csv(PROCESSED_DIR / "tcga_coadread_processed.csv")[["sampleId", "msi_label"]]
    df = tcga.merge(wt, on="sampleId", how="inner")
    print(f"After merging with MSI labels: {len(df)} samples")

    # Target-blind (unsupervised) zero-variance filter only -- does not use
    # msi_label at all, so this cannot leak label information, unlike a
    # supervised top-K-by-association filter would.
    variances = df[gene_cols].var()
    keep_genes = variances[variances > 0].index.tolist()
    dropped = len(gene_cols) - len(keep_genes)
    print(f"Dropped {dropped} zero-variance genes (target-blind); {len(keep_genes)} genes remain")

    sub = df[df["msi_label"].notna()].reset_index(drop=True)
    X, y = sub[keep_genes], sub["msi_label"].astype(str)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    rows = []
    roc_inputs = {}
    for name, (pipe, params) in _model_grid().items():
        print(f"  fitting {name} ...")
        gs = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_trainval, (y_trainval == "MSI-H").astype(int))
        best = gs.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, list(best.classes_).index(1)]
        y_pred_label = np.where(y_pred == 1, "MSI-H", "MSS/MSI-L")

        metrics = ev.evaluate_classification(y_test, y_pred_label, y_proba, "MSI-H")
        metrics.update({
            "experiment": EXPERIMENT_NAME, "model": name, "best_params": json.dumps(gs.best_params_),
            "n_genes_used": len(keep_genes), "n_pca_components": N_PCA_COMPONENTS,
            "n_train": len(X_trainval), "n_test": len(X_test),
        })
        rows.append(metrics)
        roc_inputs[name] = (y_test, y_proba)
        print(f"    {name}: ROC-AUC={metrics['roc_auc']:.3f}")

    ev.plot_roc_curves(roc_inputs, "MSI-H", "ROC - Whole-transcriptome baseline (Experiment 4)",
                        FIG_DIR / "roc_curve_exp4_whole_transcriptome.png")

    results_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "whole_transcriptome_baseline_results.csv", index=False)

    # Direct comparison table: ferroptosis panel (Experiment 1) vs whole
    # transcriptome (Experiment 4), same target, same TCGA cohort.
    exp1 = pd.read_csv(RESULTS_DIR / "classification_results.csv")
    exp1 = exp1[exp1["experiment"] == "exp1_msi_raw_genes"][["model", "roc_auc", "accuracy", "f1"]]
    exp1 = exp1.rename(columns={"roc_auc": "roc_auc_ferroptosis_panel_35_genes",
                                  "accuracy": "accuracy_ferroptosis_panel", "f1": "f1_ferroptosis_panel"})
    exp4 = results_df[["model", "roc_auc", "accuracy", "f1"]].rename(
        columns={"roc_auc": "roc_auc_whole_transcriptome_pca50", "accuracy": "accuracy_whole_transcriptome",
                 "f1": "f1_whole_transcriptome"})
    comparison = exp1.merge(exp4, on="model")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comparison.round(3).to_csv(TABLES_DIR / "ferroptosis_vs_whole_transcriptome_comparison.csv", index=False)

    print("\n=== Ferroptosis panel (35 genes) vs whole-transcriptome baseline (~19,200 genes, PCA-50) ===")
    print(comparison.round(3).to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR / 'whole_transcriptome_baseline_results.csv'}, "
          f"{TABLES_DIR / 'ferroptosis_vs_whole_transcriptome_comparison.csv'}")
    return results_df, comparison


if __name__ == "__main__":
    main()
