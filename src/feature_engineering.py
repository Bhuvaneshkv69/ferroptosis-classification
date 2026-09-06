"""
Feature engineering for the TCGA COADREAD classification task, and the
complementary (literature-style) ferroptosis survival risk-score experiment.

Two feature sets are produced so that "raw genes" vs "engineered/signature
features" can be honestly compared (Experiment 2 vs Experiment 3):

  A. RAW    -- the 35 individual ferroptosis gene expression values (log2).
  B. ENGINEERED -- a small set of derived, biologically-motivated summaries:
       - ferroptosis_score_mean : mean of per-gene z-scores (unweighted
         signature score, the simplest and most common form used in the
         literature reviewed in docs/DATASET.md)
       - ferroptosis_score_pc1  : first principal component of the
         standardized gene matrix (captures the dominant co-expression axis)
       - a handful of clinically available covariates (AGE) where present

A COMPLEMENTARY, exploratory experiment mirrors the published CRC
ferroptosis-signature literature (Cox regression on overall survival, then
median-split into "High risk"/"Low risk"). This risk label is a linear
function of the same expression features used to predict it, so predicting
it back from those features is partly circular by construction -- exactly the
same methodological property as the four published signatures reviewed in
docs/DATASET.md. It is kept SEPARATE from the primary, non-circular MSI-status
classification target and is always reported with this caveat attached
(see docs/LIMITATIONS.md).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"


def add_engineered_features(df: pd.DataFrame, gene_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    z = StandardScaler().fit_transform(df[gene_cols])
    z_df = pd.DataFrame(z, columns=[f"{g}_z" for g in gene_cols], index=df.index)

    df["ferroptosis_score_mean"] = z_df.mean(axis=1)

    pca = PCA(n_components=1, random_state=42)
    df["ferroptosis_score_pc1"] = pca.fit_transform(z_df).ravel()
    explained_var = float(pca.explained_variance_ratio_[0])

    return df, explained_var


def build_ferroptosis_risk_score(df: pd.DataFrame, gene_cols: list[str]):
    """
    Univariate Cox screen (p < 0.10) followed by a multivariate Cox model on
    the screened genes, mirroring the two-stage univariate-then-multivariate
    approach used by the four published CRC ferroptosis-signature papers
    (see docs/DATASET.md). Genes are z-scored so coefficients are comparable.
    Returns the risk score, the median-split High/Low label, and the fitted
    coefficients (for interpretability / the comparison table).
    """
    surv = df[["OS_MONTHS", "OS_EVENT"] + gene_cols].dropna()
    z = StandardScaler().fit_transform(surv[gene_cols])
    z_df = pd.DataFrame(z, columns=gene_cols, index=surv.index)
    cox_input = pd.concat([surv[["OS_MONTHS", "OS_EVENT"]], z_df], axis=1)

    univariate_p = {}
    for g in gene_cols:
        cph = CoxPHFitter()
        try:
            cph.fit(cox_input[["OS_MONTHS", "OS_EVENT", g]], duration_col="OS_MONTHS", event_col="OS_EVENT")
            univariate_p[g] = float(cph.summary.loc[g, "p"])
        except Exception:
            univariate_p[g] = 1.0

    screened = [g for g, p in univariate_p.items() if p < 0.10]
    if len(screened) < 2:
        screened = sorted(univariate_p, key=univariate_p.get)[:5]

    cph_multi = CoxPHFitter(penalizer=0.1)
    cph_multi.fit(cox_input[["OS_MONTHS", "OS_EVENT"] + screened], duration_col="OS_MONTHS", event_col="OS_EVENT")

    coefs = cph_multi.summary["coef"]
    risk_score = z_df[screened].mul(coefs, axis=1).sum(axis=1)
    threshold = risk_score.median()
    risk_label = np.where(risk_score >= threshold, "High risk", "Low risk")

    result = pd.Series(index=df.index, dtype="object")
    result.loc[surv.index] = risk_label
    score_series = pd.Series(np.nan, index=df.index)
    score_series.loc[surv.index] = risk_score.values

    coef_table = cph_multi.summary.reset_index().rename(columns={"index": "gene"})
    return result, score_series, {"univariate_p": univariate_p, "screened_genes": screened}, coef_table


def run(df: pd.DataFrame, gene_cols: list[str]):
    df_fe, pc1_explained_var = add_engineered_features(df, gene_cols)
    risk_label, risk_score, cox_meta, coef_table = build_ferroptosis_risk_score(df_fe, gene_cols)
    df_fe["ferroptosis_risk_group"] = risk_label
    df_fe["ferroptosis_risk_score"] = risk_score

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "tcga_coadread_features.csv"
    df_fe.to_csv(out_path, index=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    coef_table.to_csv(TABLES_DIR / "cox_risk_score_coefficients.csv", index=False)

    meta = {
        "pc1_explained_variance_ratio": pc1_explained_var,
        "cox_screened_genes": cox_meta["screened_genes"],
        "cox_univariate_p_values": cox_meta["univariate_p"],
        "risk_group_counts": pd.Series(risk_label).value_counts(dropna=True).to_dict(),
    }
    with open(PROCESSED_DIR / "feature_engineering_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Engineered features saved -> {out_path}")
    print(f"PC1 explained variance ratio: {pc1_explained_var:.3f}")
    print(f"Cox model screened {len(cox_meta['screened_genes'])} genes (p<0.10): {cox_meta['screened_genes']}")
    print(f"Risk group counts: {meta['risk_group_counts']}")
    return df_fe


if __name__ == "__main__":
    tcga = pd.read_csv(PROCESSED_DIR / "tcga_coadread_processed.csv")
    with open(PROCESSED_DIR / "tcga_preprocessing_metadata.json") as f:
        meta = json.load(f)
    gene_cols = meta["ferroptosis_genes_used"]
    run(tcga, gene_cols)
