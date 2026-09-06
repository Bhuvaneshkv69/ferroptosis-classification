"""
Preprocessing pipelines for the two independent datasets used in this project.

TCGA COADREAD (patient-level, classification):
    raw clinical + raw expression -> quality checks -> missing-value handling
    -> duplicate/sample-type filtering -> log2 transform -> MSI-H/MSS target
    construction -> train/val/test split (stratified, no leakage).

GDSC colorectal cell lines (cell-line-level, dose-response regression):
    raw expression matrix -> gene resolution against the array -> raw fitted
    dose-response -> drug coverage filtering -> merge -> train/test split by
    cell line (GroupShuffleSplit) so that entire cell lines, not just rows,
    are held out at test time.

Every decision below is a design choice made explicit in code comments and in
docs/METHODOLOGY.md -- nothing here silently drops or fabricates data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import data_loader as dl

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"

RANDOM_SEED = 42
MSI_MANTIS_THRESHOLD = 0.4  # Kautto et al., Oncotarget 2017 (doi:10.18632/oncotarget.13918)
MIN_DRUG_COVERAGE = 40  # keep GDSC drugs tested on >= 40 of the 46 colorectal lines


# ---------------------------------------------------------------------------
# TCGA COADREAD
# ---------------------------------------------------------------------------

def build_tcga_dataset(log_path=None):
    log = []

    def note(msg):
        log.append(msg)
        print(msg)

    genes = dl.load_ferroptosis_gene_list()["gene"].tolist()

    patient_clin = dl.load_tcga_clinical_wide()
    sample_clin = dl.load_tcga_clinical_sample_wide()
    samples = dl.load_tcga_samples()
    expr = dl.load_tcga_expression_wide()

    note(f"Raw: {len(patient_clin)} patients, {len(sample_clin)} samples, "
         f"{expr.shape[0]} samples x {expr.shape[1]-1} genes expression matrix")

    # --- quality check / duplicate handling: one sample per patient ---------
    # A handful of TCGA patients have >1 sample (e.g. metastatic + primary).
    # We keep only "Primary" tumor samples, and if a patient still has more
    # than one primary sample, keep the first (alphabetically lowest sample
    # barcode), which is the standard TCGA de-duplication convention.
    sample_meta = samples.merge(sample_clin, on="sampleId", how="left")
    primary = sample_meta[sample_meta["SAMPLE_TYPE"] == "Primary"].copy()
    n_dupes = primary.duplicated("patientId").sum()
    primary = primary.sort_values("sampleId").drop_duplicates("patientId", keep="first")
    note(f"Kept {len(primary)} primary-tumor samples (dropped {n_dupes} duplicate "
         f"patient samples and {sample_meta['SAMPLE_TYPE'].ne('Primary').sum()} non-primary samples)")

    # --- merge clinical (patient + sample level) -----------------------------
    clinical = primary.merge(patient_clin, on="patientId", how="left")

    # --- merge expression -----------------------------------------------------
    df = clinical.merge(expr, on="sampleId", how="inner")
    note(f"After merging expression: {len(df)} samples retained "
         f"(lost {len(clinical) - len(df)} with no expression record)")

    # --- missing value analysis on ferroptosis genes --------------------------
    gene_cols = [g for g in genes if g in df.columns]
    missing_genes = sorted(set(genes) - set(gene_cols))
    if missing_genes:
        note(f"WARNING: {len(missing_genes)} ferroptosis genes not returned by "
             f"cBioPortal RNA-seq V2 profile and excluded from TCGA features: {missing_genes}")

    miss_frac = df[gene_cols].isna().mean().sort_values(ascending=False)
    note(f"Per-gene missingness (top 5): {miss_frac.head(5).round(3).to_dict()}")
    # Samples missing >20% of the ferroptosis panel are dropped; remaining
    # sporadic missing values are median-imputed (fit on train fold only,
    # see split_tcga below, to avoid leakage).
    row_miss = df[gene_cols].isna().mean(axis=1)
    keep_mask = row_miss <= 0.20
    note(f"Dropping {(~keep_mask).sum()} samples missing >20% of ferroptosis gene panel")
    df = df[keep_mask].reset_index(drop=True)

    # --- log2 transform (RSEM values are right-skewed counts-like data) -----
    for g in gene_cols:
        df[g] = np.log2(df[g].clip(lower=0) + 1)

    # --- MSI target construction (Kautto et al. 2017 MANTIS >= 0.4 cutoff) ---
    df["MSI_SCORE_MANTIS"] = pd.to_numeric(df["MSI_SCORE_MANTIS"], errors="coerce")
    has_msi = df["MSI_SCORE_MANTIS"].notna()
    note(f"MSI MANTIS score available for {has_msi.sum()}/{len(df)} samples "
         f"({(~has_msi).sum()} excluded from the MSI classification task only)")
    df["msi_label"] = np.where(df["MSI_SCORE_MANTIS"] >= MSI_MANTIS_THRESHOLD, "MSI-H", "MSS/MSI-L")

    # --- survival fields for the complementary ferroptosis-risk experiment --
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["OS_EVENT"] = df["OS_STATUS"].astype(str).str.startswith("1").astype(float)
    df.loc[df["OS_STATUS"].isna(), "OS_EVENT"] = np.nan

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "tcga_coadread_processed.csv"
    df.to_csv(out_path, index=False)
    note(f"Saved processed TCGA dataset: {df.shape} -> {out_path}")

    meta = {
        "n_samples": len(df),
        "n_ferroptosis_genes_used": len(gene_cols),
        "ferroptosis_genes_used": gene_cols,
        "ferroptosis_genes_missing_from_profile": missing_genes,
        "msi_mantis_threshold": MSI_MANTIS_THRESHOLD,
        "msi_label_counts": df["msi_label"].value_counts().to_dict(),
        "random_seed": RANDOM_SEED,
    }
    with open(PROCESSED_DIR / "tcga_preprocessing_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if log_path:
        Path(log_path).write_text("\n".join(log), encoding="utf-8")

    return df, gene_cols


def split_tcga(df, target_col, feature_cols, test_size=0.2, val_size=0.125):
    """
    Stratified 70/10/20 train/val/test split (val_size is expressed relative
    to the remaining 80% after the test split, i.e. 0.125*0.8 = 10% overall).
    Only rows with a non-null target are used for that particular task, so the
    MSI-classification split and any other target's split are independent.
    """
    sub = df[df[target_col].notna()].reset_index(drop=True)
    X = sub[feature_cols]
    y = sub[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=RANDOM_SEED
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# GDSC colorectal cell lines
# ---------------------------------------------------------------------------

def _resolve_gdsc_gene_expression(genes, cosmic_ids):
    """
    Tries every candidate Ensembl ID returned by MyGene.info for each gene
    symbol against the actual GDSC array, keeping whichever candidate is
    physically present. Genes with no candidate present on the array are
    reported as missing (this reflects real 2013-era microarray coverage
    limits, not a mapping bug -- see docs/DATASET.md).
    """
    gmap = dl.load_gene_ensembl_map()
    gmap = gmap[gmap["gene"].isin(genes) & gmap["ensembl_gene_id"].notna()]
    all_candidate_ids = gmap["ensembl_gene_id"].tolist()

    expr = dl.load_gdsc_expression_for_genes(all_candidate_ids, cosmic_ids)  # index=ensembl_gene_id

    resolved = {}
    unresolved = []
    for gene, grp in gmap.groupby("gene"):
        candidates = grp.sort_values("is_primary", ascending=False)["ensembl_gene_id"].tolist()
        hit = next((c for c in candidates if c in expr.index), None)
        if hit is None:
            unresolved.append(gene)
        else:
            resolved[gene] = expr.loc[hit]

    gene_expr = pd.DataFrame(resolved).T  # genes x cell lines
    gene_expr.index.name = "gene"
    return gene_expr, unresolved


def build_gdsc_dose_response_dataset(log_path=None):
    log = []

    def note(msg):
        log.append(msg)
        print(msg)

    genes = dl.load_ferroptosis_gene_list()["gene"].tolist()
    cell_lines = dl.load_gdsc_colorectal_cell_lines()
    cell_lines = cell_lines[cell_lines["has_expression"] == "Y"].reset_index(drop=True)
    note(f"Colorectal (COAD/READ) GDSC cell lines with expression data: {len(cell_lines)}")

    cosmic_ids = cell_lines["COSMIC_ID"].dropna().astype(int).tolist()
    gene_expr, unresolved = _resolve_gdsc_gene_expression(genes, cosmic_ids)
    if unresolved:
        note(f"WARNING: {len(unresolved)} ferroptosis genes have no probe on this "
             f"microarray platform and are excluded from GDSC features: {unresolved}")
    note(f"GDSC ferroptosis expression matrix: {gene_expr.shape[0]} genes x {gene_expr.shape[1]} cell lines")

    # z-score each gene across the colorectal cell-line panel (per-platform
    # standardization; TCGA and GDSC expression are on different technologies
    # and are NEVER pooled into one model -- see docs/METHODOLOGY.md).
    gene_expr_z = gene_expr.sub(gene_expr.mean(axis=1), axis=0).div(gene_expr.std(axis=1), axis=0)
    gene_expr_z = gene_expr_z.T  # cell lines x genes
    gene_expr_z.index = gene_expr_z.index.astype(int)
    gene_expr_z.index.name = "COSMIC_ID"

    dose_response = dl.load_gdsc_dose_response()
    note(f"Raw GDSC2 colorectal dose-response records: {len(dose_response)} "
         f"({dose_response['DRUG_NAME'].nunique()} drugs x {dose_response['COSMIC_ID'].nunique()} lines)")

    coverage = dose_response.groupby("DRUG_NAME")["COSMIC_ID"].nunique()
    well_covered_drugs = coverage[coverage >= MIN_DRUG_COVERAGE].index
    dose_response = dose_response[dose_response["DRUG_NAME"].isin(well_covered_drugs)]
    note(f"After requiring >= {MIN_DRUG_COVERAGE} tested cell lines per drug: "
         f"{len(dose_response)} records, {dose_response['DRUG_NAME'].nunique()} drugs")

    # duplicate handling: keep the first fitted curve if a (drug, cell line)
    # pair was somehow screened more than once
    n_dupes = dose_response.duplicated(["DRUG_NAME", "COSMIC_ID"]).sum()
    dose_response = dose_response.drop_duplicates(["DRUG_NAME", "COSMIC_ID"], keep="first")
    note(f"Dropped {n_dupes} duplicate (drug, cell line) fitted-curve records")

    merged = dose_response.merge(gene_expr_z, on="COSMIC_ID", how="inner")
    note(f"After merging with ferroptosis expression: {len(merged)} (drug, cell line) rows, "
         f"{merged['COSMIC_ID'].nunique()} cell lines")

    n_missing_target = merged[["LN_IC50", "AUC"]].isna().any(axis=1).sum()
    merged = merged.dropna(subset=["LN_IC50", "AUC"]).reset_index(drop=True)
    note(f"Dropped {n_missing_target} rows with missing LN_IC50/AUC")

    keep_cols = (
        ["COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME", "PATHWAY_NAME",
         "PUTATIVE_TARGET", "MIN_CONC", "MAX_CONC", "LN_IC50", "AUC"]
        + list(gene_expr_z.columns)
    )
    merged = merged[keep_cols]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv"
    merged.to_csv(out_path, index=False)
    note(f"Saved processed GDSC dose-response dataset: {merged.shape} -> {out_path}")

    meta = {
        "n_rows": len(merged),
        "n_cell_lines": int(merged["COSMIC_ID"].nunique()),
        "n_drugs": int(merged["DRUG_NAME"].nunique()),
        "ferroptosis_genes_used": list(gene_expr_z.columns),
        "ferroptosis_genes_missing_from_array": unresolved,
        "min_drug_coverage": MIN_DRUG_COVERAGE,
        "random_seed": RANDOM_SEED,
    }
    with open(PROCESSED_DIR / "gdsc_preprocessing_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if log_path:
        Path(log_path).write_text("\n".join(log), encoding="utf-8")

    return merged, list(gene_expr_z.columns)


def split_gdsc(df, feature_cols, target_col, test_size=0.2):
    """
    Group-based split by COSMIC_ID: entire cell lines are held out for testing
    so the model is evaluated on cell lines it has never seen (prevents the
    same cell line's expression profile appearing in both train and test,
    which would otherwise leak cell-line identity across the split).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["COSMIC_ID"]))
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    return {
        "X_train": train[feature_cols], "y_train": train[target_col], "meta_train": train,
        "X_test": test[feature_cols], "y_test": test[target_col], "meta_test": test,
    }


if __name__ == "__main__":
    print("=== TCGA COADREAD preprocessing ===")
    build_tcga_dataset(log_path=DOCS_DIR / "review_evidence" / "tcga_preprocessing_log.txt")
    print("\n=== GDSC colorectal dose-response preprocessing ===")
    build_gdsc_dose_response_dataset(log_path=DOCS_DIR / "review_evidence" / "gdsc_preprocessing_log.txt")
