"""
Downloads an INDEPENDENT colorectal cancer cohort -- CPTAC-2 Prospective
Colon Cancer (Cell 2019) -- from the public cBioPortal REST API, for genuine
external (cross-cohort, cross-institution, cross-sequencing-batch)
validation of the MSI classifier trained on TCGA COADREAD.

Dataset: coad_cptac_2019
Source:  cBioPortal for Cancer Genomics (https://www.cbioportal.org)
Citation: Vasaikar et al., "Proteogenomic Analysis of Human Colon Cancer
          Reveals New Therapeutic Opportunities", Cell 2019.

Why this is a genuine external validation cohort (not just another TCGA
split): different patient population, different research consortium (NCI
CPTAC vs TCGA), different sample-processing pipeline and RNA-seq batch,
independently called MSI_STATUS clinical label (not the MANTIS score used
for TCGA -- see docs/DATASET.md for how the two labeling schemes compare).

This script performs NO filtering, imputation, or transformation -- that is
done in src/external_validation.py. It only retrieves and persists the raw
API responses, exactly like src/fetch_tcga.py does for the training cohort.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://www.cbioportal.org/api"
STUDY_ID = "coad_cptac_2019"
MRNA_PROFILE = f"{STUDY_ID}_rna_seq_v2_mrna"
SAMPLE_LIST = f"{STUDY_ID}_all"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "cptac_coad_2019"
GENE_LIST_PATH = Path(__file__).resolve().parent.parent / "data" / "unified_gene_signature.csv"

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def _get(path, params=None):
    r = requests.get(f"{API}{path}", headers=HEADERS, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _post(path, body, params=None):
    r = requests.post(f"{API}{path}", headers=HEADERS, json=body, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def fetch_clinical_sample_data():
    print("Fetching sample-level clinical data (MSI_STATUS, ...) ...")
    data = _get(f"/studies/{STUDY_ID}/clinical-data",
                params={"clinicalDataType": "SAMPLE", "projection": "SUMMARY", "pageSize": 5000})
    df = pd.DataFrame(data)
    out = RAW_DIR / "clinical_sample_long.csv"
    df.to_csv(out, index=False)
    print(f"  saved {len(df)} rows -> {out}")
    return df


def fetch_gene_entrez_ids(genes):
    print(f"Resolving Entrez IDs for {len(genes)} genes ...")
    resolved = _post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", genes)
    df = pd.DataFrame(resolved)
    found = set(df["hugoGeneSymbol"]) if len(df) else set()
    missing = sorted(set(genes) - found)
    if missing:
        print(f"  WARNING: {len(missing)} gene symbols not resolved: {missing}")
    out = RAW_DIR / "gene_entrez_map.csv"
    df.to_csv(out, index=False)
    print(f"  saved {len(df)} gene mappings -> {out}")
    return df


def fetch_mrna_expression(entrez_ids):
    print(f"Fetching RNA-seq expression for {len(entrez_ids)} genes across all samples ...")
    body = {"entrezGeneIds": [int(x) for x in entrez_ids], "sampleListId": SAMPLE_LIST}
    data = _post(f"/molecular-profiles/{MRNA_PROFILE}/molecular-data/fetch", body, params={"projection": "SUMMARY"})
    df = pd.DataFrame(data)
    out = RAW_DIR / "mrna_expression_long.csv"
    df.to_csv(out, index=False)
    print(f"  saved {len(df)} expression records -> {out}")
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    gene_df = pd.read_csv(GENE_LIST_PATH)
    genes = sorted(gene_df["gene"].unique().tolist())

    meta = {
        "study_id": STUDY_ID,
        "source": "cBioPortal REST API (https://www.cbioportal.org/api)",
        "underlying_data": "CPTAC-2 Prospective Colon Cancer (Vasaikar et al., Cell 2019)",
        "purpose": "Independent external validation cohort for the TCGA-trained MSI classifier",
        "retrieved_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "n_genes_requested": len(genes),
        "genes_requested": genes,
    }

    fetch_clinical_sample_data()
    gene_map = fetch_gene_entrez_ids(genes)
    entrez_ids = gene_map["entrezGeneId"].dropna().astype(int).tolist()
    fetch_mrna_expression(entrez_ids)

    with open(RAW_DIR / "fetch_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Raw CPTAC-2 COAD external-validation snapshot written to {RAW_DIR}")


if __name__ == "__main__":
    main()
