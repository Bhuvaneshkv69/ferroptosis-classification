"""
Downloads TCGA Colorectal Adenocarcinoma (PanCancer Atlas) data from the public
cBioPortal REST API and writes raw, unmodified snapshots to data/raw/tcga/.

Dataset: coadread_tcga_pan_can_atlas_2018
Source:  cBioPortal for Cancer Genomics (https://www.cbioportal.org)
         Original data: TCGA PanCancer Atlas (https://gdc.cancer.gov/about-data/publications/pancanatlas)
Citation: TCGA, Cell 2018 (multiple PanCancer Atlas marker papers, see docs/DATASET.md)

This script performs NO filtering, imputation, or transformation -- that is done
in src/preprocessing.py. It only retrieves and persists the raw API responses.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://www.cbioportal.org/api"
STUDY_ID = "coadread_tcga_pan_can_atlas_2018"
MRNA_PROFILE = f"{STUDY_ID}_rna_seq_v2_mrna"
SAMPLE_LIST = f"{STUDY_ID}_all"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "tcga"
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


def _fetch_clinical_data(clinical_data_type):
    data = []
    offset = 0
    page_size = 5000
    while True:
        chunk = _get(
            f"/studies/{STUDY_ID}/clinical-data",
            params={
                "clinicalDataType": clinical_data_type,
                "projection": "SUMMARY",
                "pageSize": page_size,
                "pageNumber": offset // page_size,
            },
        )
        data.extend(chunk)
        if len(chunk) < page_size:
            break
        offset += page_size
    return pd.DataFrame(data)


def fetch_clinical_patient_data():
    print("Fetching patient-level clinical data ...")
    df = _fetch_clinical_data("PATIENT")
    out = RAW_DIR / "clinical_patient_long.csv"
    df.to_csv(out, index=False)
    print(f"  saved {len(df)} rows -> {out}")
    return df


def fetch_clinical_sample_data():
    print("Fetching sample-level clinical data (MSI scores, mutation count, sample type, ...) ...")
    df = _fetch_clinical_data("SAMPLE")
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
        print(f"  WARNING: {len(missing)} gene symbols not resolved by cBioPortal: {missing}")
    out = RAW_DIR / "gene_entrez_map.csv"
    df.to_csv(out, index=False)
    print(f"  saved {len(df)} gene mappings -> {out}")
    return df


def fetch_mrna_expression(entrez_ids):
    print(f"Fetching RNA-seq V2 (RSEM) expression for {len(entrez_ids)} genes across all samples ...")
    body = {"entrezGeneIds": [int(x) for x in entrez_ids], "sampleListId": SAMPLE_LIST}
    data = _post(f"/molecular-profiles/{MRNA_PROFILE}/molecular-data/fetch", body, params={"projection": "SUMMARY"})
    df = pd.DataFrame(data)
    out = RAW_DIR / "mrna_expression_long.csv"
    df.to_csv(out, index=False)
    print(f"  saved {len(df)} expression records -> {out}")
    return df


def fetch_sample_list():
    print("Fetching sample metadata ...")
    data = _get(f"/studies/{STUDY_ID}/samples", params={"projection": "SUMMARY", "pageSize": 5000})
    df = pd.DataFrame(data)
    out = RAW_DIR / "samples.csv"
    df.to_csv(out, index=False)
    print(f"  saved {len(df)} samples -> {out}")
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    gene_df = pd.read_csv(GENE_LIST_PATH)
    genes = sorted(gene_df["gene"].unique().tolist())

    meta = {
        "study_id": STUDY_ID,
        "source": "cBioPortal REST API (https://www.cbioportal.org/api)",
        "underlying_data": "TCGA PanCancer Atlas, Colorectal Adenocarcinoma (COAD+READ)",
        "retrieved_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "n_genes_requested": len(genes),
        "genes_requested": genes,
    }

    fetch_sample_list()
    fetch_clinical_patient_data()
    fetch_clinical_sample_data()
    gene_map = fetch_gene_entrez_ids(genes)
    entrez_ids = gene_map["entrezGeneId"].dropna().astype(int).tolist()
    fetch_mrna_expression(entrez_ids)

    with open(RAW_DIR / "fetch_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Raw TCGA snapshot written to {RAW_DIR}")


if __name__ == "__main__":
    main()
