"""
Downloads the FULL protein-coding-gene RNA-seq V2 expression matrix for the
same TCGA COADREAD (PanCancer Atlas) cohort already used for the ferroptosis-
panel classifier, so a genuine whole-transcriptome baseline can be trained
and compared against the 35-gene ferroptosis-panel model (see
src/whole_transcriptome_baseline.py and docs/FUTURE_WORK.md item 2).

Fetched in chunks (the cBioPortal API is called once per ~2000-gene batch)
and appended incrementally to data/raw/tcga/whole_transcriptome_expression_long.csv
so memory stays bounded regardless of gene-panel size.

This script performs NO filtering/transformation beyond restricting to
protein-coding genes (pseudogenes, ncRNA, etc. are excluded as they are not
meaningfully comparable to the ferroptosis panel's protein-coding genes) --
QC, log2 transform, and feature reduction happen in
src/whole_transcriptome_baseline.py.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://www.cbioportal.org/api"
STUDY_ID = "coadread_tcga_pan_can_atlas_2018"
MRNA_PROFILE = f"{STUDY_ID}_rna_seq_v2_mrna"
SAMPLE_LIST = f"{STUDY_ID}_all"
CHUNK_SIZE = 2000

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "tcga"
OUT_PATH = RAW_DIR / "whole_transcriptome_expression_long.csv"
GENE_LIST_PATH = RAW_DIR / "whole_transcriptome_gene_list.csv"

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def _session():
    s = requests.Session()
    retry = Retry(total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                   allowed_methods=["GET", "POST"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


SESSION = _session()


def fetch_all_protein_coding_genes():
    print("Listing all protein-coding genes known to cBioPortal ...")
    r = SESSION.get(f"{API}/genes", params={"projection": "SUMMARY", "pageSize": 30000, "pageNumber": 0},
                     headers=HEADERS, timeout=120)
    r.raise_for_status()
    genes = [g for g in r.json() if g.get("type") == "protein-coding" and g.get("entrezGeneId", 0) > 0]
    df = pd.DataFrame(genes)
    df.to_csv(GENE_LIST_PATH, index=False)
    print(f"  {len(df)} protein-coding genes -> {GENE_LIST_PATH}")
    return df


def fetch_expression_in_chunks(entrez_ids):
    print(f"Fetching expression for {len(entrez_ids)} genes in chunks of {CHUNK_SIZE} ...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    total_rows = 0
    t0 = time.time()
    n_chunks = (len(entrez_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(entrez_ids), CHUNK_SIZE):
        chunk = entrez_ids[i:i + CHUNK_SIZE]
        body = {"entrezGeneIds": [int(x) for x in chunk], "sampleListId": SAMPLE_LIST}

        data = None
        last_err = None
        for attempt in range(5):
            try:
                r = SESSION.post(f"{API}/molecular-profiles/{MRNA_PROFILE}/molecular-data/fetch",
                                  params={"projection": "SUMMARY"}, json=body, headers=HEADERS, timeout=180)
                r.raise_for_status()
                data = r.json()
                break
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = e
                wait = 5 * (attempt + 1)
                print(f"    chunk {i // CHUNK_SIZE + 1}: network error ({e.__class__.__name__}), "
                      f"retrying in {wait}s (attempt {attempt + 1}/5) ...")
                time.sleep(wait)
        if data is None:
            raise RuntimeError(f"Failed to fetch chunk {i // CHUNK_SIZE + 1}/{n_chunks} after 5 attempts") from last_err

        df = pd.DataFrame(data)[["sampleId", "entrezGeneId", "value"]]
        df.to_csv(OUT_PATH, mode="a", header=(i == 0), index=False)
        total_rows += len(df)
        elapsed = time.time() - t0
        print(f"  chunk {i // CHUNK_SIZE + 1}/{n_chunks}: "
              f"{len(df)} rows ({total_rows} total, {elapsed:.0f}s elapsed)")
    return total_rows


def main():
    gene_df = fetch_all_protein_coding_genes()
    entrez_ids = gene_df["entrezGeneId"].astype(int).tolist()
    total_rows = fetch_expression_in_chunks(entrez_ids)

    meta = {
        "study_id": STUDY_ID,
        "source": "cBioPortal REST API (https://www.cbioportal.org/api)",
        "purpose": "Whole-transcriptome baseline for comparison against the 35-gene ferroptosis panel",
        "retrieved_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "n_genes_requested": len(entrez_ids),
        "n_rows_fetched": total_rows,
    }
    with open(RAW_DIR / "whole_transcriptome_fetch_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. {total_rows} expression records for {len(entrez_ids)} genes -> {OUT_PATH}")


if __name__ == "__main__":
    main()
