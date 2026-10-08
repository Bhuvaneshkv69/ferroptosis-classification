"""Fetches extra CPTAC-2 colon RNA-seq genes (mismatch-repair, immune, and optionally ALL protein-coding genes)
for external validation of the extended / whole-transcriptome models. Raw API output only; no processing.

    python scripts/fetch_cptac_extended.py [--all-genes]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import extended_data as ed  # noqa: E402
import fetch_external_validation as fx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-genes", action="store_true")
    a = ap.parse_args()
    cfg = ed.feature_config()
    if a.all_genes:
        gl = pd.read_csv(ed.RAW / "whole_transcriptome_gene_list.csv")
        ids = gl["entrezGeneId"].dropna().astype(int).unique().tolist()
        gmap = gl[["entrezGeneId", "hugoGeneSymbol"]].drop_duplicates()
    else:
        genes = cfg["mmr"]["genes"] + cfg["immune"]["genes"]
        gmap = pd.DataFrame(fx._post("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", genes))[
            ["entrezGeneId", "hugoGeneSymbol"]]
        ids = gmap["entrezGeneId"].astype(int).tolist()
    frames = []
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        body = {"entrezGeneIds": chunk, "sampleListId": fx.SAMPLE_LIST}
        data = None
        for attempt in range(3):
            try:
                data = fx._post(f"/molecular-profiles/{fx.MRNA_PROFILE}/molecular-data/fetch", body,
                                params={"projection": "SUMMARY"})
                break
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, e)
                time.sleep(3)
        frames.append(pd.DataFrame(data)[["sampleId", "entrezGeneId", "value"]])
        print(f"  fetched genes {i}-{i + len(chunk)} of {len(ids)}", flush=True)
    out = pd.concat(frames).merge(gmap, on="entrezGeneId", how="left")
    name = "mrna_expression_all_genes_long.csv" if a.all_genes else "mrna_expression_extended_long.csv"
    path = fx.RAW_DIR / name
    out.to_csv(path, index=False)
    print("saved", len(out), "rows ->", path)


if __name__ == "__main__":
    main()
