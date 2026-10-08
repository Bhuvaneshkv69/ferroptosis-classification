"""
Resolves HUGO gene symbols in the unified ferroptosis signature to Ensembl gene
IDs via the public MyGene.info API, and saves the result as a raw, reproducible
lookup table. Needed because the GDSC cell-line expression matrix
(sanger1018_brainarray_ensemblgene_rma.txt.gz) is indexed by Ensembl gene ID,
not HUGO symbol.
"""
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UNIFIED_PATH = DATA_DIR / "unified_gene_signature.csv"
OUT_DIR = DATA_DIR / "raw" / "gdsc"
OUT_PATH = OUT_DIR / "gene_ensembl_map.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(UNIFIED_PATH)["gene"].tolist()

    r = requests.post(
        "https://mygene.info/v3/query",
        data={"q": ",".join(genes), "scopes": "symbol", "fields": "symbol,ensembl.gene", "species": "human"},
        timeout=60,
    )
    r.raise_for_status()
    res = r.json()

    rows = []
    for x in res:
        symbol = x.get("query")
        ens = x.get("ensembl")
        if ens is None:
            rows.append({"gene": symbol, "ensembl_gene_id": None, "is_primary": True})
            continue
        if isinstance(ens, dict):
            ens = [ens]
        for i, e in enumerate(ens):
            rows.append({"gene": symbol, "ensembl_gene_id": e.get("gene"), "is_primary": i == 0})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    n_missing = df.loc[df["is_primary"], "ensembl_gene_id"].isna().sum()
    print(f"Resolved {df['gene'].nunique()} genes -> {OUT_PATH}")
    if n_missing:
        print(f"  WARNING: {n_missing} genes had no Ensembl ID returned by MyGene.info")


if __name__ == "__main__":
    main()
