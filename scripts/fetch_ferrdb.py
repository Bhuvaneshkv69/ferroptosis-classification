"""Convert raw FerrDb V2 API JSON (data/raw/ferrdb/ferrdb_*_raw.json) into clean CSVs.

Source: FerrDb V2 (http://www.zhounan.org/ferrdb/current/), gene API endpoint
http://www.datasj.com:40013/bt2104/api/gene/ (POST geneType=driver|suppressor|marker|unclassified).
"""
import json
import csv
from pathlib import Path

RAW_DIR = Path("data/raw/ferrdb")
OUT_DIR = Path("data/raw/ferrdb")

GENE_TYPES = ["driver", "suppressor", "marker", "unclassified"]

for gt in GENE_TYPES:
    with open(RAW_DIR / f"ferrdb_{gt}_raw.json", encoding="utf-8") as f:
        records = json.load(f)
    out_path = OUT_DIR / f"ferrdb_v2_{gt}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "hgncid", "name", "experiments", "gene_type"])
        writer.writeheader()
        for r in records:
            r["gene_type"] = gt
            writer.writerow(r)
    print(f"{gt}: {len(records)} genes -> {out_path}")

# Combined driver+suppressor+marker list (the panel the project actually uses)
seen = {}
for gt in ["driver", "suppressor", "marker"]:
    with open(RAW_DIR / f"ferrdb_{gt}_raw.json", encoding="utf-8") as f:
        for r in json.load(f):
            sym = r["symbol"]
            if sym in seen:
                seen[sym]["gene_type"] += f"|{gt}"
            else:
                seen[sym] = {**r, "gene_type": gt}

combined_path = OUT_DIR / "ferrdb_v2_driver_suppressor_marker.csv"
with open(combined_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["symbol", "hgncid", "name", "experiments", "gene_type"])
    writer.writeheader()
    for sym in sorted(seen):
        writer.writerow(seen[sym])

print(f"combined (driver+suppressor+marker, deduped): {len(seen)} genes -> {combined_path}")
