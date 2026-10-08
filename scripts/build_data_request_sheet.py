"""Append the complete (all-sample) tables to docs/DATA_REQUEST_SHEET.xlsx.

Whole-transcriptome and GDSC raw/expression matrices exceed Excel's 1,048,576
row limit, so they stay as files under data/raw/ (listed on the 'Large Files' tab).
"""
import gzip
import re
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
XLSX = ROOT / "docs" / "DATA_REQUEST_SHEET.xlsx"


def tcga_clinical():
    pat = pd.read_csv(RAW / "tcga/clinical_patient_long.csv").pivot_table(
        index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    smp = pd.read_csv(RAW / "tcga/clinical_sample_long.csv").pivot_table(
        index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first")
    s = pd.read_csv(RAW / "tcga/samples.csv")[["sampleId", "patientId", "sampleType"]]
    out = s.merge(smp, on="sampleId", how="left").merge(pat, on="patientId", how="left", suffixes=("", "_patient"))
    return out


def tcga_expression():
    m = pd.read_csv(RAW / "tcga/mrna_expression_long.csv", usecols=["sampleId", "entrezGeneId", "value"])
    g = pd.read_csv(RAW / "tcga/gene_entrez_map.csv")
    sym = dict(zip(g.iloc[:, 0], g.iloc[:, 1])) if g.iloc[:, 0].dtype != object else dict(zip(g.iloc[:, 1], g.iloc[:, 0]))
    m["gene"] = m["entrezGeneId"].map(sym).fillna(m["entrezGeneId"].astype(str))
    return m.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="first").reset_index()


def geo_samples(name):
    rows = {}
    with gzip.open(RAW / f"geo/{name}_series_matrix.txt.gz", "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            key, vals = parts[0][len("!Sample_"):], parts[1:]
            if key == "characteristics_ch1":
                for i, v in enumerate(vals):
                    # GSE14333 packs "k: v; k: v" into one cell
                    for kv in re.split(r";\s*(?=[\w.\- ()]+:)", v):
                        if ":" in kv:
                            k, val = kv.split(":", 1)
                            rows.setdefault(k.strip(), [None] * len(vals))[i] = val.strip()
            elif key in ("title", "geo_accession", "source_name_ch1"):
                rows[key] = vals
    return pd.DataFrame(rows)


def main():
    sheets = {
        "TCGA_Clinical": tcga_clinical(),
        "TCGA_FerrGene_Expression": tcga_expression(),
        "FerrDb_Genes": pd.read_csv(RAW / "ferrdb/ferrdb_v2_driver_suppressor_marker.csv"),
        "GDSC_CellLines": pd.read_excel(RAW / "gdsc/Cell_Lines_Details.xlsx", sheet_name="Cell line details"),
        "GSE39582_Samples": geo_samples("GSE39582"),
        "GSE17536_Samples": geo_samples("GSE17536"),
        "GSE14333_Samples": geo_samples("GSE14333"),
        "Large Files": pd.DataFrame([
            ("TCGA whole transcriptome", "data/raw/tcga/whole_transcriptome_expression_long.csv", "19,222 genes x 594 samples (long format, ~11M rows)"),
            ("GDSC expression matrix", "data/raw/gdsc/sanger1018_brainarray_ensemblgene_rma.txt(.gz)", "all cell lines x all genes"),
            ("GDSC raw dose-response", "data/raw/gdsc/GDSC2_public_raw_data_24Jul22.csv", "2.1 GB"),
            ("GDSC fitted dose-response", "data/raw/gdsc/GDSC2_fitted_dose_response.csv", "IC50 / AUC all lines and drugs"),
            ("GEO expression matrices", "data/raw/geo/GSE*_series_matrix.txt.gz", "probe-level expression, all samples"),
        ], columns=["Dataset", "Path", "Why not in Excel"]),
    }
    wb = load_workbook(XLSX)
    for n in sheets:
        if n in wb.sheetnames:
            del wb[n]
    wb.save(XLSX)
    with pd.ExcelWriter(XLSX, mode="a", engine="openpyxl") as w:
        for n, df in sheets.items():
            df.to_excel(w, sheet_name=n, index=False)
            ws = w.sheets[n]
            ws.freeze_panes = "B2"
            print(n, df.shape)


if __name__ == "__main__":
    main()
