"""
Feature sets for the extended MSI experiments: the 35-gene ferroptosis core, + mismatch-repair genes,
+ immune genes ("Extended molecular model"), and the whole transcriptome. Built on the SAME labelled samples
and the SAME locked split as the 35-gene pipeline (see benchmark.locked_split); gene lists come from
configs/feature_sets.yaml (fixed from the literature, not from this data).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tcga"
PROCESSED = ROOT / "data" / "processed"
WT_CACHE = PROCESSED / "tcga_whole_transcriptome_log2.pkl"


def feature_config() -> dict:
    return yaml.safe_load(open(ROOT / "configs" / "feature_sets.yaml", encoding="utf-8"))


def core_genes() -> list[str]:
    return json.load(open(PROCESSED / "tcga_preprocessing_metadata.json"))["ferroptosis_genes_used"]


def feature_set_genes(name: str) -> list[str]:
    """Gene list for core / core_plus_mmr / extended_molecular (not whole_transcriptome)."""
    cfg = feature_config()
    if name == "core":
        return core_genes()
    out = []
    for part in cfg[name]["composition"]:
        out += core_genes() if part == "core" else cfg[part]["genes"]
    return list(dict.fromkeys(out))


def build_whole_transcriptome(force: bool = False) -> pd.DataFrame:
    """samples x genes, log2(RSEM+1), cached. NaNs are kept (imputed inside each training fold)."""
    if WT_CACHE.exists() and not force:
        return pd.read_pickle(WT_CACHE)
    long = pd.read_csv(RAW / "whole_transcriptome_expression_long.csv", usecols=["sampleId", "entrezGeneId", "value"])
    gl = pd.read_csv(RAW / "whole_transcriptome_gene_list.csv")[["entrezGeneId", "hugoGeneSymbol"]]
    long = long.merge(gl.drop_duplicates("entrezGeneId"), on="entrezGeneId")
    wide = long.pivot_table(index="sampleId", columns="hugoGeneSymbol", values="value", aggfunc="first")
    wide.columns.name = None
    wide = np.log2(wide.clip(lower=0) + 1.0).astype("float32")
    wide.to_pickle(WT_CACHE)
    return wide


def load_matrix(name: str, sample_ids) -> pd.DataFrame:
    """Rows aligned to `sample_ids`. Core genes come from the processed table (already log2); every other set
    from the whole-transcriptome matrix (log2 of the same RSEM values -- identical to the core transform)."""
    wt = build_whole_transcriptome()
    ids = list(sample_ids)
    if name == "whole_transcriptome":
        return wt.loc[ids].reset_index(drop=True)
    core = pd.read_csv(PROCESSED / "tcga_coadread_processed.csv").set_index("sampleId")
    parts = []
    for g in feature_set_genes(name):
        src = core.loc[ids, g] if g in core.columns else wt.loc[ids, g].astype(float)
        parts.append(src.rename(g))
    return pd.concat(parts, axis=1).reset_index(drop=True)


def dev_gene_filter(X_dev: pd.DataFrame, max_missing: float) -> list[str]:
    """Label-blind, DEVELOPMENT-set-only column filter (missing-value fraction)."""
    return X_dev.columns[X_dev.isna().mean() <= max_missing].tolist()
