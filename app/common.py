"""
Shared bootstrap + cached accessors for every page. Everything shown in the app is read from files the
pipeline produced (data/processed/, results/benchmark/, models/registry.json + champions.json) -- no
number, label, model name or importance is hard-coded here.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import model_registry as reg  # noqa: E402
import model_zoo as mz  # noqa: E402
import predict as prd  # noqa: E402
import validation as val  # noqa: E402

RESULTS_DIR = ROOT / "results"
BENCH_DIR = RESULTS_DIR / "benchmark"
FIG_DIR = ROOT / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"
SESSIONS_PATH = RESULTS_DIR / "analysis_sessions.json"

DISCLAIMER = ("For research use only. Model outputs are computational predictions, not clinical "
              "diagnoses or treatment recommendations.")


# ------------------------------------------------------------------ availability
def data_available() -> bool:
    return (PROCESSED_DIR / "tcga_coadread_features.csv").exists() and \
           (PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv").exists()


def classifier_ready() -> bool:
    return reg.current_best("classification") is not None


def drug_model_ready() -> bool:
    return reg.current_best("dose_response") is not None


# ------------------------------------------------------------------ cached data
@st.cache_data(show_spinner=False)
def load_tcga_features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "tcga_coadread_features.csv")


@st.cache_data(show_spinner=False)
def load_tcga_meta() -> dict:
    return json.load(open(PROCESSED_DIR / "tcga_preprocessing_metadata.json"))


@st.cache_data(show_spinner=False)
def load_gdsc_processed() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv")


@st.cache_data(show_spinner=False)
def load_gdsc_meta() -> dict:
    return json.load(open(PROCESSED_DIR / "gdsc_preprocessing_metadata.json"))


@st.cache_data(show_spinner=False)
def load_gene_signature() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "unified_gene_signature.csv")


@st.cache_data(show_spinner=False)
def load_csv(relpath: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relpath)


@st.cache_data(show_spinner=False)
def load_json(relpath: str) -> dict:
    return json.load(open(ROOT / relpath))


def bench_csv(name: str):
    p = BENCH_DIR / name
    return pd.read_csv(p) if p.exists() else None


def bench_json(name: str):
    p = BENCH_DIR / name
    return json.load(open(p)) if p.exists() else None


@st.cache_data(show_spinner=False)
def metric_help() -> dict:
    return yaml.safe_load(open(ROOT / "configs" / "model_selection.yaml", encoding="utf-8"))["metric_help"]


@st.cache_resource(show_spinner="Loading model...")
def production(task: str, _stamp: str = ""):
    """(entry, bundle) of the current champion. `_stamp` busts the cache when the champion changes."""
    return prd.load_production(task)


def champion_stamp(task: str) -> str:
    rec = reg.champion_history().get(task) or {}
    return f"{rec.get('name')}:{rec.get('version')}"


def get_production(task: str):
    return production(task, champion_stamp(task))


@st.cache_data(show_spinner=False)
def reference_stats(genes: tuple) -> dict:
    df = load_tcga_features()
    out = {}
    for g in genes:
        raw = 2.0 ** df[g].dropna() - 1.0
        out[g] = {"median": float(raw.median()), "p5": float(raw.quantile(0.05)), "p95": float(raw.quantile(0.95))}
    return out


# ------------------------------------------------------------------ small UI helpers
def stability_stars(std: float) -> str:
    """Stability rating from the fold-to-fold standard deviation of the primary metric (stated in the UI)."""
    n = 5 if std <= 0.02 else 4 if std <= 0.035 else 3 if std <= 0.05 else 2 if std <= 0.08 else 1
    return "★" * n + "☆" * (5 - n)


def pct(x, d=0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def pm(mean, std, d=3) -> str:
    return f"{mean:.{d}f} ± {std:.{d}f}"


def metric_with_help(col, label: str, value: str, key: str, delta: str | None = None):
    col.metric(label, value, delta=delta, help=metric_help().get(key))


def figure(name: str, caption: str | None = None, width="stretch"):
    path = FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width=width)
    else:
        st.caption(f"Figure not yet generated: {name}.")


# ------------------------------------------------------------------ analysis log
def list_analysis_sessions() -> list[dict]:
    if not SESSIONS_PATH.exists():
        return []
    try:
        return json.load(open(SESSIONS_PATH, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def log_analysis_session(kind: str, summary: dict, name: str | None = None) -> dict:
    entries = list_analysis_sessions()
    nid = (entries[-1]["id"] + 1) if entries else 1
    entry = {"id": nid, "timestamp": datetime.now(timezone.utc).isoformat(), "kind": kind,
             "name": name or f"{kind.replace('_', ' ').title()} #{nid}", **summary}
    entries.append(entry)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(entries, open(SESSIONS_PATH, "w", encoding="utf-8"), indent=2, default=str)
    return entry


def full_cohort_csv(df: pd.DataFrame, genes: list[str], raw_scale: bool, id_col: str | None = None) -> bytes:
    """Entire real cohort as an upload-ready CSV with exactly the required gene columns; raw_scale=True inverts
    the stored log2(x+1) values back to the raw scale the Analysis page's upload expects."""
    out = df[[g for g in genes if g in df.columns]].copy()
    if raw_scale:
        out = 2.0 ** out - 1.0
    return out.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def _wt_matrix() -> pd.DataFrame:
    import extended_data as ed
    return ed.build_whole_transcriptome()


@st.cache_data(show_spinner=False)
def extended_reference_stats(genes: tuple) -> dict:
    """Median / 5th / 95th percentile (RAW scale) of the extra genes in the training cohort."""
    wt = _wt_matrix()
    out = {}
    for g in genes:
        raw = 2.0 ** wt[g].dropna().astype(float) - 1.0
        out[g] = {"median": float(raw.median()), "p5": float(raw.quantile(0.05)), "p95": float(raw.quantile(0.95))}
    return out


def extended_log2(sample_ids, genes) -> pd.DataFrame:
    """log2 values of extra (mismatch-repair / immune) genes for real TCGA samples."""
    wt = _wt_matrix()
    return wt.loc[list(sample_ids), list(genes)].astype(float).reset_index(drop=True)
