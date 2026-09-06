"""
Shared bootstrap + cached data/model accessors for every page of the
Streamlit application. Every function here reads real files the pipeline
produced (data/processed/, results/, models/registry.json) -- nothing is
hard-coded. Import this at the top of every page:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common import *
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import inference as inf  # noqa: E402
import model_registry as reg  # noqa: E402
import validation as val  # noqa: E402

RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"

DISCLAIMER = (
    "This system is intended for computational research and does not provide "
    "clinical diagnosis, treatment recommendations, or patient-specific dosing."
)


def data_available() -> bool:
    return (PROCESSED_DIR / "tcga_coadread_features.csv").exists() and \
           (PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv").exists()


def models_available() -> bool:
    return (ROOT / "models" / "registry.json").exists() and len(reg.list_models()) > 0


@st.cache_data(show_spinner=False)
def load_tcga_features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "tcga_coadread_features.csv")


@st.cache_data(show_spinner=False)
def load_tcga_meta() -> dict:
    with open(PROCESSED_DIR / "tcga_preprocessing_metadata.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_gdsc_processed() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv")


@st.cache_data(show_spinner=False)
def load_gdsc_meta() -> dict:
    with open(PROCESSED_DIR / "gdsc_preprocessing_metadata.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_gene_signature() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "unified_gene_signature.csv")


@st.cache_data(show_spinner=False)
def load_csv(relpath: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relpath)


@st.cache_data(show_spinner=False)
def load_registry() -> list[dict]:
    return reg.list_models()


@st.cache_resource(show_spinner=True)
def get_classification_model(model_name: str, experiment: str = "exp1_msi_raw_genes"):
    return inf.load_classification_model(model_name, experiment)


@st.cache_resource(show_spinner=True)
def get_dose_response_model(model_name: str, target: str = "LN_IC50"):
    return inf.load_dose_response_model(model_name, target)


def figure(name: str, caption: str | None = None, width: int | str = "stretch"):
    path = FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width=width)
    else:
        st.info(f"Figure not yet generated: {name}. Run `python main.py` to produce it.")


def sidebar_status():
    with st.sidebar:
        st.markdown("### Pipeline status")
        st.write("Processed data:", "✅" if data_available() else "❌ run `python main.py`")
        st.write("Persisted models:", "✅" if models_available() else "❌ run `python scripts/train_and_persist.py`")
        st.caption(DISCLAIMER)
