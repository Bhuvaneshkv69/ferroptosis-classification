"""
Workspace entry point.

Run with:
    streamlit run app/main.py

Navigation is explicit (st.navigation). The production models are looked up from the
registry's champion record (models/champions.json); every number shown is read live
from data/processed/, results/benchmark/, or models/registry.json -- nothing is hard-coded.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from common import DISCLAIMER, classifier_ready, data_available, drug_model_ready  # noqa: E402
from theme import inject_css, init_theme, persist_theme, status_pill, theme_toggle  # noqa: E402

st.set_page_config(
    page_title="Ferroptosis Research Workspace",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_theme()          # one centralized theme (header toggle > session > ?theme= > light)
inject_css()

PAGES_DIR = APP_DIR / "pages"

theme_toggle()        # Light / Dark switch in the application header (applies to every page)
persist_theme()       # remembers the choice across browser refreshes

nav = st.navigation(
    {
        "Analysis": [
            st.Page(PAGES_DIR / "analyze.py", title="Analysis", icon=":material/biotech:", default=True),
        ],
        "Advanced": [
            st.Page(PAGES_DIR / "model_lab.py", title="Model Lab", icon=":material/science:"),
            st.Page(PAGES_DIR / "compare_models.py", title="Single-model details", icon=":material/hub:"),
            st.Page(PAGES_DIR / "data_explorer.py", title="Data", icon=":material/database:"),
            st.Page(PAGES_DIR / "train_compare.py", title="Train / Compare", icon=":material/model_training:"),
            st.Page(PAGES_DIR / "documentation.py", title="Documentation", icon=":material/menu_book:"),
        ],
    }
)

with st.sidebar:
    st.markdown(
        '<div style="font-weight:650; font-size:0.95rem; letter-spacing:-0.01em;">🧬 Ferroptosis Workspace</div>'
        '<div style="font-size:0.74rem; opacity:0.6; margin-bottom:0.9rem;">Computational research platform</div>',
        unsafe_allow_html=True,
    )
    st.caption("SYSTEM")
    status_pill("Data ready" if data_available() else "Data unavailable", "good" if data_available() else "critical")
    st.write("")
    ready = classifier_ready() and drug_model_ready()
    status_pill("Models ready" if ready else "Models incomplete", "good" if ready else "warning")
    st.divider()
    st.caption(DISCLAIMER)

nav.run()
