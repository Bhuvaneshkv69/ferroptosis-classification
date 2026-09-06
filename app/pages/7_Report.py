import sys
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403

import reporting  # noqa: E402

st.set_page_config(page_title="Report | Ferroptosis Pipeline", page_icon="🧬", layout="wide")
sidebar_status()
st.title("📄 Computational Report")
st.caption(
    "Generates a single, self-contained HTML report (dataset stats, classification and dose-response "
    "results, interpretability, limitations) by reading the pipeline's current output files. "
    "Nothing in the report is typed in by hand -- regenerating it after a pipeline re-run produces "
    "different, still-accurate numbers."
)

if not (RESULTS_DIR / "classification_results.csv").exists():
    st.error("No results found. Run `python main.py` first.")
    st.stop()

if st.button("Generate report now", type="primary"):
    with st.spinner("Reading results/, tables/, figures/, models/registry.json ..."):
        path = reporting.generate_report()
    st.success(f"Report generated: {path.relative_to(ROOT)}")
    st.download_button(
        "Download report (HTML)",
        path.read_bytes(),
        file_name=path.name,
        mime="text/html",
    )
    with st.expander("Preview"):
        st.components.v1.html(path.read_text(encoding="utf-8"), height=800, scrolling=True)

reports_dir = RESULTS_DIR / "reports"
if reports_dir.exists():
    existing = sorted(reports_dir.glob("*.html"), reverse=True)
    if existing:
        st.markdown("### Previously generated reports")
        for p in existing[:10]:
            c1, c2 = st.columns([4, 1])
            c1.write(p.name)
            c2.download_button("Download", p.read_bytes(), file_name=p.name, mime="text/html", key=str(p))
