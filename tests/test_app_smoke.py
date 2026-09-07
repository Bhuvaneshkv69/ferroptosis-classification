"""
End-to-end smoke tests for the Streamlit application using Streamlit's
in-process AppTest harness (runs each page's real Python script against the
real backend -- no browser needed, but no mocking either: these exercise the
actual data files and persisted models on disk).
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    APP_DIR / "main.py",
    APP_DIR / "pages" / "1_Data.py",
    APP_DIR / "pages" / "2_Preprocessing.py",
    APP_DIR / "pages" / "3_Ferroptosis_Classification.py",
    APP_DIR / "pages" / "4_Drug_Response.py",
    APP_DIR / "pages" / "5_Model_Performance.py",
    APP_DIR / "pages" / "6_Interpretability.py",
    APP_DIR / "pages" / "7_Report.py",
    APP_DIR / "pages" / "8_External_Validation.py",
]


def _prereqs_present():
    return (ROOT / "data" / "processed" / "tcga_coadread_features.csv").exists() and \
           (ROOT / "models" / "registry.json").exists()


@pytest.mark.skipif(not _prereqs_present(), reason="Run `python main.py` and "
                     "`python scripts/train_and_persist.py` first")
@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_page_runs_without_exception(page):
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()
    assert not at.exception, f"{page.name} raised: {[str(e) for e in at.exception]}"


@pytest.mark.skipif(not _prereqs_present(), reason="requires processed data + persisted models")
def test_classification_page_runs_a_real_prediction():
    at = AppTest.from_file(str(APP_DIR / "pages" / "3_Ferroptosis_Classification.py"), default_timeout=120)
    at.run()
    assert not at.exception
    # First radio option is "pick a real TCGA sample" (default selection)
    buttons = [b for b in at.button if b.label == "Run prediction"]
    assert buttons, "Run prediction button not found"
    buttons[0].click().run()
    assert not at.exception
    # A predicted-label metric should now be present
    metric_labels = [m.label for m in at.metric]
    assert "Predicted label" in metric_labels


@pytest.mark.skipif(not _prereqs_present(), reason="requires processed data + persisted models")
def test_drug_response_page_runs_a_real_prediction():
    at = AppTest.from_file(str(APP_DIR / "pages" / "4_Drug_Response.py"), default_timeout=120)
    at.run()
    assert not at.exception
    buttons = [b for b in at.button if b.label == "Run prediction"]
    assert buttons, "Run prediction button not found"
    buttons[0].click().run()
    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    assert any("Predicted" in lbl for lbl in metric_labels)


@pytest.mark.skipif(not _prereqs_present(), reason="requires processed data + persisted models")
def test_report_page_generates_a_real_report():
    at = AppTest.from_file(str(APP_DIR / "pages" / "7_Report.py"), default_timeout=120)
    at.run()
    assert not at.exception
    buttons = [b for b in at.button if b.label == "Generate report now"]
    assert buttons
    buttons[0].click().run()
    assert not at.exception
    success = [s.value for s in at.success]
    assert any("Report generated" in s for s in success)
