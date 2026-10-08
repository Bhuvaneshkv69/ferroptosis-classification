"""
End-to-end smoke tests for the application using Streamlit's in-process AppTest harness (each page's real
script runs against the real data files and the real registered champion models -- no mocking).
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE_PATHS = ["pages/analyze.py", "pages/model_lab.py", "pages/compare_models.py", "pages/data_explorer.py", "pages/train_compare.py",
              "pages/documentation.py"]


def _ready():
    champs = ROOT / "models" / "champions.json"
    return (ROOT / "data" / "processed" / "tcga_coadread_features.csv").exists() and champs.exists() and \
        {"classification", "dose_response"} <= set(json.load(open(champs)))


pytestmark = pytest.mark.skipif(not _ready(), reason="run scripts/run_benchmark.py (both tasks) first")


def _app() -> AppTest:
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"main.py raised: {[str(e) for e in at.exception]}"
    return at


def _btn(at, label):
    b = [x for x in at.button if x.label == label]
    assert b, f"button '{label}' not found; have {[x.label for x in at.button]}"
    return b[0]


@pytest.mark.parametrize("page", PAGE_PATHS)
def test_page_runs_without_exception(page):
    at = _app()
    at.switch_page(page)
    at.run()
    assert not at.exception, f"{page} raised: {[str(e) for e in at.exception]}"


def _start_reference_flow(at, choice):
    _btn(at, "Start Analysis").click().run()
    at.segmented_control(key="input_mode").set_value("Reference sample").run()
    assert not at.exception
    _btn(at, "Continue").click().run()
    seg = [s for s in at.segmented_control if s.label == "Analysis"][0]
    seg.set_value(choice).run()
    _btn(at, "Run analysis").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_classification_flow_produces_a_real_prediction():
    at = _app()
    at.switch_page("pages/analyze.py").run()
    _start_reference_flow(at, "Classification")
    labels = [m.label for m in at.metric]
    assert "Predicted subgroup" in labels
    assert any(l.startswith(("Model probability", "Calibrated probability", "MSI-High probability")) for l in labels)
    # accuracy may appear only as an explicitly labelled MODEL-PERFORMANCE metric, never as this sample's result
    assert all(l in ("Locked-test accuracy", "Validation accuracy") for l in labels if "accuracy" in l.lower())
    prob = [m for m in at.metric if m.label.startswith(("Model probability", "Calibrated probability", "MSI-High probability"))][0].value
    assert prob.endswith("%") and 0 <= int(prob[:-1]) <= 100


def test_both_flow_shows_drug_response_in_ic50_units():
    at = _app()
    at.switch_page("pages/analyze.py").run()
    _start_reference_flow(at, "Both")
    labels = {m.label: m.value for m in at.metric}
    assert labels["Predicted response"].startswith("IC50 ≈ ")
    assert "Typical error (MAE)" in labels
    assert not any(k.lower().startswith("accuracy") and "drug" in k.lower() for k in labels)


def test_manual_entry_path_scores_a_sample():
    at = _app()
    at.switch_page("pages/analyze.py").run()
    _btn(at, "Start Analysis").click().run()
    at.segmented_control(key="input_mode").set_value("Enter sample").run()
    assert not at.exception
    fields = [n for n in at.number_input if n.key and n.key.startswith("man_")]
    assert len(fields) == 35
    _btn(at, "Continue").click().run()
    seg = [s for s in at.segmented_control if s.label == "Analysis"][0]
    seg.set_value("Classification").run()
    _btn(at, "Run analysis").click().run()
    assert not at.exception
    assert "Predicted subgroup" in [m.label for m in at.metric]


def test_guided_exploration_options_all_render():
    at = _app()
    at.switch_page("pages/analyze.py").run()
    _start_reference_flow(at, "Both")
    pick = [s for s in at.segmented_control if s.label == "Explore"][0]
    for opt in ["Explore drug response", "Compare models", "Export analysis", "Understand this result"]:
        pick.set_value(opt).run()
        assert not at.exception, f"{opt}: {[str(e) for e in at.exception]}"


def test_model_page_shows_real_leaderboard_and_verdict():
    at = _app()
    at.switch_page("pages/compare_models.py").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Best validated model" in text
    assert any(("90% target" in m.value) for m in at.markdown)


def test_try_an_example_goes_straight_to_results():
    at = _app()
    at.switch_page("pages/analyze.py").run()
    _btn(at, "Try an example").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    labels = {m.label: m.value for m in at.metric}
    assert "Predicted subgroup" in labels and labels["Predicted response"].startswith("IC50")


def _champion_is_ensemble():
    import model_registry as reg
    e = reg.current_best("classification")
    return bool(e and e.get("extra", {}).get("kind") == "ensemble")


def test_core_model_agreement_view_shows_only_if_the_champion_is_really_an_ensemble():
    at = _app()
    at.switch_page("pages/analyze.py").run()
    _start_reference_flow(at, "Classification")
    labels = [e.label for e in at.expander]
    assert ("Model agreement" in labels) == _champion_is_ensemble()


def test_extended_model_is_selectable_and_labelled_as_not_ferroptosis_only():
    import model_registry as reg
    if reg.current_best("classification_extended") is None:
        pytest.skip("no extended model registered")
    at = _app()
    at.switch_page("pages/analyze.py").run()
    _btn(at, "Start Analysis").click().run()
    at.segmented_control(key="input_mode").set_value("Reference sample").run()
    _btn(at, "Continue").click().run()
    fs = reg.current_best("classification_extended")["feature_pipeline"].split("|")[0]
    import terminology as T
    label = T.WITH_GENES[fs]
    at.segmented_control(key="clf_model_pick").set_value(label).run()
    text = " ".join(i.value for i in at.info)
    assert "not** a ferroptosis-only classifier" in text or "not a ferroptosis-only classifier" in text
    seg = [x for x in at.segmented_control if x.label == "Analysis"][0]
    seg.set_value("Classification").run()
    _btn(at, "Run analysis").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    captions = " ".join(c.value for c in at.caption)
    assert "not a ferroptosis-only classifier" in captions
    assert "Predicted subgroup" in [m.label for m in at.metric]


def test_model_lab_reports_the_90_percent_statement_from_results():
    at = _app()
    at.switch_page("pages/model_lab.py").run()
    assert not at.exception, [str(e) for e in at.exception]
    text = " ".join(x.value for x in list(at.success) + list(at.warning))
    assert "90% target" in text
