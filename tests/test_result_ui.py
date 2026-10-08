"""Result popup, latest-analysis card, model-performance display and dark mode (no model is trained here)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
for p in (str(APP_DIR), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

pytestmark = pytest.mark.skipif(not (ROOT / "models" / "champions.json").exists(), reason="no production models")

from streamlit.testing.v1 import AppTest  # noqa: E402

BANNED = ("success", "diagnosis confirmed", "chance of cancer", "chance the patient", "positive", "negative", "danger",
          "unhealthy", "healthy", "safe ")


# ------------------------------------------------------------------------------------------------ helpers
def _app(query_theme=None):
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=180)
    if query_theme:
        at.query_params["theme"] = query_theme
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _btn(at, label):
    b = [x for x in at.button if x.label == label]
    assert b, f"no button {label!r}; have {[x.label for x in at.button]}"
    return b[0]


def _reference_run(at, choice="Classification", model_label=None):
    at.switch_page("pages/analyze.py").run()
    _btn(at, "Start Analysis").click().run()
    at.segmented_control(key="input_mode").set_value("Reference sample").run()
    _btn(at, "Continue").click().run()
    if model_label:
        at.segmented_control(key="clf_model_pick").set_value(model_label).run()
    [s for s in at.segmented_control if s.label == "Analysis"][0].set_value(choice).run()
    _btn(at, "Run analysis").click().run()
    return at


def _popup_html(at):
    return " ".join(m.value for m in at.markdown if "ws-result" in m.value)


def _real_inputs(n=3):
    import model_registry as reg
    import predict as prd
    entry = reg.current_best("classification")
    bundle = reg.load_model(entry)
    tc = pd.read_csv(ROOT / "data" / "processed" / "tcga_coadread_features.csv").dropna(subset=["msi_label"])
    X = tc[bundle["features"]].head(n).reset_index(drop=True)
    return entry, bundle, X, prd.classify(bundle, X)


def _contrast(c1: str, c2: str) -> float:
    def lum(h):
        h = h.lstrip("#")
        rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    a, b = sorted((lum(c1), lum(c2)), reverse=True)
    return (a + 0.05) / (b + 0.05)


# ------------------------------------------------------------------------------------------------ pure summary logic
def test_summary_uses_the_actual_prediction_and_probability():
    import result_view as rv
    import predict as prd
    entry, bundle, X, out = _real_inputs()
    for i in range(len(out)):
        members = prd.member_table(bundle, X.iloc[[i]]) if prd.is_ensemble(bundle) else None
        s = rv.summarize_sample(out.iloc[i], bundle=bundle, entry=entry, members=members, source="upload")
        assert s["raw_class"] == out.iloc[i]["predicted_class"]
        assert s["predicted_class"] == rv.display_class(out.iloc[i]["predicted_class"])
        assert s["probability"] == pytest.approx(float(out.iloc[i]["model_probability"]))      # the real model output, not a constant
        assert s["probability_valid"] and entry["calibration"] in ("sigmoid", "isotonic")


def test_probability_is_never_fabricated():
    import result_view as rv
    entry, bundle, X, out = _real_inputs(1)
    row = out.iloc[0]
    uncal = rv.summarize_sample(row, bundle=bundle, entry={**entry, "calibration": "none"}, members=None, source="upload")
    assert uncal["probability"] is None and not uncal["probability_valid"]
    html = rv.result_card_html(uncal)
    assert "%" not in html                                                    # no percentage of any kind is shown
    assert "Predicted class:" in html and "Model confidence information is available in Detailed Results." in html
    assert "Prediction probability" not in html
    nan_row = row.copy()
    nan_row["model_probability"] = np.nan
    assert rv.summarize_sample(nan_row, bundle=bundle, entry=entry, members=None, source="upload")["probability"] is None


def test_model_agreement_is_computed_from_the_real_member_predictions():
    import result_view as rv
    import predict as prd
    entry, bundle, X, out = _real_inputs(4)
    assert prd.is_ensemble(bundle)
    for i in range(len(out)):
        mt = prd.member_table(bundle, X.iloc[[i]])
        s = rv.summarize_sample(out.iloc[i], bundle=bundle, entry=entry, members=mt, source="upload")
        a = s["agreement"]
        votes = np.where(mt["supports_high"].to_numpy(bool), bundle["pos_label"], bundle["neg_label"])
        assert a["n"] == len(mt) == 5
        assert a["agree"] == int((votes == out.iloc[i]["predicted_class"]).sum())
        assert [m["probability"] for m in a["members"]] == pytest.approx(list(mt["probability"]))
        assert f"{a['agree']} of {a['n']} models agree" in rv.result_card_html(s)
        assert len(rv.members_text(s).splitlines()) >= a["n"]
    single = rv.summarize_sample(out.iloc[0], bundle=bundle, entry=entry, members=None, source="upload")
    assert single["agreement"] is None and "no model agreement" in rv.result_card_html(single)


def test_model_performance_matches_the_authoritative_registry():
    import model_registry as reg
    import result_view as rv
    final = json.load(open(ROOT / "results" / "final_model_registry.json", encoding="utf-8"))
    for task in ("classification", "classification_extended"):
        entry = reg.current_best(task)
        perf = rv.performance_for(entry)
        m = next(x for x in final["models"] if x["version"] == entry["version"])
        assert perf["test_accuracy"] == m["locked_test"]["accuracy"] and perf["test_auc"] == m["locked_test"]["roc_auc"]
        assert perf["cv_accuracy"] == m["cv"]["accuracy_mean"] and perf["cv_accuracy_sd"] == m["cv"]["accuracy_std"]
        assert perf["test_recall"] == m["locked_test"]["recall"]
    assert rv.performance_for({"version": "v-not-in-registry"}) is None          # missing -> absent, never invented
    assert rv.pct1(0.8633) == "86.3%"


def test_explanations_for_msi_h_and_mss_and_neutral_language():
    import result_view as rv
    entry, bundle, X, out = _real_inputs(1)
    pos, neg = out.iloc[0].copy(), out.iloc[0].copy()
    pos["predicted_class"], neg["predicted_class"] = bundle["pos_label"], bundle["neg_label"]
    for row, needle in ((pos, "classified the molecular profile as <b>MSI-H</b> based on the gene-expression features provided"),
                        (neg, "classified the molecular profile as <b>MSS</b>")):
        s = rv.summarize_sample(row, bundle=bundle, entry=entry, members=None, source="manual")
        html = rv.result_card_html(s)
        assert needle in html and "not be interpreted as a medical diagnosis or treatment recommendation" in html
        text = re.sub(r"<[^>]+>", " ", html).lower()
        for w in BANNED:
            assert w not in text, w
        assert f"Predicted: {s['predicted_class']}" in html
    assert "Predicted: MSI-H" in rv.result_card_html(rv.summarize_sample(pos, bundle=bundle, entry=entry, members=None, source="manual"))


def test_extended_models_are_labelled_and_never_called_ferroptosis_classifiers():
    import model_registry as reg
    import predict as prd
    import result_view as rv
    entry = reg.current_best("classification_extended")
    bundle = reg.load_model(entry)
    tc = pd.read_csv(ROOT / "data" / "processed" / "tcga_coadread_features.csv").dropna(subset=["msi_label"]).head(1)
    import extended_data as ed
    X = pd.concat([tc[[g for g in bundle["features"] if g in tc.columns]].reset_index(drop=True),
                   ed.load_matrix("core_plus_mmr", tc["sampleId"])[[g for g in bundle["features"] if g not in tc.columns]].reset_index(drop=True)], axis=1)
    out = prd.classify(bundle, X)
    s = rv.summarize_sample(out.iloc[0], bundle=bundle, entry=entry, members=None, source="upload")
    assert s["model_label"] == "35-gene + MMR Extended Model (39 genes)" and not s["is_ferroptosis_only"]
    assert "not a ferroptosis-only classifier" in rv.result_card_html(s)
    # the 53-feature model keeps its own name
    fake = {**entry, "feature_pipeline": "extended_molecular|raw|none"}
    assert rv.summarize_sample(out.iloc[0], bundle=bundle, entry=fake, members=None, source="upload")["model_label"] == "Extended Molecular Model (53 features)"


# ------------------------------------------------------------------------------------------------ app flows
def test_popup_appears_after_a_successful_classification_with_the_real_result():
    at = _reference_run(_app())
    assert not at.exception, [str(e) for e in at.exception]
    html = _popup_html(at)
    s = at.session_state["latest"]["summary"]
    res = at.session_state["result"]["clf"].iloc[0]
    assert html and f"Predicted: {s['predicted_class']}" in html
    assert s["raw_class"] == res["predicted_class"] and s["probability"] == pytest.approx(float(res["model_probability"]))
    assert f"{100 * s['probability']:.0f}%" in html                          # the shown percentage IS the model probability
    assert "What does this mean?" in html and "not be interpreted as a medical diagnosis" in html
    assert {"View Detailed Results", "Close"} <= {b.label for b in at.button}
    assert "Locked-test accuracy" in [m.label for m in at.metric] and "Test AUC" in [m.label for m in at.metric]


def test_popup_performance_numbers_are_the_registry_values_and_not_the_sample_probability():
    at = _reference_run(_app())
    final = json.load(open(ROOT / "results" / "final_model_registry.json", encoding="utf-8"))["models"][0]
    vals = {m.label: m.value for m in at.metric}
    assert vals["Locked-test accuracy"] == f"{100 * final['locked_test']['accuracy']:.1f}%"
    assert vals["Test AUC"] == f"{final['locked_test']['roc_auc']:.3f}"


def test_the_same_result_component_is_used_for_manual_entry():
    at = _app()
    at.switch_page("pages/analyze.py").run()
    _btn(at, "Start Analysis").click().run()
    at.segmented_control(key="input_mode").set_value("Enter sample").run()
    _btn(at, "Continue").click().run()
    [s for s in at.segmented_control if s.label == "Analysis"][0].set_value("Classification").run()
    _btn(at, "Run analysis").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    html = _popup_html(at)
    assert "Model result" in html and at.session_state["latest"]["summary"]["input_type"] == "Manual entry"
    assert "Predicted:" in html


def test_a_failed_analysis_shows_no_result_popup(monkeypatch):
    import predict as prd

    def boom(bundle, X):
        raise ValueError("This model needs 35 genes; 3 are missing from the input (GPX4, ACSL4, SLC7A11).")
    monkeypatch.setattr(prd, "classify", boom)
    at = _reference_run(_app())
    assert not at.exception
    assert any("Analysis could not be completed" in e.value for e in at.error)
    assert any("3 are missing" in m.value for m in at.markdown)               # the real reason, in plain words
    assert not _popup_html(at) and "latest" not in at.session_state
    assert "View Detailed Results" not in [b.label for b in at.button]


def test_latest_analysis_stays_after_closing_and_reopens_the_detail():
    at = _reference_run(_app())
    summary = at.session_state["latest"]["summary"]
    _btn(at, "Close").click().run()                                            # popup closed: the page body is the detail view
    assert not _popup_html(at)
    _btn(at, "New analysis").click().run()
    card = " ".join(m.value for m in at.markdown if "Latest analysis" in m.value)
    assert "Latest analysis" in card and f"Predicted {summary['predicted_class']}" in card
    assert summary["input_type"] in card and summary["model_label"] in card
    if summary["probability_valid"]:
        assert f"probability {100 * summary['probability']:.0f}%" in card
    assert "TCGA-" not in card                                                  # no sample identifiers in the card
    _btn(at, "Open detailed results").click().run()
    assert at.session_state["step"] == "results" and any(m.label == "Predicted subgroup" for m in at.metric)


# ------------------------------------------------------------------------------------------------ dark mode
def test_dark_mode_loads_and_uses_the_dark_tokens():
    import theme
    at = _app()
    assert at.session_state["theme"] == "light"
    at.segmented_control(key="theme_pick").set_value(theme.LABELS["dark"]).run()
    assert not at.exception and at.session_state["theme"] == "dark"
    css = " ".join(m.value for m in at.markdown if "data-ws-theme" in m.value)
    assert "data-ws-theme='dark'" in css and f"--ws-bg: {theme.TOKENS['dark']['bg']}" in css
    assert theme.TOKENS["dark"]["bg"] not in ("#000000", "#000")                 # not pure black


def test_theme_persists_across_page_navigation_and_via_the_url():
    import theme
    at = _app()
    at.segmented_control(key="theme_pick").set_value(theme.LABELS["dark"]).run()
    for page in ("pages/model_lab.py", "pages/data_explorer.py", "pages/analyze.py"):
        at.switch_page(page).run()
        assert not at.exception and at.session_state["theme"] == "dark", page
    assert _app(query_theme="dark").session_state["theme"] == "dark"             # restored from ?theme= after a refresh
    assert _app().session_state["theme"] == "light"


def test_charts_use_the_active_theme_and_are_readable_in_dark_mode():
    import plotly.graph_objects as go
    import theme
    for name in ("light", "dark"):
        t = theme.TOKENS[name]
        fig = theme.styled_figure(go.Figure(go.Bar(x=[1, 2], y=[1, 2])), 300, name=name)
        assert fig.layout.paper_bgcolor == fig.layout.plot_bgcolor == t["chart_bg"]            # no white panel in dark mode
        assert fig.layout.font.color == t["text"] and fig.layout.xaxis.tickfont.color == t["text"]
        assert fig.layout.yaxis.gridcolor == t["grid"] and fig.layout.legend.font.color == t["text"]
        assert fig.layout.hoverlabel.bgcolor == t["hover_bg"]
        # measurable readability: text, secondary text and chart series against the page background
        assert _contrast(t["text"], t["bg"]) >= 7 and _contrast(t["text2"], t["bg"]) >= 4.5 and _contrast(t["text2"], t["surface"]) >= 4.5
        if name == "dark":      # the new theme is held to the strict thresholds (WCAG: 3:1 graphics, 4.5:1 text)
            assert all(_contrast(c, t["bg"]) >= 3 for c in t["categorical"] + [t["accent"], t["muted"], t["highlight"]])
            assert _contrast(t["accent_text"], t["accent"]) >= 4.5
        else:                   # the original light palette is unchanged; its known limits: series 3-5 are below 3:1 on the page background
            assert _contrast(t["accent"], t["bg"]) >= 4.0 and _contrast(t["accent_text"], t["accent"]) >= 4.0
    assert theme.palette("dark")["accent"] == theme.TOKENS["dark"]["accent"] != theme.palette("light")["accent"]


def test_no_page_hard_codes_chart_colours_or_bypasses_the_themed_chart_helper():
    for page in (ROOT / "app" / "pages").glob("*.py"):
        text = page.read_text(encoding="utf-8")
        assert "st.plotly_chart(" not in text and ".plotly_chart(" not in text, page.name      # all charts go through theme.chart()
        assert not re.findall(r'["\']#[0-9a-fA-F]{6}["\']', text), page.name                    # colours come from the theme tokens


def test_result_popup_is_responsive_and_accessible_by_construction():
    import theme
    css = theme._BASE_CSS
    assert "min(44rem, calc(100vw - 1.5rem))" in css                           # never wider than the viewport
    assert ":focus-visible" in css and "outline: 2px solid var(--ws-accent)" in css
    import result_view as rv
    entry, bundle, X, out = _real_inputs(1)
    html = rv.result_card_html(rv.summarize_sample(out.iloc[0], bundle=bundle, entry=entry, members=None, source="upload"))
    assert 'role="progressbar"' in html and "aria-valuenow" in html and "aria-label" in html      # the meter has a text equivalent
