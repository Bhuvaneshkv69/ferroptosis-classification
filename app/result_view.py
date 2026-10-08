"""
The ONE analysis-result component. CSV uploads, manual entry, reference samples and the example all end here, so there
is a single implementation of: the result summary, the result popup, the "Latest analysis" card and the model-
performance card.

Rules enforced here
  * Every shown number comes from the production model's real output or from results/final_model_registry.json.
  * The probability shown is the model's probability of the MSI-H class ("Prediction probability"); it is shown only when
    the production model is probability-calibrated, otherwise it is omitted (never invented).
  * Model agreement is computed from the real per-model predictions of the combined model; single models show none.
  * Model performance (accuracy, AUC...) is kept visibly separate from this sample's prediction.
  * Neutral research language: "Predicted MSI-H" / "Predicted MSS", never positive/negative, safe/dangerous, diagnosis.
"""
from __future__ import annotations

import html as _html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import terminology as T
from theme import section_label

ROOT = Path(__file__).resolve().parent.parent
FINAL_REGISTRY = ROOT / "results" / "final_model_registry.json"

INPUT_TYPES = {"upload": "CSV upload", "manual": "Manual entry", "tcga_reference": "Reference sample",
               "gdsc_reference": "Cell-line reference"}
CLASS_TEXT = {"MSI-H": "MSI-H", "MSS/MSI-L": "MSS / MSI-L"}
DIFFERENCE_HELP = ("Prediction probability describes this individual model output. Accuracy describes how often the model was "
                   "correct across the evaluation dataset. They are not the same thing.")
DISCLAIMER_SHORT = ("This result comes from a computational research model. It should not be interpreted as a medical "
                    "diagnosis or treatment recommendation.")


def display_class(label: str) -> str:
    return CLASS_TEXT.get(label, label)


def explanation_text(is_pos: bool) -> str:
    if is_pos:
        return "The model classified the molecular profile as <b>MSI-H</b> based on the gene-expression features provided."
    return ("The model classified the molecular profile as <b>MSS</b> (the microsatellite-stable / MSI-low group) based on the "
            "gene-expression features provided.")


def probability_is_valid(entry: dict, p) -> bool:
    """A probability is shown only if the production model is probability-calibrated and the value is a real number."""
    try:
        ok = np.isfinite(float(p))
    except (TypeError, ValueError):
        ok = False
    return bool(ok and entry.get("calibration") in ("sigmoid", "isotonic"))


# ---------------------------------------------------------------------------------------------------- summary (pure)
def summarize_sample(row, *, bundle: dict, entry: dict, members: pd.DataFrame | None, source: str, n_samples: int = 1,
                     sample_index: int = 0, now: datetime | None = None) -> dict:
    """Everything the popup / latest card shows for ONE sample, derived from the real model outputs.
    `row` is one row of predict.classify(); `members` is predict.member_table() for that sample (ensembles only)."""
    pos, neg = bundle["pos_label"], bundle["neg_label"]
    pred = str(row["predicted_class"])
    is_pos = pred == pos
    prob = float(row["model_probability"])
    valid = probability_is_valid(entry, prob)
    agreement = None
    if members is not None and len(members):
        import model_zoo as mz
        klass = np.where(members["supports_high"].to_numpy(bool), pos, neg)
        agrees = klass == pred
        agreement = {"agree": int(agrees.sum()), "n": int(len(members)),
                     "members": [{"name": mz.PLAIN_ENGLISH.get(m, m), "klass": display_class(k), "agrees": bool(a),
                                  "probability": float(p)}
                                 for m, k, a, p in zip(members["member"], klass, agrees, members["probability"])]}
    fs = T.feature_set_of(entry)
    now = now or datetime.now(timezone.utc)
    return {"predicted_class": display_class(pred), "raw_class": pred, "is_pos": is_pos,
            "probability": prob if valid else None, "probability_valid": valid,
            "threshold": float(row["threshold"]) if "threshold" in row else None,
            "agreement": agreement, "model_label": T.WITH_GENES[fs], "model_name": T.NAME[fs], "feature_set": fs,
            "is_ferroptosis_only": T.FERROPTOSIS_ONLY[fs], "n_features": len(bundle["features"]),
            "model_kind": "combined model" if agreement else "single model",
            "input_type": INPUT_TYPES.get(source, source), "timestamp": now.isoformat(),
            "n_samples": int(n_samples), "sample_index": int(sample_index)}


def performance_for(entry: dict, registry_path: Path = FINAL_REGISTRY) -> dict | None:
    """Model-performance metrics of the production model from the AUTHORITATIVE registry (None if not recorded)."""
    try:
        reg = json.load(open(registry_path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    m = next((x for x in reg["models"] if x["version"] == entry.get("version")), None)
    if m is None:
        return None
    cv, te = m["cv"], m["locked_test"]
    return {"id": m["id"], "label": T.WITH_GENES[m["id"]], "name": T.NAME[m["id"]], "features": m["feature_set"],
            "n_features": m["n_features"], "cv_accuracy": cv["accuracy_mean"], "cv_accuracy_sd": cv["accuracy_std"],
            "test_accuracy": te["accuracy"], "test_auc": te["roc_auc"], "test_recall": te["recall"],
            "test_n": int(te["n"]), "test_msi_h": int(te["n_positive"]), "is_ferroptosis_only": m["is_ferroptosis_only_classifier"],
            "source": "results/final_model_registry.json"}


def pct1(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------------------------------- html pieces
def result_card_html(s: dict) -> str:
    esc = _html.escape
    meter = ""
    if s["probability_valid"]:
        p = s["probability"]
        tick = f'<div class="tick" style="left:{100 * s["threshold"]:.1f}%" title="decision threshold"></div>' if s.get("threshold") is not None else ""
        pr = f"{100 * p:.0f}%"
        meter = (f'<div class="ws-result-label">Prediction probability <span class="ws-meter-note">(model probability of MSI-H)</span></div>'
                 f'<div class="ws-result-value">{pr}</div>'
                 f'<div class="ws-meter" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{100 * p:.0f}" '
                 f'aria-label="Model probability of MSI-H: {pr}"><div class="fill" style="width:{100 * p:.1f}%"></div>{tick}</div>'
                 f'<div class="ws-meter-note">0% &nbsp;·&nbsp; tick = decision threshold &nbsp;·&nbsp; 100%</div>')
    else:
        meter = (f'<div class="ws-result-label">Predicted class: {esc(s["predicted_class"])}</div>'
                 '<div class="ws-meter-note">Model confidence information is available in Detailed Results.</div>')
    if s["agreement"]:
        a = s["agreement"]
        agree = (f'<div class="ws-result-label">Model agreement</div>'
                 f'<div class="ws-agree">{a["agree"]} of {a["n"]} models agree</div>'
                 f'<div class="ws-meter-note">with the combined result · each model applies its own decision threshold, '
                 f'so this is not a simple vote</div>')
    else:
        agree = '<div class="ws-meter-note" style="margin-top:0.7rem">Single model: there is no model agreement to show.</div>'
    extra = ""
    if not s["is_ferroptosis_only"]:
        extra = ('<div class="ws-note">This model also uses mismatch-repair' + (" and immune" if s["feature_set"] == "extended_molecular" else "")
                 + " genes. It is an extended model, not a ferroptosis-only classifier.</div>")
    return (f'<div class="ws-result" role="group" aria-label="Analysis result">'
            f'<div class="ws-result-kicker">Model result</div>'
            f'<div class="ws-result-class">Predicted: {esc(s["predicted_class"])}</div>'
            f'<div class="ws-result-model">{esc(s["model_label"])} · {esc(s["model_kind"])}</div>'
            f'{meter}{agree}<hr class="ws-divider">'
            f'<div class="ws-explain"><b>What does this mean?</b><br>{explanation_text(s["is_pos"])}</div>'
            f'{extra}<div class="ws-note"><b>Important:</b> {esc(DISCLAIMER_SHORT)}</div></div>')


def members_text(s: dict) -> str:
    """Plain-text list (symbols AND words, never colour alone) of the real per-model predictions."""
    a = s["agreement"]
    lines = [f"- **{m['name']}** — {m['klass']} {'✓ agrees with the combined result' if m['agrees'] else '✗ differs from the combined result'}"
             for m in a["members"]]
    return ("\n".join(lines) + f"\n\n**Combined result: {s['predicted_class']}** ({a['agree']} of {a['n']} models agree). "
            "Each model's class uses its own decision threshold; the combined result averages the models' probabilities and "
            "applies the combined threshold, so it is not a simple majority vote.")


def render_performance(perf: dict | None, *, compact: bool = False):
    """MODEL PERFORMANCE (how the model did on held-out evaluation data); never this sample's probability."""
    st.markdown('<div class="ws-perf-head">Model performance (held-out evaluation, not this sample)</div>', unsafe_allow_html=True)
    if perf is None:
        st.caption("Performance details are not recorded for this model in the authoritative results registry.")
        return
    if compact:
        c1, c2 = st.columns(2)
        c1.metric("Locked-test accuracy", pct1(perf["test_accuracy"]), help=DIFFERENCE_HELP)
        c2.metric("Test AUC", f"{perf['test_auc']:.3f}", help=DIFFERENCE_HELP)
        return
    c = st.columns(4)
    c[0].metric("Validation accuracy", f"{pct1(perf['cv_accuracy'])} ± {100 * perf['cv_accuracy_sd']:.1f}",
                help="Cross-validation accuracy on the development data (mean ± SD across folds). " + DIFFERENCE_HELP)
    c[1].metric("Locked-test accuracy", pct1(perf["test_accuracy"]),
                help=f"Accuracy on {perf['test_n']} held-out patients ({perf['test_msi_h']} MSI-H). " + DIFFERENCE_HELP)
    c[2].metric("Locked-test AUC", f"{perf['test_auc']:.3f}", help="Ranking quality on the held-out patients (0.5 = chance, 1 = perfect). " + DIFFERENCE_HELP)
    c[3].metric("MSI-H recall", pct1(perf["test_recall"]), help="Share of the held-out MSI-H patients that the model found.")
    st.caption(f"These are model performance metrics, not this sample's probability. Source: {perf['source']}.")


def render_performance_card(perf: dict | None, entry: dict):
    """Detailed-results card: which production model, which features, how it performed."""
    section_label("Production model and its performance")
    fs = T.feature_set_of(entry)
    st.markdown(f"**{T.WITH_GENES[fs]}** — " + ("a ferroptosis-gene classifier." if T.FERROPTOSIS_ONLY[fs] else
                "an extended model; it is not a ferroptosis-only classifier."))
    if perf:
        st.caption(f"Features: {perf['features']}")
    render_performance(perf)
    with st.popover("What is the difference?"):
        st.write(DIFFERENCE_HELP)


# ---------------------------------------------------------------------------------------------------- popup
@st.dialog("Analysis complete", width="large")
def _result_dialog(ctx: dict):
    n = ctx["n"]
    idx = 0
    if n > 1:
        idx = st.selectbox("Sample", range(n), format_func=lambda i: ctx["ids"][i] if ctx.get("ids") else f"Sample {i + 1}",
                           key="popup_sample")
    s = ctx["summary"](idx)
    st.markdown(result_card_html(s), unsafe_allow_html=True)
    if s["agreement"]:
        with st.expander("How was this result produced?"):
            st.markdown(members_text(s))
    render_performance(ctx["perf"], compact=True)
    left, right = st.columns(2)
    if left.button("View Detailed Results", type="primary", key="popup_detail"):
        st.session_state["explore"] = "Understand this result"
        st.rerun()
    if right.button("Close", key="popup_close"):
        st.rerun()


def show_result_dialog(ctx: dict):
    _result_dialog(ctx)


# ---------------------------------------------------------------------------------------------------- latest analysis
def latest_record(summary: dict, *, inputs, result, analysis, clf_model) -> dict:
    """What the 'Latest analysis' card keeps (session memory only; no sample identifiers in the visible summary)."""
    return {"summary": summary, "inputs": inputs, "result": result, "analysis": analysis, "clf_model": clf_model}


def latest_line(s: dict) -> str:
    esc = _html.escape
    t = datetime.fromisoformat(s["timestamp"]).astimezone().strftime("%H:%M")
    prob = f" · MSI-H probability {100 * s['probability']:.0f}%" if s["probability_valid"] else ""
    agree = f" · {s['agreement']['agree']} of {s['agreement']['n']} models agree" if s["agreement"] else ""
    many = f" · {s['n_samples']} samples (first shown)" if s["n_samples"] > 1 else ""
    return (f'<div class="ws-latest" role="region" aria-label="Latest analysis"><b>Latest analysis</b> '
            f'<span class="meta">{esc(t)} · {esc(s["input_type"])}</span><br>'
            f'<b>Predicted {esc(s["predicted_class"])}</b>{prob}{agree}{many}<br>'
            f'<span class="meta">{esc(s["model_label"])}</span></div>')


def render_latest(S):
    """Compact 'Latest analysis' card for the top of the Analysis workspace; opens the detailed result."""
    rec = S.get("latest")
    if not rec:
        return
    st.markdown(latest_line(rec["summary"]), unsafe_allow_html=True)
    a, b, _ = st.columns([1.3, 1.2, 5])
    if a.button("Open detailed results", key="latest_open"):
        S["inputs"], S["result"], S["analysis"] = rec["inputs"], rec["result"], rec["analysis"]
        S["step"], S["explore"] = "results", "Understand this result"
        st.rerun()
    if b.button("Show summary", key="latest_popup"):
        S["inputs"], S["result"], S["analysis"] = rec["inputs"], rec["result"], rec["analysis"]
        S["step"], S["popup_pending"] = "results", True
        st.rerun()
    st.write("")


def plain_error(exc: Exception) -> str:
    """A short, plain-language reason for a failed analysis (never a traceback, never a fake result)."""
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, KeyError):
        return ("A gene column that the model needs is missing from the input. Check the column names against the CSV template "
                "and try again.")
    return ("Something went wrong while scoring this sample, so no result is shown. Check that the values are numeric and "
            f"complete, then try again. (Technical note: {type(exc).__name__}.)")
