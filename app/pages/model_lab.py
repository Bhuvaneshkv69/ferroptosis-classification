import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403
import plotly.graph_objects as go  # noqa: E402
from theme import page_header, section_label, styled_figure, chart, palette  # noqa: E402

PAL = palette()   # chart colours come from the centralized theme (light or dark)

LAB = BENCH_DIR.parent / "ensemble_lab"
page_header("Model Lab", "Single models, ensembles and feature sets compared on the same locked protocol")

lab_path = LAB / "lab_summary.json"
if not lab_path.exists():
    st.info("The ensemble lab has not been run yet: `python scripts/run_ensemble_lab.py`.")
    st.stop()
lab = json.load(open(lab_path))
MET = metric_help()
import terminology as T  # noqa: E402
LABELS = dict(T.WITH_GENES)
tbl = pd.DataFrame(lab["table"])
tbl["label"] = tbl["feature_set"].map(LABELS)       # display names come from one place; stored artifacts are untouched

# ------------------------------------------------------------------------------------------ verdict
tgt = lab["target"]
hit = tbl[(tbl["cv_accuracy"] >= tgt) & (tbl["test_accuracy"] >= tgt)]
if len(hit):
    st.success("**90% target achieved** (cross-validation and locked test) by: "
               + "; ".join(LABELS[f] for f in hit["feature_set"])
               + ". Only the ferroptosis core counts as a ferroptosis-gene result; the others use additional genes.")
else:
    st.warning("**90% target not achieved** by any model (rule: cross-validation AND locked-test accuracy ≥ 90%).")
st.caption(f"Majority-class accuracy is {lab['majority_baseline']:.1%}; accuracy alone is easy to inflate when only "
           "16% of patients are MSI-High, so recall, balanced accuracy and MCC are shown too.")

# ------------------------------------------------------------------------------------------ summary table
section_label("Best model per feature set")
best_core = reg.current_best("classification")
best_ext = reg.current_best("classification_extended")
show = pd.DataFrame({
    "Feature set": tbl["label"], "Genes": tbl["n_features"],
    "Final model": tbl.apply(lambda r: ("Ensemble (" + r["final"].split(":")[-1].replace("_", " ") + ")")
                             if r["kind"] == "ensemble" else mz.PLAIN_ENGLISH.get(r["best_single"], r["best_single"]), axis=1),
    "CV acc.": tbl["cv_accuracy"].round(3), "Test acc.": tbl["test_accuracy"].round(3),
    "CV AUC": tbl["cv_auc"].round(3), "Test AUC": tbl["test_auc"].round(3),
    "Test recall": tbl["test_recall"].round(3), "Test MCC": tbl["test_mcc"].round(3),
    "External AUC": tbl["external_auc"].round(3)})
st.dataframe(show, hide_index=True, width="stretch", column_config={
    "CV acc.": st.column_config.NumberColumn(help=MET["accuracy"]), "CV AUC": st.column_config.NumberColumn(help=MET["roc_auc"]),
    "Test recall": st.column_config.NumberColumn(help=MET["sensitivity"]),
    "Test MCC": st.column_config.NumberColumn(help=MET["mcc"])})
for nc in lab.get("not_completed", []):
    st.info(f"**{T.NAME.get(nc['feature_set'], nc.get('label', nc['feature_set']))}: experiment not completed.** No metrics are shown for it, and no "
            "earlier or historical figure is substituted.", icon=":material/hourglass_empty:")
if not (LAB.parent / "benchmark" / "regression_ensemble_check.json").exists():
    st.info("**Drug-response ensemble: not completed.** Only the single-model drug-response champion is available; "
            "its estimates are computational in-vitro values, not doses or treatment recommendations.",
            icon=":material/hourglass_empty:")
c1, c2 = st.columns(2)
if best_core:
    c1.markdown(f"**Current best model (ferroptosis core):** {T.display(best_core)}  \n"
                f"<span style='opacity:.65;font-size:.85rem'>serves the Analysis page by default</span>", unsafe_allow_html=True)
if best_ext:
    c2.markdown(f"**Current best extended model:** {T.display(best_ext)}  \n"
                f"<span style='opacity:.65;font-size:.85rem'>adds {'mismatch-repair and immune genes' if T.feature_set_of(best_ext) == 'extended_molecular' else 'mismatch-repair genes'}: not a ferroptosis-only model</span>",
                unsafe_allow_html=True)
for k, v in lab.get("routing", {}).items():
    st.caption(f"Routing · {k}: " + (json.dumps(v["reason"]) if isinstance(v.get("reason"), list) else str(v.get("reason"))))

# ------------------------------------------------------------------------------------------ per feature set
section_label("Details")
fs = st.selectbox("Feature set", list(tbl["feature_set"]), format_func=lambda k: LABELS[k])
d = LAB / fs
if not (d / "summary.json").exists():
    st.info(f"{LABELS[fs]}: results not available (experiment not completed).")
    st.stop()
summ = json.load(open(d / "summary.json"))
fin = summ["final"]
st.markdown(f"**{LABELS[fs]}** · {summ['n_features']} features · final model: "
            f"**{'ensemble ' + fin['name'] if fin['kind'] == 'ensemble' else mz.PLAIN_ENGLISH.get(summ['best_single'])}**")
if fs in ("extended_molecular", "core_plus_mmr", "whole_transcriptome"):
    st.info("This model uses genes beyond the 35-gene ferroptosis panel. It must not be described as a "
            "ferroptosis-gene classifier.", icon=":material/info:")

t1, t2, t3, t4, t5, t6 = st.tabs(["Base models", "Complementarity", "Ensembles", "Thresholds", "Calibration",
                                  "Validation"])
with t1:
    b = pd.read_csv(d / "base_learners_cv.csv").sort_values("roc_auc_mean", ascending=False)
    st.dataframe(pd.DataFrame({
        "Model": b["name"].map(lambda n: mz.PLAIN_ENGLISH.get(n, n)), "CV AUC": b["roc_auc_mean"].round(3),
        "± SD": b["roc_auc_std"].round(3), "CV accuracy": b["accuracy_mean"].round(3), "F1": b["f1_mean"].round(3),
        "Balanced acc.": b["balanced_accuracy_mean"].round(3), "Recall": b["recall_mean"].round(3),
        "Stability": b["roc_auc_std"].map(stability_stars), " ": np.where(b["selected"], "★ best single", "")}),
        hide_index=True, width="stretch")
    if summ.get("skipped_learners"):
        st.caption("Not evaluated on this feature set: " + ", ".join(summ["skipped_learners"])
                   + " (computational cost; no benefit on the 35-gene core).")
    if summ.get("wt_selection"):
        st.caption("High-dimensional feature selection (inside every training fold): " + summ["wt_selection"]["why"])
with t2:
    cs = pd.read_csv(d / "complementarity_summary.csv")
    dis = pd.read_csv(d / "complementarity_disagreement.csv", index_col=0)
    cor = pd.read_csv(d / "complementarity_proba_corr.csv", index_col=0)
    nm = [mz.PLAIN_ENGLISH.get(n, n) + (f" ({n})" if n.startswith("logreg") else "") for n in dis.index]
    h1, h2 = st.columns(2)
    for col, M, title, zmax in ((h1, dis, "Disagreement rate", None), (h2, cor, "Probability correlation", 1.0)):
        fig = go.Figure(go.Heatmap(z=M.values, x=nm, y=nm, colorscale=[[0, PAL["heat_low"]], [1, PAL["accent"]]],
                                   zmin=0, zmax=zmax, text=np.round(M.values, 2), texttemplate="%{text}",
                                   hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>", showscale=False))
        fig.update_layout(title=title, yaxis=dict(autorange="reversed"))
        chart(fig, 420, container=col, width="stretch", config={"displayModeBar": False})
    st.dataframe(cs.round(3).rename(columns={"mean_proba_corr": "avg probability correlation",
                                              "mean_disagreement": "avg disagreement",
                                              "mean_error_overlap": "avg error overlap"}), hide_index=True, width="stretch")
    st.caption(f"If every error of every model were made by all models, an ensemble could not help. An 'oracle' that is right "
               f"whenever any model is right would score {summ['oracle_accuracy']:.1%}, versus {summ['best_single_accuracy_oof']:.1%} "
               "for the best single model: that gap is the most any ensemble could recover.")
    for r in summ["auto_member_log"]:
        st.caption(f"{mz.PLAIN_ENGLISH.get(r['model'], r['model'])}: {r['decision']} ({r['why']})")
with t3:
    e = pd.read_csv(d / "ensemble_cv.csv").sort_values("roc_auc_mean", ascending=False)
    st.dataframe(pd.DataFrame({
        "Ensemble": e["name"].str.replace(":", " · ").str.replace("_", " "), "Members": e["members"].str.replace("_", " "),
        "CV AUC": e["roc_auc_mean"].round(3), "CV accuracy": e["accuracy_mean"].round(3),
        "Balanced acc.": e["balanced_accuracy_mean"].round(3), "Recall": e["recall_mean"].round(3),
        "Passes CV gate": np.where(e["cv_gate_passed"], "yes", "no")}), hide_index=True, width="stretch")
    sc = summ["single_cv"]
    st.caption(f"Reference: best single model ({mz.PLAIN_ENGLISH.get(summ['best_single'])}, calibrated) has CV AUC "
               f"{sc['roc_auc_mean']:.3f}, accuracy {sc['accuracy_mean']:.3f}, recall {sc['recall_mean']:.3f}.")
    g = summ["gate"]
    if summ["chosen_ensemble_by_cv"]:
        st.markdown(f"**Candidate chosen on cross-validation alone:** {summ['chosen_ensemble_by_cv']} — "
                    + ("**adopted**" if g.get("adopted") else "**not adopted**"))
        for part, title in (("cv", "Cross-validation checks"), ("test", "Locked-test checks (run once)")):
            if g.get(part):
                with st.expander(title):
                    st.dataframe(pd.DataFrame(g[part]), hide_index=True, width="stretch")
        if g.get("leakage"):
            with st.expander("Leakage and reproducibility"):
                st.json(g["leakage"], expanded=True)
    else:
        st.markdown("**No ensemble passed the cross-validation gate: the best single model stays.**")
with t4:
    tr = pd.read_csv(d / "threshold_tradeoff.csv")
    st.dataframe(tr.round(3), hide_index=True, width="stretch")
    st.caption("Each threshold was chosen on cross-validated development predictions only, then applied unchanged to the "
               f"locked test. The pre-defined production criterion is MCC. Accuracy-optimised thresholds usually give up recall.")
with t5:
    p = d / "ensemble_calibration.csv"
    if p.exists():
        st.dataframe(pd.read_csv(p).round(4), hide_index=True, width="stretch")
        st.caption("Brier score (lower is better) and expected calibration error for the reference ensemble.")
    else:
        st.caption("Calibration of the ensemble members was compared on the core and extended feature sets.")
with t6:
    t = fin["test"]
    cols = st.columns(5)
    for col, k, lab_ in zip(cols, ["accuracy", "balanced_accuracy", "recall", "roc_auc", "mcc"],
                            ["Accuracy", "Balanced acc.", "Recall (MSI-High)", "AUC", "MCC"]):
        lo, hi = t["ci95"][k]
        col.metric(lab_, f"{t[k]:.3f}", delta=f"95% interval {lo:.2f} – {hi:.2f}", delta_color="off")
    st.caption(f"Locked test: {t['n']} patients, {t['n_positive']} MSI-High. Cross-validation accuracy "
               f"{fin['cv']['accuracy_mean']:.3f} ± {fin['cv']['accuracy_std']:.3f}.")
    ext = fin.get("external")
    if ext and "metrics" in ext:
        m = ext["metrics"]
        st.markdown(f"**External validation (CPTAC-2, n={ext['n']}):** AUC {m['roc_auc']:.3f}, accuracy {m['accuracy']:.3f} "
                    f"(majority {ext['majority_accuracy']:.3f}), recall {m['recall']:.3f}, using {ext['n_shared_features']} of "
                    f"{ext['n_features']} features available in that cohort.")
    elif ext:
        st.caption("External validation was not possible for this feature set.")
    imp = pd.read_csv(d / "feature_importance_permutation.csv").head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["importance_mean"], y=imp["feature"], orientation="h", marker_color=PAL["accent"],
                           error_x=dict(array=imp["importance_std"])))
    fig.update_layout(xaxis_title="Drop in AUC when the feature is shuffled (locked test set)")
    chart(fig, 360, width="stretch", config={"displayModeBar": False})
    st.caption("These features contributed to the model's predictions; this is not evidence of biological causation.")
