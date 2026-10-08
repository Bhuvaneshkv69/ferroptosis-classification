import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403
import plotly.graph_objects as go  # noqa: E402
from theme import page_header, section_label, styled_figure, chart, CATEGORICAL, palette  # noqa: E402
import result_view as rv  # noqa: E402

PAL = palette()   # chart colours come from the centralized theme (light or dark)

S = st.session_state
S.setdefault("step", "home")


def reset():
    for k in ("step", "inputs", "analysis", "result", "explain_cache", "explore", "pending_choice", "auto_run", "popup_pending"):
        S.pop(k, None)
    S["step"] = "home"


if not (classifier_ready() or drug_model_ready()):
    page_header("Ferroptosis analysis")
    st.error("No validated model is registered yet. Run `python scripts/run_benchmark.py` (classification) and "
             "`python scripts/run_benchmark.py --task regression` once, or use **Train / Compare** to build them.")
    st.stop()

clf_entry = clf_bundle = drug_entry = drug_bundle = None
if classifier_ready():
    clf_entry, clf_bundle = get_production("classification")
if drug_model_ready():
    drug_entry, drug_bundle = get_production("dose_response")
GENES = clf_bundle["features"] if clf_bundle else drug_bundle["genes"]

# Optional extended model (adds mismatch-repair + immune genes). Clearly separate from the core model.
ext_entry = ext_bundle = None
if reg.current_best("classification_extended") is not None:
    ext_entry, ext_bundle = get_production("classification_extended")
EXT_EXTRA = [g for g in ext_bundle["features"] if g not in GENES] if ext_bundle else []
import terminology as T  # noqa: E402
CORE_LABEL = T.WITH_GENES["core"]
# The label and description come from the REGISTERED extended model, so they always match what is loaded.
EXT_FS = ext_entry["feature_pipeline"].split("|")[0] if ext_entry else "extended_molecular"
EXT_LABEL = T.WITH_GENES.get(EXT_FS, "Extended model")
EXT_ADDS = ("mismatch-repair genes" if EXT_FS == "core_plus_mmr" else "mismatch-repair and immune genes")

# =====================================================================================================
# STEP 0 -- HOME
# =====================================================================================================
def stepper(active: int):
    names = ["Sample", "Analysis", "Results"]
    parts = [f"**{i + 1}. {n}**" if i == active else f"<span style='opacity:.45'>{i + 1}. {n}</span>"
             for i, n in enumerate(names)]
    st.markdown("&nbsp;&nbsp;›&nbsp;&nbsp;".join(parts), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def warm_up():
    """Import the explanation library once, in the background, so the first result appears quickly."""
    import threading
    threading.Thread(target=lambda: __import__("shap"), daemon=True).start()
    return True


def run_example():
    """One-click demo: a real TCGA tumour through both analyses."""
    tcga = load_tcga_features()
    tcga = tcga[tcga["msi_label"].notna()].reset_index(drop=True)
    row = tcga[tcga["msi_label"] == "MSI-H"].iloc[[0]] if (tcga["msi_label"] == "MSI-H").any() else tcga.iloc[[0]]
    sid = row["sampleId"].iloc[0]
    base_df = row[GENES].reset_index(drop=True)
    if EXT_EXTRA:
        base_df = pd.concat([base_df, extended_log2([sid], EXT_EXTRA)], axis=1)
    S["inputs"] = {"df": base_df, "ids": [sid], "already_log2": True, "source": "tcga_reference",
                   "truth": row["msi_label"].iloc[0], "native": "rnaseq", "has_ext": bool(EXT_EXTRA)}
    S["pending_choice"] = "Both" if (clf_bundle and drug_bundle) else ("Classification" if clf_bundle else "Drug Response")
    S["step"] = "choose"


if S["step"] == "home":
    warm_up()
    page_header("Ferroptosis analysis",
                "Analyze a molecular sample and explore its computational ferroptosis classification and "
                "drug-response profile.")
    rv.render_latest(S)
    st.write("")
    b1, b2, _ = st.columns([1.2, 1.4, 5])
    if b1.button("Start Analysis", type="primary"):
        S["step"] = "input"
        st.rerun()
    if b2.button("Try an example", help="Runs a real study sample through both analyses so you can see what the "
                                         "results look like."):
        run_example()
        S["auto_run"] = True
        st.rerun()
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Provide a sample**\n\nUpload a CSV, type values in, or pick a reference sample.")
    c2.markdown("**2 · Choose an analysis**\n\nClassification, drug response, or both.")
    c3.markdown("**3 · Explore**\n\nSee what drove the result, compare compounds, export.")
    st.write("")
    c1, c2 = st.columns(2)
    if clf_entry:
        t = clf_entry["test_metrics"]
        c1.caption(f"**Classification** · {T.display(clf_entry)} · checked on {t['n']} held-out patients "
                   f"(AUC {t['roc_auc']:.2f}). Details under *Models*.")
    if drug_entry:
        t = drug_entry["test_metrics"]
        c2.caption(f"**Drug-response regression** · {T.drug_display(drug_entry)} · checked on {t['n_cell_lines']} held-out "
                   f"cell lines (mean error {t['mae']:.2f} ln-units).")
    st.stop()

# =====================================================================================================
# STEP 1 -- INPUT
# =====================================================================================================
if S["step"] == "input":
    stepper(0)
    page_header("Provide a sample", "Expression values for the 35 ferroptosis-related genes.")
    rv.render_latest(S)
    mode = st.segmented_control("Input", ["Upload CSV", "Enter sample", "Reference sample"], default="Upload CSV",
                                label_visibility="collapsed", key="input_mode") or "Upload CSV"
    inputs, problem = None, None
    already_log2 = False

    if mode == "Upload CSV":
        a, b = st.columns([3, 2])
        with b:
            tmpl = val.make_template_csv(GENES)
            st.download_button("Download CSV template", tmpl.to_csv(index=False).encode(), "ferroptosis_template.csv",
                               "text/csv")
            if EXT_EXTRA:
                ext_tmpl = val.make_template_csv(GENES + EXT_EXTRA)
                st.download_button(f"Template for the {EXT_LABEL} ({len(GENES) + len(EXT_EXTRA)} genes)",
                                   ext_tmpl.to_csv(index=False).encode(), "extended_template.csv", "text/csv",
                                   help=f"Adds {EXT_ADDS}. Optional: only needed for the {EXT_LABEL}.")
            ex_df = (2.0 ** load_tcga_features().dropna(subset=["msi_label"]).sample(3, random_state=1)[GENES] - 1.0)
            ex_df.insert(0, "sample_id", ["example_1", "example_2", "example_3"])
            st.download_button("Download an example file (3 real samples)", ex_df.round(2).to_csv(index=False).encode(),
                               "ferroptosis_example.csv", "text/csv",
                               help="Real study samples in the right format, to try the upload.")
            with st.expander("Input options"):
                already_log2 = st.checkbox("My values are already log2-transformed", False,
                                           help="Default: raw RSEM-style expression; the app applies log2(x+1).")
        with a:
            up = st.file_uploader("CSV with one row per sample and one column per gene", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
                v = val.validate_expression_dataframe(df, GENES)
                for w in v["warnings"]:
                    st.warning(w)
                if not v["is_valid"]:
                    for e in v["errors"]:
                        st.error(e)
                else:
                    ids = df[df.columns[0]].astype(str).tolist() if df.columns[0] not in GENES else \
                        [f"row {i + 1}" for i in range(len(df))]
                    extra_present = [g for g in EXT_EXTRA if g in df.columns]
                    inputs = {"df": df[GENES + extra_present], "ids": ids, "already_log2": already_log2,
                              "source": "upload", "truth": None, "native": "rnaseq",
                              "has_ext": bool(EXT_EXTRA) and len(extra_present) == len(EXT_EXTRA)}
                    st.success(f"{len(df)} sample(s) validated.")
                    with st.expander("Preview"):
                        st.dataframe(df.head(5), hide_index=True, width="stretch")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read that file: {e}")

    elif mode == "Enter sample":
        st.caption("Raw expression values (RSEM-style). Fields start at the training cohort's median; hover a field "
                   "for its typical range.")
        ref = reference_stats(tuple(GENES))
        q = st.text_input("Find a gene", "", placeholder="e.g. GPX4")
        vals = {}
        cols = st.columns(5)
        shown = [g for g in GENES if q.strip().lower() in g.lower()]
        for i, g in enumerate(shown):
            with cols[i % 5]:
                vals[g] = st.number_input(g, min_value=0.0, value=round(ref[g]["median"], 2), step=1.0,
                                          help=f"typical range {ref[g]['p5']:.1f} – {ref[g]['p95']:.1f}",
                                          key=f"man_{g}")
        full = {g: (vals[g] if g in vals else st.session_state.get(f"man_{g}", round(ref[g]["median"], 2)))
                for g in GENES}
        if EXT_EXTRA:
            xref = extended_reference_stats(tuple(EXT_EXTRA))
            with st.expander(f"Extra genes for the {EXT_LABEL} (optional)"):
                xc = st.columns(5)
                for i, g in enumerate(EXT_EXTRA):
                    with xc[i % 5]:
                        full[g] = st.number_input(g, min_value=0.0, value=round(xref[g]["median"], 2), step=1.0,
                                                  help=f"typical range {xref[g]['p5']:.1f} - {xref[g]['p95']:.1f}",
                                                  key=f"manx_{g}")
        inputs = {"df": pd.DataFrame([full]), "ids": ["Entered sample"], "already_log2": False, "source": "manual",
                  "truth": None, "native": "rnaseq", "has_ext": bool(EXT_EXTRA)}

    else:
        kind = st.radio("Reference cohort", ["TCGA tumour (classification + drug response)",
                                             "GDSC cell line (drug response)"], horizontal=True)
        if kind.startswith("TCGA"):
            tcga = load_tcga_features()
            tcga = tcga[tcga["msi_label"].notna()].reset_index(drop=True)
            sid = st.selectbox("Sample", tcga["sampleId"].tolist())
            row = tcga[tcga["sampleId"] == sid].iloc[[0]]
            st.caption("Real study sample; its recorded label is hidden until after the analysis. "
                       "Note: it may belong to the model's training data.")
            base_df = row[GENES].reset_index(drop=True)
            if EXT_EXTRA:
                base_df = pd.concat([base_df, extended_log2([sid], EXT_EXTRA)], axis=1)
            inputs = {"df": base_df, "ids": [sid], "already_log2": True, "source": "tcga_reference",
                      "truth": row["msi_label"].iloc[0], "native": "rnaseq", "has_ext": bool(EXT_EXTRA)}
        else:
            if drug_bundle is None:
                st.error("No drug-response model is registered.")
                st.stop()
            g = load_gdsc_processed().drop_duplicates("COSMIC_ID").reset_index(drop=True)
            name = st.selectbox("Cell line", sorted(g["CELL_LINE_NAME"].unique()))
            row = g[g["CELL_LINE_NAME"] == name].iloc[[0]]
            inputs = {"df": row[drug_bundle["genes"]].reset_index(drop=True), "ids": [name], "already_log2": True,
                      "source": "gdsc_reference", "truth": None, "native": "gdsc",
                      "cosmic_id": int(row["COSMIC_ID"].iloc[0]), "cell_line": name}

    st.write("")
    left, right, _ = st.columns([1, 1.4, 6])
    if left.button("Back"):
        reset()
        st.rerun()
    if right.button("Continue", type="primary", disabled=inputs is None,
                    help=None if inputs is not None else "Provide a valid sample first."):
        S["inputs"] = inputs
        S["step"] = "choose"
        st.rerun()
    st.stop()

# =====================================================================================================
# STEP 2 -- CHOOSE ANALYSIS
# =====================================================================================================
if S["step"] == "choose":
    inputs = S["inputs"]
    stepper(1)
    page_header("Choose an analysis", f"{len(inputs['df'])} sample(s) ready.")
    rv.render_latest(S)
    options = ["Classification", "Drug Response", "Both"]
    if inputs["native"] == "gdsc":
        options = ["Drug Response"]
        st.info("A cell-line profile is on the drug-response model's own scale, so only Drug Response applies.")
    avail = [o for o in options if (o == "Classification" and clf_bundle) or (o == "Drug Response" and drug_bundle)
             or (o == "Both" and clf_bundle and drug_bundle)]
    clf_model = "core"
    if ext_bundle is not None and inputs["native"] == "rnaseq" and not S.get("auto_run"):
        st.caption("Classification model")
        pick = st.segmented_control("Classification model", [CORE_LABEL, EXT_LABEL], default=CORE_LABEL,
                                    label_visibility="collapsed", key="clf_model_pick") or CORE_LABEL
        clf_model = "extended" if pick == EXT_LABEL else "core"
        if clf_model == "extended":
            st.info(f"The {EXT_LABEL} adds {EXT_ADDS} to the ferroptosis genes. It is a more "
                    "accurate molecular model, but it is **not** a ferroptosis-only classifier.", icon=":material/info:")
            if not inputs.get("has_ext"):
                st.warning(f"This file does not contain the {len(EXT_EXTRA)} extra genes. Use the extended template, "
                           "or choose the core model.")
    ext_blocked = clf_model == "extended" and not inputs.get("has_ext")
    default_choice = S.get("pending_choice") if S.get("pending_choice") in avail else avail[-1]
    auto = S.pop("auto_run", False)
    if auto:
        choice = default_choice
    else:
        choice = st.segmented_control("Analysis", avail, default=default_choice,
                                      label_visibility="collapsed") or default_choice
        st.caption({"Classification": "Predicts the molecular subgroup (MSI status) from the ferroptosis-gene profile.",
                    "Drug Response": "Estimates in-vitro sensitivity (IC50) of a colorectal cell line with this profile.",
                    "Both": "Runs both analyses on the same sample."}[choice])
        left, right, _ = st.columns([1, 1.6, 6])
        if left.button("Back"):
            S["step"] = "input"
            st.rerun()
    if auto or right.button("Run analysis", type="primary", disabled=ext_blocked):
        try:
            with st.status("Running analysis", expanded=True) as status:
                res = {"choice": choice}
                st.write("Preparing expression values")
                all_cols = [c for c in GENES + EXT_EXTRA if c in inputs["df"].columns]
                Xlog = prd.to_log2(inputs["df"], all_cols, inputs["already_log2"]) if inputs["native"] == "rnaseq" else None
                res["clf_model"] = clf_model
                run_bundle = ext_bundle if clf_model == "extended" else clf_bundle
                run_entry = ext_entry if clf_model == "extended" else clf_entry
                if choice in ("Classification", "Both"):
                    st.write("Scoring with the validated " + ("combined model" if prd.is_ensemble(run_bundle) else "classifier"))
                    res["clf"] = prd.classify(run_bundle, Xlog)
                if choice in ("Drug Response", "Both"):
                    st.write("Preparing the drug-response profile")
                    if inputs["native"] == "gdsc":
                        res["Xg"], res["mapped"] = inputs["df"].astype(float).reset_index(drop=True), False
                    else:
                        res["Xg"], res["mapped"] = prd.harmonize_to_gdsc(Xlog, drug_bundle["genes"]), True
                res["Xlog"] = Xlog
                latest = None
                if "clf" in res:      # the popup summary is built from the REAL outputs before anything is shown
                    members = prd.member_table(run_bundle, Xlog.iloc[[0]]) if prd.is_ensemble(run_bundle) else None
                    summary = rv.summarize_sample(res["clf"].iloc[0], bundle=run_bundle, entry=run_entry, members=members,
                                                  source=inputs["source"], n_samples=len(inputs["df"]))
                    latest = rv.latest_record(summary, inputs=inputs, result=res, analysis=choice, clf_model=clf_model)
                status.update(label="Analysis complete", state="complete")
        except Exception as exc:  # noqa: BLE001
            S.pop("popup_pending", None)     # a failed analysis never shows a result
            st.error("**Analysis could not be completed.**")
            st.write(rv.plain_error(exc))
            st.stop()
        S["result"], S["analysis"] = res, choice
        if latest is not None:
            S["latest"], S["popup_pending"] = latest, True
        log_analysis_session("classification" if choice == "Classification" else
                             "drug_response" if choice == "Drug Response" else "both",
                             {"source": inputs["source"], "n_samples": len(inputs["df"])})
        S["step"], S["explore"] = "results", "Understand this result"
        st.rerun()
    st.stop()

# =====================================================================================================
# STEP 3 -- RESULTS
# =====================================================================================================
inputs, res, choice = S["inputs"], S["result"], S["analysis"]
ids = inputs["ids"]
stepper(2)
page_header("Analysis complete")
top = st.columns([3, 1])
if len(ids) > 1:
    sel = top[0].selectbox("Sample", range(len(ids)), format_func=lambda i: ids[i])
else:
    sel = 0
    top[0].caption(f"Sample: **{ids[0]}**")
if top[1].button("New analysis"):
    reset()
    st.rerun()

MET = metric_help()
if res.get("clf_model") == "extended" and ext_bundle is not None:
    cb, ce, model_label = ext_bundle, ext_entry, EXT_LABEL
else:
    cb, ce, model_label = clf_bundle, clf_entry, CORE_LABEL

if S.pop("popup_pending", False) and "clf" in res:      # the result popup: one component for every input route
    def _summary(i):
        members = prd.member_table(cb, res["Xlog"].iloc[[i]]) if prd.is_ensemble(cb) else None
        return rv.summarize_sample(res["clf"].iloc[i], bundle=cb, entry=ce, members=members, source=inputs["source"],
                                   n_samples=len(ids), sample_index=i)
    rv.show_result_dialog({"n": len(ids), "ids": ids, "summary": _summary, "perf": rv.performance_for(ce)})

# ---------------------------------------------------------------- classification
if "clf" in res:
    r = res["clf"].iloc[sel]
    prob, is_high = float(r["model_probability"]), r["predicted_class"] == cb["pos_label"]
    ens = prd.is_ensemble(cb)
    section_label("Ferroptosis-gene–based classification" if cb is clf_bundle else f"{EXT_LABEL} · classification")
    if cb is not clf_bundle:
        st.caption(f"Uses {EXT_ADDS} in addition to the ferroptosis genes; "
                   "it is not a ferroptosis-only classifier.")
    label = "MSI-High" if is_high else "MSS / MSI-Low"
    c1, c2, c3 = st.columns([2, 1.3, 2])
    c1.metric("Predicted subgroup", label,
              help=("Microsatellite-instability status, predicted from the " + str(len(cb["features"])) +
                    " genes of the " + model_label.lower() + "."))
    calibrated = ce["calibration"] in ("sigmoid", "isotonic")
    c2.metric("MSI-High probability", pct(prob),
              help=("Calibrated probability of MSI-High (" + ce["calibration"] + " scaling)."
                    if calibrated else "Model probability of MSI-High. It is a model score, not a "
                    "verified confidence; see calibration on the Models page."))
    if inputs.get("truth"):
        c3.metric("Study label", inputs["truth"], help="Recorded MSI status of this TCGA sample.")
    tm = ce["test_metrics"]
    who = "The combined model" if ens else "The model"
    st.markdown(f"**What does this mean?** {who} puts this molecular profile in the **{label}** class "
                f"(probability of MSI-High: {pct(prob)}), based on patterns learned from the training data.")
    if ens:
        n_in = int(r["n_members"])
        st.caption(f"Combined model · {n_in} independently trained models contribute to this result.")
    if not is_high:
        st.caption(f"Keep in mind: in held-out testing this model found {tm['tp']:.0f} of "
                   f"{tm['tp'] + tm['fn']:.0f} MSI-High tumours, so a Low result does not rule MSI-High out.")
    st.caption("Why: genes that contributed to this prediction (not evidence that they cause anything).")
    key = (tuple(ids), sel)
    cache = S.setdefault("explain_cache", {})
    if key not in cache:
        with st.spinner("Explaining the prediction"):
            cache[key] = prd.explain_local(cb, res["Xlog"].iloc[sel], top_k=15)
    ex = cache[key]
    show = ex.head(8).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=show["contribution"], y=show["gene"], orientation="h",
        marker_color=[PAL["accent"] if v >= 0 else PAL["muted"] for v in show["contribution"]],
        customdata=np.c_[show["value_log2"], show["z_vs_training"]],
        hovertemplate="<b>%{y}</b><br>contribution %{x:+.3f}<br>log2 value %{customdata[0]:.2f}<br>"
                      "vs training %{customdata[1]:+.1f} SD<extra></extra>"))
    fig.update_layout(xaxis_title="Contribution to model probability (blue: toward MSI-High, grey: away)",
                      yaxis_title=None)
    chart(fig, 300, width="stretch", config={"displayModeBar": False})
    st.caption("Top contributing genes: " + ", ".join(ex["gene"].head(5)) + f" · method: {ex.attrs.get('method', '')}")
    if ens:
        with st.expander("Model agreement"):
            mt = prd.member_table(cb, res["Xlog"].iloc[[sel]])
            from model_zoo import PLAIN_ENGLISH as _PE
            agree = int(mt["supports_high"].sum())
            fig2 = go.Figure(go.Bar(
                x=mt["probability"], y=[_PE.get(m, m) for m in mt["member"]], orientation="h",
                marker_color=[PAL["accent"] if v else PAL["muted"] for v in mt["supports_high"]],
                text=["High ✓" if v else "Low" for v in mt["supports_high"]], textposition="outside",
                hovertemplate="%{y}: probability %{x:.2f}<extra></extra>"))
            fig2.update_layout(xaxis=dict(title="Each model's own MSI-High probability", range=[0, 1.1]),
                               yaxis=dict(autorange="reversed"))
            chart(fig2, 60 + 44 * len(mt), width="stretch",
                            config={"displayModeBar": False})
            st.caption(f"{agree} of {len(mt)} models support MSI-High. Combined result: **{label}**.")

if "clf" in res:      # which production model produced the result, and how that model performed overall
    rv.render_performance_card(rv.performance_for(ce), ce)

# ---------------------------------------------------------------- drug response
if "Xg" in res:
    section_label("Drug response")
    dt = prd.drug_table(drug_bundle)
    default = list(dt["DRUG_NAME"]).index("5-Fluorouracil") if "5-Fluorouracil" in set(dt["DRUG_NAME"]) else 0
    drug = st.selectbox("Select compound (type to search)", dt["DRUG_NAME"].tolist(), index=default,
                        key="drug_choice")
    pred = prd.predict_ln_ic50(drug_bundle, res["Xg"].iloc[[sel]], drug).iloc[0]
    meas = load_gdsc_processed()
    mm = meas[meas["DRUG_NAME"] == drug]
    pctl = float((mm["LN_IC50"] < pred["ln_ic50"]).mean())
    mae = drug_entry["cv_metrics"]["mae_mean"]
    d1, d2, d3 = st.columns([1.4, 1.4, 2])
    d1.metric("Predicted response", f"IC50 ≈ {pred['ic50_uM']:.2f} µM", help="Concentration predicted to halve "
              "cell viability in vitro. Lower = more sensitive.")
    d2.metric("Typical error (MAE)", f"±{mae:.2f} ln-units",
              help=MET["mae"] + f" Cross-validated on whole held-out cell lines (about ×{np.exp(mae):.1f}-fold).")
    d3.markdown(f"This profile is predicted to be **more sensitive than {pctl:.0%}** of the {len(mm)} colorectal "
                f"cell lines measured for **{drug}**.")
    gain = drug_entry["test_metrics"].get("mae_gain_vs_drug_mean")
    st.caption("In-vitro sensitivity of a cancer cell line, not a clinical dose or treatment recommendation. "
               + (f"Most of this prediction reflects the compound itself: on held-out cell lines the model's error was "
                  f"{gain:+.1%} versus simply using the compound's average, so treat cell-line-specific differences "
                  f"as weak evidence." if gain is not None else ""))
    if res.get("mapped"):
        st.warning("Exploratory: this tumour's RNA-seq values were mapped onto the cell-line (microarray) scale by "
                   "quantile matching, which has not been validated.", icon=":material/info:")
    if inputs.get("cell_line"):
        own = mm[mm["COSMIC_ID"] == inputs["cosmic_id"]]
        if len(own):
            lock = [f for f in PROCESSED_DIR.glob("locked_split_gdsc_*.json")]
            tl = set(json.load(open(lock[0]))["test_cell_lines"]) if lock else set()
            st.caption(f"Measured LN(IC50) for {inputs['cell_line']} in GDSC: {own['LN_IC50'].iloc[0]:.2f} "
                       f"({'locked-test cell line' if inputs['cosmic_id'] in tl else 'part of the training data'}).")
    with st.expander("View graph"):
        row = prd.drug_table(drug_bundle).set_index("DRUG_NAME").loc[drug]
        conc = np.logspace(np.log10(row["MIN_CONC"]), np.log10(row["MAX_CONC"]), 120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=conc, y=1 / (1 + conc / pred["ic50_uM"]), mode="lines", name="Predicted (illustrative)",
                                 line=dict(color=PAL["accent"], width=2.5),
                                 hovertemplate="%{x:.3g} µM → viability %{y:.2f}<extra></extra>"))
        fig.add_vline(x=pred["ic50_uM"], line_dash="dash", line_color=PAL["muted"],
                      annotation_text="predicted IC50", annotation_position="top")
        fig.add_trace(go.Scatter(x=np.exp(mm["LN_IC50"]), y=np.full(len(mm), 0.04), mode="markers",
                                 name="Measured IC50, other cell lines", text=mm["CELL_LINE_NAME"],
                                 marker=dict(color=PAL["subtle"], size=7, symbol="line-ns-open"),
                                 hovertemplate="%{text}: IC50 %{x:.3g} µM<extra></extra>"))
        raw_path = PROCESSED_DIR / "gdsc_raw_dose_response_curves.csv"
        if raw_path.exists():
            raw = pd.read_csv(raw_path)
            raw = raw[raw["DRUG_NAME"] == drug]
            if len(raw):
                fig.add_trace(go.Scatter(x=raw["CONC"], y=raw["mean_viability"], mode="markers",
                                         name="Measured viability (GDSC raw)", text=raw["CELL_LINE_NAME"],
                                         marker=dict(color=PAL["highlight"], size=5, opacity=0.6),
                                         hovertemplate="%{text}<br>%{x:.3g} µM → %{y:.2f}<extra></extra>"))
        fig.update_xaxes(type="log", title="Concentration (µM)")
        fig.update_yaxes(title="Relative viability", range=[0, 1.25])
        chart(fig, 340, width="stretch")
        st.caption("The blue curve is an illustrative single-slope curve drawn through the predicted IC50 "
                   "(no slope is predicted). Grey ticks and orange points are real GDSC measurements when available.")

# ---------------------------------------------------------------- guided exploration
st.write("")
section_label("What would you like to explore?")
opts = ["Understand this result", "Explore drug response", "Compare models", "Export analysis"]
explore = st.segmented_control("Explore", opts, default=S.get("explore", opts[0]), label_visibility="collapsed",
                               key="explore_pick") or opts[0]

if explore == "Understand this result":
    if "clf" in res:
        t, e = ce["test_metrics"], ce.get("external_metrics") or {}
        st.markdown(
            f"On **{t['n']}** held-out patients ({t['n_positive']} MSI-High) the model found "
            f"**{t['recall']:.0%}** of MSI-High tumours and correctly ruled out **{t['specificity']:.0%}** of the "
            f"others (AUC {t['roc_auc']:.2f}). "
            + (f"On an independent cohort ({e.get('n')} samples) the AUC was **{e['roc_auc']:.2f}**. " if e else "")
            + "Treat a single prediction as a research hypothesis.")
        if len(ids) > 1:
            tbl = res["clf"].copy()
            tbl.insert(0, "sample", ids)
            tbl["model_probability"] = tbl["model_probability"].round(3)
            st.dataframe(tbl, width="stretch", hide_index=True)
        ex_all = S["explain_cache"][(tuple(ids), sel)]
        st.dataframe(ex_all[["gene", "contribution", "direction", "value_log2", "z_vs_training"]].round(3),
                     width="stretch", hide_index=True)
    else:
        st.info("Run a classification analysis to see an explanation.")
elif explore == "Explore drug response":
    if "Xg" not in res:
        st.info("This analysis did not include drug response. Start a new analysis and choose Drug Response or Both.")
    else:
        allp = prd.predict_all_drugs(drug_bundle, res["Xg"].iloc[[sel]])
        f1, f2 = st.columns([2, 3])
        paths = f1.multiselect("Filter by pathway", sorted(allp["PATHWAY_NAME"].unique()))
        qs = f2.text_input("Search compound or target", "")
        view = allp
        if paths:
            view = view[view["PATHWAY_NAME"].isin(paths)]
        if qs.strip():
            m = view["DRUG_NAME"].str.contains(qs, case=False) | view["PUTATIVE_TARGET"].str.contains(qs, case=False)
            view = view[m]
        view = view.sort_values("ln_ic50")
        fig = go.Figure(go.Scatter(x=view["ln_ic50"], y=view["DRUG_NAME"], mode="markers",
                                   marker=dict(size=7, color=PAL["accent"]), customdata=view[["PATHWAY_NAME", "ic50_uM"]],
                                   hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>IC50 ≈ %{customdata[1]:.3g} µM<extra></extra>"))
        fig.update_layout(xaxis_title="Predicted LN(IC50)  (lower = more sensitive)", yaxis=dict(showticklabels=False))
        chart(fig, 320, width="stretch")
        tab = view[["DRUG_NAME", "PATHWAY_NAME", "PUTATIVE_TARGET", "ic50_uM", "ln_ic50"]].rename(
            columns={"DRUG_NAME": "Compound", "PATHWAY_NAME": "Pathway", "PUTATIVE_TARGET": "Target",
                     "ic50_uM": "Predicted IC50 (µM)", "ln_ic50": "LN(IC50)"}).round(3)
        st.dataframe(tab, width="stretch", hide_index=True, height=300)
        st.caption(f"Typical error about ±{drug_entry['cv_metrics']['mae_mean']:.2f} ln-units. Much of the "
                   "difference between compounds reflects the compounds themselves (see Models → Drug response).")
elif explore == "Compare models":
    lb = bench_csv("model_zoo_cv.csv")
    if lb is None:
        st.info("No benchmark results yet.")
    else:
        lb = lb.sort_values("roc_auc_mean", ascending=False).head(5)
        st.dataframe(pd.DataFrame({"Model": [mz.PLAIN_ENGLISH.get(n, n) for n in lb["name"]],
                                   "CV AUC": lb["roc_auc_mean"].round(3), "CV F1": lb["f1_mean"].round(3),
                                   "Stability": [stability_stars(s) for s in lb["roc_auc_std"]]}),
                     width="stretch", hide_index=True)
        st.page_link("pages/compare_models.py", label="Open the full model comparison",
                     icon=":material/arrow_forward:")
else:
    import json as _json
    payload = {"analysis": choice, "samples": ids, "created": datetime.now(timezone.utc).isoformat(),
               "classification_model": ce["name"] + " " + ce["version"] if ce else None,
               "disclaimer": DISCLAIMER}
    if "clf" in res:
        payload["classification"] = res["clf"].assign(sample=ids).round(4).to_dict(orient="records")
    st.download_button("Download results (JSON)", _json.dumps(payload, indent=2), "ferroptosis_analysis.json",
                       "application/json")
    if "clf" in res:
        st.download_button("Download classification table (CSV)",
                           res["clf"].assign(sample=ids).to_csv(index=False).encode(), "classification.csv",
                           "text/csv")

# ---------------------------------------------------------------- technical details
st.write("")
with st.expander("Technical details"):
    t_model, t_members, t_data, t_val, t_metrics, t_imp = st.tabs(
        ["Model", "Individual models", "Dataset", "Validation", "Metrics", "Feature importance"])
    with t_members:
        if "clf" in res and prd.is_ensemble(cb):
            mt = prd.member_table(cb, res["Xlog"].iloc[[sel]])
            mt["member"] = mt["member"].map(lambda m: mz.PLAIN_ENGLISH.get(m, m))
            st.dataframe(mt.round(3), hide_index=True, width="stretch")
            st.caption(f"Combination method: {cb['spec'].get('method')}. Members are tuned on training data only; "
                       "the combiner learns from out-of-fold predictions.")
        else:
            st.caption("The current classifier is a single model.")
    with t_model:
        if ce:
            st.write(f"**Classifier:** {T.display(ce)} (`{ce['name']}`, {ce['version']})")
            st.write(f"Feature pipeline: `{ce['feature_pipeline']}` · calibration: `{ce['calibration']}`")
            st.json(ce["hyperparameters"], expanded=False)
        if drug_entry:
            st.write(f"**Drug-response regression:** {T.drug_display(drug_entry)} (`{drug_entry['name']}`, "
                     f"{drug_entry['version']})")
            st.json(drug_entry["hyperparameters"], expanded=False)
    with t_data:
        if ce:
            st.write(f"Classification trained on TCGA COAD/READ (`{ce['dataset']}`); label = MSI status from "
                     "the DNA-based MANTIS score (≥ 0.4). Features: 35 ferroptosis genes, log2(RSEM+1).")
        if drug_entry:
            st.write(f"Drug response trained on GDSC2 colorectal cell lines (`{drug_entry['dataset']}`).")
    with t_val:
        sm = bench_json("benchmark_summary.json")
        if sm:
            st.write("Protocol: locked test set → nested cross-validation on the rest → chosen procedure refit → "
                     "locked test once → separate external cohort.")
            st.json(sm["selection"], expanded=False)
            st.json(sm["leakage_checks"], expanded=False)
    with t_metrics:
        if ce:
            cv, tm = ce["cv_metrics"], ce["test_metrics"]
            st.write("**Classification (cross-validation mean ± SD | locked test)**")
            rows = [(k, pm(cv[f"{k}_mean"], cv[f"{k}_std"]), f"{tm[k]:.3f}")
                    for k in ("roc_auc", "pr_auc", "f1", "balanced_accuracy", "mcc", "accuracy", "brier")]
            st.dataframe(pd.DataFrame(rows, columns=["Metric", "Cross-validation", "Locked test"]),
                         width="stretch", hide_index=True)
        if drug_entry:
            cv, tm = drug_entry["cv_metrics"], drug_entry["test_metrics"]
            st.write("**Drug response (regression; never shown as accuracy)**")
            rows = [(k.upper(), pm(cv[f"{k}_mean"], cv[f"{k}_std"]), f"{tm[k]:.3f}") for k in
                    ("mae", "rmse", "r2", "pearson", "spearman")]
            st.dataframe(pd.DataFrame(rows, columns=["Metric", "Cross-validation", "Held-out cell lines"]),
                         width="stretch", hide_index=True)
    with t_imp:
        imp = bench_csv("feature_importance_permutation.csv")
        if imp is not None:
            top = imp.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["importance_mean"], y=top["gene"], orientation="h",
                                   error_x=dict(array=top["importance_std"]), marker_color=PAL["accent"]))
            fig.update_layout(xaxis_title="Drop in AUC when the gene is shuffled (locked test set)")
            chart(fig, 360, width="stretch", config={"displayModeBar": False})
            st.caption("Genes that contributed to the model's predictions overall; not evidence of causation.")
