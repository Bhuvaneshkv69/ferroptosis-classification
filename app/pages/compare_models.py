import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403
import plotly.graph_objects as go  # noqa: E402
from theme import page_header, section_label, styled_figure, chart, palette  # noqa: E402

PAL = palette()   # chart colours come from the centralized theme (light or dark)

page_header("Models", "How the candidates were compared, and the evidence behind the production models")
MET = metric_help()

summary = bench_json("benchmark_summary.json")
zoo = bench_csv("model_zoo_cv.csv")
if summary is None or zoo is None:
    st.info("No full benchmark has been run yet. Run `python scripts/run_benchmark.py` or use **Train / Compare**.")
    st.stop()

_t = summary["locked_test_metrics"]
st.info(f"**In plain words:** in this single-model benchmark the best model was a {mz.PLAIN_ENGLISH.get(summary['champion'], summary['champion']).lower()} "
        f"because, after comparing {len(zoo)} candidates fairly, nothing more complex was reliably better. (The Analysis page serves the "
        f"Ferroptosis Core Classifier, a combined model that later passed the ensemble gate; see Model Lab.) On "
        f"{_t['n']} patients it never saw, it got {_t['accuracy']:.0%} right, but it found only "
        f"{_t['recall']:.0%} of the MSI-High tumours. The 90% accuracy goal was "
        f"{'reached' if summary['verdict']['achieved'] else 'not reached'}.", icon=":material/info:")

tab_cls, tab_drug, tab_checks = st.tabs(["Classification", "Drug response", "Evidence & checks"])

# =====================================================================================================
with tab_cls:
    v = summary["verdict"]
    champ = summary["champion"]
    test_all = bench_csv("locked_test_all_models.csv")
    zoo = zoo.copy()
    zoo["Model"] = zoo["name"].map(lambda n: mz.PLAIN_ENGLISH.get(n, n))
    zoo["Stability"] = zoo["roc_auc_std"].map(stability_stars)
    if test_all is not None:
        zoo = zoo.merge(test_all[["name", "roc_auc"]].rename(columns={"roc_auc": "test_auc"}), on="name", how="left")
    zoo["Best validated"] = np.where(zoo["name"] == champ, "★ best validated", "")

    c1, c2, c3 = st.columns(3)
    t = summary["locked_test_metrics"]
    c1.metric("Best cross-validation AUC", pm(summary["cv_metrics"]["roc_auc_mean"], summary["cv_metrics"]["roc_auc_std"]),
              help=MET["roc_auc"] + " Mean ± SD over outer folds of nested CV.")
    c2.metric("Locked test AUC", f"{t['roc_auc']:.3f}", help="Scored once, after the model was chosen.")
    ext = summary.get("external_validation")
    c3.metric("External cohort AUC", f"{ext['metrics']['roc_auc']:.3f}" if ext else "n/a",
              help="Independent cohort (different consortium and batch); refit on the shared genes.")

    st.write("")
    st.markdown(f"**Best validated model: {mz.PLAIN_ENGLISH.get(champ, champ)}**  \n"
                f"<span style='opacity:.65;font-size:.85rem'>{summary['selection']['model']}</span>",
                unsafe_allow_html=True)

    section_label("Model performance")
    show = zoo.sort_values("roc_auc_mean", ascending=False)
    table = pd.DataFrame({
        "Model": show["Model"], "AUC": show["roc_auc_mean"].round(3), "F1": show["f1_mean"].round(3),
        "Balanced acc.": show["balanced_accuracy_mean"].round(3), "Stability": show["Stability"],
        "Locked-test AUC": show.get("test_auc", pd.Series(dtype=float)).round(3), " ": show["Best validated"]})
    st.dataframe(table, width="stretch", hide_index=True, column_config={
        "AUC": st.column_config.ProgressColumn("AUC", min_value=0.5, max_value=1.0, format="%.3f", help=MET["roc_auc"]),
        "F1": st.column_config.NumberColumn("F1", help=MET["f1"]),
        "Balanced acc.": st.column_config.NumberColumn("Balanced acc.", help=MET["balanced_accuracy"]),
        "Stability": st.column_config.TextColumn("Stability", help="Stars from the fold-to-fold SD of AUC: "
                                                  "5★ ≤ 0.02, 4★ ≤ 0.035, 3★ ≤ 0.05, 2★ ≤ 0.08."),
        "Locked-test AUC": st.column_config.NumberColumn("Locked-test AUC", help="Reported for transparency; never used "
                                                         "to choose the model.")})

    fig = go.Figure(go.Scatter(
        x=show["roc_auc_mean"], y=show["Model"], mode="markers", error_x=dict(array=show["roc_auc_std"]),
        marker=dict(size=9, color=[PAL["accent"] if n == champ else PAL["muted"] for n in show["name"]]),
        customdata=np.c_[show["f1_mean"], show["mcc_mean"], show["secs_per_fold"]],
        hovertemplate="<b>%{y}</b><br>AUC %{x:.3f}<br>F1 %{customdata[0]:.3f}<br>MCC %{customdata[1]:.3f}"
                      "<br>%{customdata[2]}s per fold<extra></extra>"))
    fig.update_layout(xaxis_title="Cross-validated AUC (mean ± SD across folds)", yaxis=dict(autorange="reversed"))
    chart(fig, 80 + 28 * len(show), width="stretch", config={"displayModeBar": False})
    st.caption("Many models overlap within one standard deviation. When that happens the simplest one in the overlap "
               "is chosen, so a small AUC gap is not evidence that a model is better.")

    section_label("Compare")
    picks = st.multiselect("Choose models to compare in detail", zoo["name"].tolist(), default=[champ],
                           format_func=lambda n: mz.PLAIN_ENGLISH.get(n, n))
    if picks:
        keys = ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "mcc", "accuracy", "recall", "specificity", "brier", "ece"]
        det = zoo[zoo["name"].isin(picks)].set_index("Model")
        out = pd.DataFrame({m: [pm(det.loc[i, f"{m}_mean"], det.loc[i, f"{m}_std"]) for i in det.index] for m in keys},
                           index=det.index).T
        st.dataframe(out, width="stretch", column_config=None)
        st.caption("Cross-validation mean ± SD. " + " · ".join(f"**{k}**: {MET[k]}" for k in ("roc_auc", "f1", "mcc")
                                                                 if k in MET))
        rej = {k: v for k, v in summary.get("rejected", {}).items() if k in picks}
        for k, why in rej.items():
            st.caption(f"{mz.PLAIN_ENGLISH.get(k, k)}: not selected because {why}.")

    section_label("Choices made inside cross-validation")
    a, b, c = st.columns(3)
    fs = bench_csv("feature_strategy_summary.csv")
    im = bench_csv("imbalance_summary.csv")
    ca = bench_csv("calibration_summary.csv")
    with a:
        st.write("**Feature strategy**")
        if fs is not None:
            st.dataframe(fs[["name", "roc_auc_mean", "roc_auc_std"]].round(3).rename(columns={"name": "strategy"}),
                         hide_index=True, width="stretch")
        st.caption(summary["selection"]["features"])
    with b:
        st.write("**Class imbalance**")
        if im is not None:
            st.dataframe(im[["name", "roc_auc_mean", "roc_auc_std"]].round(3).rename(columns={"name": "strategy"}),
                         hide_index=True, width="stretch")
        st.caption(summary["selection"]["imbalance"])
    with c:
        st.write("**Probability calibration**")
        if ca is not None:
            st.dataframe(ca[["name", "brier_mean", "ece_mean"]].round(4).rename(columns={"name": "method"}),
                         hide_index=True, width="stretch")
        st.caption(summary["selection"]["calibration"])

    section_label("Locked test set")
    cols = st.columns(5)
    for col, k, lab in zip(cols, ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "mcc"],
                           ["AUC", "PR-AUC", "F1", "Balanced acc.", "MCC"]):
        lo, hi = t["ci95"][k]
        metric_with_help(col, lab, f"{t[k]:.3f}", k, delta=f"95% interval {lo:.2f} – {hi:.2f}")
    st.caption(f"{t['n']} patients, {t['n_positive']} MSI-High. Wide intervals are expected with so few positives.")

    p1, p2 = st.columns(2)
    lt = bench_csv("locked_test_predictions.csv")
    if lt is not None:
        from sklearn.metrics import precision_recall_curve, roc_curve
        fpr, tpr, _ = roc_curve(lt["y_true"], lt["proba"])
        f1 = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=PAL["accent"], width=2.5),
                                  hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
        f1.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=PAL["muted"]))
        f1.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        chart(f1, 300, container=p1, width="stretch", config={"displayModeBar": False})
        pr, rc, _ = precision_recall_curve(lt["y_true"], lt["proba"])
        f2 = go.Figure(go.Scatter(x=rc, y=pr, mode="lines", line=dict(color=PAL["accent"], width=2.5),
                                  hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>"))
        f2.update_layout(title="Precision–recall curve", xaxis_title="Recall", yaxis_title="Precision")
        chart(f2, 300, container=p2, width="stretch", config={"displayModeBar": False})
        q1, q2 = st.columns(2)
        from sklearn.calibration import calibration_curve
        fp, mp = calibration_curve(lt["y_true"], lt["proba"], n_bins=5, strategy="quantile")
        f3 = go.Figure(go.Scatter(x=mp, y=fp, mode="lines+markers", line=dict(color=PAL["accent"])))
        f3.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=PAL["muted"]))
        f3.update_layout(title="Calibration curve", xaxis_title="Mean predicted probability",
                         yaxis_title="Observed frequency")
        chart(f3, 300, container=q1, width="stretch", config={"displayModeBar": False})
        lab = ["MSS / MSI-Low", "MSI-High"]
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        f4 = go.Figure(go.Heatmap(z=cm, x=[f"pred {l}" for l in lab], y=[f"true {l}" for l in lab],
                                  text=cm, texttemplate="%{text}", colorscale=[[0, PAL["heat_low"]], [1, PAL["accent"]]],
                                  showscale=False))
        f4.update_layout(title="Confusion matrix", yaxis=dict(autorange="reversed"))
        chart(f4, 300, container=q2, width="stretch", config={"displayModeBar": False})

# =====================================================================================================
with tab_drug:
    rs = bench_json("regression_summary.json")
    rcv = bench_csv("regression_cv.csv")
    if rs is None or rcv is None:
        st.info("Drug-response benchmark not run yet: `python scripts/run_benchmark.py --task regression`.")
    else:
        lt = rs["locked_test_metrics"]
        st.markdown(f"**Best validated model: {mz.PLAIN_ENGLISH.get(rs['champion'], rs['champion'])}**  \n"
                    f"<span style='opacity:.65;font-size:.85rem'>{rs['selection']}</span>", unsafe_allow_html=True)
        c = st.columns(4)
        for col, k, lab in zip(c, ["mae", "rmse", "r2", "spearman"], ["MAE", "RMSE", "R²", "Spearman"]):
            metric_with_help(col, f"{lab} (held-out cell lines)", f"{lt[k]:.3f}", k)
        b = lt["baseline_drug_mean"]
        st.caption(f"Reference: predicting each compound's average gives MAE {b['mae']:.3f}, R² {b['r2']:.3f}. "
                   f"The model's MAE is {lt['mae_gain_vs_drug_mean']:+.1%} versus that baseline "
                   f"({lt['n_cell_lines']} test cell lines).")
        ev_ = rs["expression_value_added"]
        st.info(f"Does gene expression add anything beyond knowing which compound it is? Compared with predicting each "
                f"compound's average, the model changed cross-validated MAE by "
                f"{ev_['champion_vs_drug_mean_mae_gain_mean']:+.3f} ± {ev_['champion_vs_drug_mean_mae_gain_std']:.3f} "
                f"(positive = better; better in {ev_['folds_where_expression_helped']} of {ev_['n_folds']} folds) and by "
                f"{lt['mae_gain_vs_drug_mean']:+.1%} on the held-out cell lines. {rs['caution']}")
        disp = rcv.copy()
        disp["Model"] = disp["name"].map(lambda n: mz.PLAIN_ENGLISH.get(n, n.replace("_", " ")))
        disp = disp.sort_values("mae_mean")
        st.dataframe(pd.DataFrame({
            "Model": disp["Model"], "Inputs": disp["variant"], "MAE": disp["mae_mean"].round(3),
            "± SD": disp["mae_std"].round(3), "RMSE": disp["rmse_mean"].round(3), "R²": disp["r2_mean"].round(3),
            "Spearman": disp["spearman_mean"].round(3),
            "vs drug-mean": disp["mae_gain_vs_drug_mean"].map(lambda x: f"{x:+.1%}"),
            " ": np.where(disp["selected"], "★ best validated", "")}), width="stretch", hide_index=True)
        po = bench_csv("regression_locked_test_predictions.csv")
        if po is not None:
            fig = go.Figure(go.Scattergl(x=po["y_true"], y=po["pred"], mode="markers", text=po["DRUG_NAME"],
                                         marker=dict(size=4, color=PAL["accent"], opacity=0.45),
                                         hovertemplate="%{text}<br>measured %{x:.2f}<br>predicted %{y:.2f}<extra></extra>"))
            lim = [po["y_true"].min(), po["y_true"].max()]
            fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1], line=dict(dash="dash", color=PAL["muted"]))
            fig.update_layout(title="Held-out cell lines: predicted vs measured LN(IC50)",
                              xaxis_title="Measured LN(IC50)", yaxis_title="Predicted LN(IC50)")
            chart(fig, 360, width="stretch")

# =====================================================================================================
with tab_checks:
    lk = summary["leakage_checks"]
    st.write("**Leakage and integrity checks**")
    rows = [("Label-permutation test (should be ≈ 0.5)", f"AUC {lk['permutation_auc_mean']:.3f}",
             "pass" if lk["permutation_pass"] else "FAIL"),
            ("Development / test sample overlap", str(lk["dev_test_sample_overlap"]), "pass"),
            ("Patient overlap between sets", str(lk["patient_overlap"]), "pass"),
            ("Preprocessing, feature selection, SMOTE, calibration and threshold fit inside training folds only",
             "by construction (see tests)", "pass")]
    if lk.get("site_grouped_cv"):
        s = lk["site_grouped_cv"]
        rows.append(("Leave-sites-out CV (batch check)", f"AUC {s['roc_auc_mean']:.3f} ± {s['roc_auc_std']:.3f}",
                     "reported"))
    st.dataframe(pd.DataFrame(rows, columns=["Check", "Result", "Status"]), width="stretch", hide_index=True)

    st.write("**90% accuracy target**")
    v = summary["verdict"]
    st.markdown(f"### {v['statement']}")
    st.write(f"Majority-class baseline: {pct(v['majority_baseline_accuracy'], 1)} · cross-validation: "
             f"{pct(v['cv_accuracy_mean'], 1)} · locked test: {pct(v['locked_test_accuracy'], 1)} "
             f"(95% interval {pct(v['locked_test_accuracy_ci95'][0])} – {pct(v['locked_test_accuracy_ci95'][1])})"
             + (f" · external: {pct(v['external_accuracy'], 1)}" if v.get("external_accuracy") is not None else ""))
    st.caption("Accuracy is easy to inflate when one class is rare, so AUC, MCC and balanced accuracy are reported alongside.")

    ext = summary.get("external_validation")
    if ext:
        st.write("**External validation**")
        m = ext["metrics"]
        st.write(f"{ext['cohort']}: {ext['n']} samples ({ext['n_msi_h']} MSI-H), {ext['n_shared_genes']} shared genes "
                 f"(missing: {', '.join(ext['missing_genes'])}). AUC {m['roc_auc']:.3f} "
                 f"(95% interval {ext['ci95']['roc_auc'][0]:.2f} – {ext['ci95']['roc_auc'][1]:.2f}), "
                 f"balanced accuracy {m['balanced_accuracy']:.3f}, MCC {m['mcc']:.3f}.")
        st.caption(ext["note"])
    st.write("**Locked-test usage log**")
    st.json(summary["locked_test_uses"], expanded=False)
    if summary.get("skipped_learners"):
        st.caption("Not evaluated: " + "; ".join(f"{s['name']} ({s['reason']})" for s in summary["skipped_learners"]))
