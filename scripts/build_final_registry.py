"""Builds results/final_model_registry.json: ONE authoritative result per model, with every other result labelled.

Read-only with respect to science: every number is copied from stored artifacts (models/registry.json, models/champions.json,
results/ensemble_lab/*, results/baseline_35_gene/*, results/benchmark/*). Nothing is retrained, re-scored or edited.

    python scripts/build_final_registry.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_registry as reg  # noqa: E402

LAB, BASE, BENCH = ROOT / "results" / "ensemble_lab", ROOT / "results" / "baseline_35_gene", ROOT / "results" / "benchmark"
AUTH = "AUTHORITATIVE_FINAL_RESULT"
import terminology as T  # noqa: E402

CORE_LABEL, MMR_LABEL, EXT_LABEL, WT_LABEL = T.CORE, T.MMR, T.EXTENDED, T.WT
GENES_ADDED = {"core": [], "core_plus_mmr": ["MLH1", "MSH2", "MSH6", "PMS2"], "extended_molecular": None}


def j(p):
    return json.load(open(p, encoding="utf-8"))


def entry_for(version, name):
    return next(e for e in reg._load_registry() if e["version"] == version and e["name"] == name)


def metrics(entry):
    cv, te = entry["cv_metrics"], entry["test_metrics"]
    ext = entry.get("external_metrics") or {}
    pick = lambda d, ks: {k: d[k] for k in ks if k in d}  # noqa: E731
    return (pick(cv, ["accuracy_mean", "accuracy_std", "roc_auc_mean", "roc_auc_std", "balanced_accuracy_mean", "recall_mean",
                      "mcc_mean", "n_folds"]),
            {**pick(te, ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "mcc",
                         "n", "n_positive", "tp", "fp", "tn", "fn"]),
             "accuracy_ci95": te["ci95"]["accuracy"], "roc_auc_ci95": te["ci95"]["roc_auc"]},
            (pick(ext, ["accuracy", "roc_auc", "recall", "n"]) if ext else None))


def provenance_for(fs, summary, lock):
    """Where this model's locked-test number was produced, and proof the stored value equals the logged scoring."""
    import re
    ev = [u for u in lock["test_evaluations"] if re.fullmatch(rf"ensemble_lab_{fs}_\d{{8}}_\d{{6}}", u["tag"])]
    cand = pd.read_csv(LAB / fs / "locked_test_all_candidates.csv").set_index("name")
    final = summary["final"]
    row = cand.loc[final["name"] if final["kind"] == "ensemble" else "single_final"]
    diffs = [abs(float(row[k]) - float(final["test"][k])) for k in ("accuracy", "roc_auc", "recall", "mcc")]
    return {"locked_test_scoring_events_for_this_model": len(ev),
            "locked_test_log_tag": ev[-1]["tag"] if ev else None, "locked_test_scored_at_utc": ev[-1]["at"] if ev else None,
            "stored_final_equals_logged_scoring": bool(max(diffs) < 1e-9), "max_abs_difference_vs_logged_scoring": max(diffs),
            "logged_scoring_file": f"results/ensemble_lab/{fs}/locked_test_all_candidates.csv",
            "lab_summary": f"results/ensemble_lab/{fs}/summary.json", "selection_basis": "cross-validation and the predefined gate only",
            "permutation_canary_auc": final["leakage"].get("permutation_auc_mean"),
            "leakage_checks_passed": final["leakage"].get("passed")}


REQUIRED_MODEL_FIELDS = ("display_name", "version", "feature_set", "n_features", "dataset", "sample_count", "split", "validation_protocol",
                         "cv", "locked_test", "artifact", "production", "provenance", "status", "result_status")
REQUIRED_CV = ("accuracy_mean", "accuracy_std")
REQUIRED_TEST = ("accuracy", "n", "n_positive", "recall", "roc_auc", "f1", "precision", "balanced_accuracy")


def validate_registry(r: dict) -> list[str]:
    """Completeness / consistency problems of the final registry (empty list = frozen-result requirements met)."""
    bad = []
    for m in r["models"]:
        bad += [f"{m['id']}: missing {k}" for k in REQUIRED_MODEL_FIELDS if m.get(k) in (None, "", {})]
        bad += [f"{m['id']}: cv missing {k}" for k in REQUIRED_CV if k not in m["cv"]]
        bad += [f"{m['id']}: locked_test missing {k}" for k in REQUIRED_TEST if k not in m["locked_test"]]
        if not m.get("artifact_exists"):
            bad.append(f"{m['id']}: artifact missing")
        if not m["provenance"]["stored_final_equals_logged_scoring"]:
            bad.append(f"{m['id']}: stored final metrics differ from the logged scoring")
        if m["provenance"]["locked_test_scoring_events_for_this_model"] != 1:
            bad.append(f"{m['id']}: expected exactly one logged locked-test scoring")
    if r["whole_transcriptome"]["status"] != "not_completed" or r["whole_transcriptome"]["metrics"] is not None:
        bad.append("whole transcriptome must be not_completed without metrics")
    bad += [f"missing top-level block {k}" for k in ("external_cohort", "feature_set_provenance", "interval_methods") if not r.get(k)]
    for m in r["models"]:
        if "cv_vs_locked_test_discrimination" not in m.get("observations", {}):
            bad.append(f"{m['id']}: discrimination observation missing")
    d = r["drug_response"]
    bad += [f"drug_response: missing {k}" for k in ("validation_splits", "cv", "locked_test", "artifact", "production") if not d.get(k)]
    return bad


def discrimination_gap(cv, te):
    """Observation only: how the locked-test AUC compares with the cross-validated AUC (stored values; nothing recomputed)."""
    gap = cv["roc_auc_mean"] - te["roc_auc"]
    return {"cv_roc_auc_mean": cv["roc_auc_mean"], "cv_roc_auc_sd": cv["roc_auc_std"], "locked_test_roc_auc": te["roc_auc"],
            "locked_test_roc_auc_interval95": te["roc_auc_ci95"], "gap_cv_minus_test": gap, "gap_in_cv_sd_units": gap / cv["roc_auc_std"],
            "cv_accuracy_mean": cv["accuracy_mean"], "locked_test_accuracy": te["accuracy"],
            "cv_recall_mean": cv.get("recall_mean"), "locked_test_recall": te["recall"],
            "note": "Locked-test AUC is below the cross-validated AUC for every model while accuracy barely moves, because accuracy is dominated "
                    "by the majority class. With 22 MSI-H test patients the AUC interval is wide, so this is reported as an observation "
                    "(possible mild optimism of the cross-validated discrimination, or sampling variation), not as a finding."}


def external_cohort_block(summ):
    """Sample accounting of the external cohort, re-derived from the stored raw and processed files."""
    ext_dir = ROOT / "data" / "external" / "cptac_coad_2019"
    c = pd.read_csv(ext_dir / "clinical_sample_long.csv")
    lab = c[c.clinicalAttributeId == "MSI_STATUS"]
    expr = set(pd.read_csv(ext_dir / "mrna_expression_long.csv").sampleId)
    proc = pd.read_csv(ROOT / "data" / "processed" / "external_cptac_coad_2019_processed.csv")
    tc = pd.read_csv(ROOT / "data" / "processed" / "tcga_coadread_features.csv")
    counts = lab.value.value_counts().to_dict()
    assert len(proc) == int(lab.sampleId.isin(expr).sum()) == int(len(lab)), "external sample accounting no longer reconciles"
    per_set = {fs: f"{s['final']['external']['n_shared_features']} of {s['final']['external']['n_features']}" for fs, s in summ.items()
               if s["final"].get("external") and "n_shared_features" in s["final"]["external"]}
    return {
        "wording": "External-cohort data were evaluated separately from the primary locked-test evaluation and were not used for model "
                   "selection or the primary performance claims.",
        "cohort": "CPTAC-2 colon (Vasaikar et al., Cell 2019); cBioPortal study coad_cptac_2019",
        "sample_accounting": {"raw_samples": int(c.sampleId.nunique()), "with_msi_status_label": int(lab.sampleId.nunique()),
                              "without_msi_status_label": int(c.sampleId.nunique() - lab.sampleId.nunique()),
                              "evaluated": int(len(proc)), "msi_h": int(counts.get("MSI-H", 0)), "mss": int(counts.get("MSS", 0)),
                              "verified_from": ["data/external/cptac_coad_2019/clinical_sample_long.csv",
                                                "data/external/cptac_coad_2019/mrna_expression_long.csv",
                                                "data/processed/external_cptac_coad_2019_processed.csv"]},
        "label_definition": "the cohort's own clinical MSI_STATUS call (MSS / MSI-H); the TCGA training labels are MANTIS >= 0.4, so the two "
                            "label definitions differ",
        "id_overlap_with_tcga": {"sample_ids": len(set(proc.sampleId) & set(tc.sampleId)),
                                 "patient_ids": len(set(c.patientId) & set(tc.patientId))},
        "independence_note": "docs/DATASET.md describes the cohort as independent (different consortium and patient population). This audit "
                             "verified only that no sample or patient IDs overlap with the TCGA table.",
        "protocol": "the model procedure is refit on development data restricted to the genes shared with the cohort, then scored; the "
                    "served models themselves are not scored on the cohort",
        "shared_genes_by_model": per_set,
        "not_used_for": "model selection, the ensemble gate, thresholds or promotion: none of select_best, cv_gate, test_gate or "
                        "evaluate_promotion reads external metrics (checked in the code)",
        "role_in_claims": "supplementary; excluded from the primary performance claims"}


def feature_provenance_block():
    sig = pd.read_csv(ROOT / "data" / "unified_gene_signature.csv")
    cats = sig.category.value_counts().to_dict()
    return {
        "gene_list": f"data/unified_gene_signature.csv ({len(sig)} genes)",
        "wording": "derived from the literature; no gene was chosen using this study's expression data or MSI labels",
        "construction": f"union of {cats.get('core_mechanism', 0)} core-mechanism genes (a published mechanistic review) and "
                        f"{cats.get('crc_prognostic_signature', 0)} genes from four published colorectal-cancer ferroptosis prognostic/"
                        "metastasis signatures; src/ferroptosis.py reads only the literature table (data/ferroptosis_genes_literature.csv)",
        "not_independent_of_tcga": "docs/DATASET.md states that TCGA COADREAD is the cohort used by all four source signature papers, so "
                                   "the panel is literature-derived but not independent of the TCGA cohort (the original selection was "
                                   "for prognosis/metastasis, not MSI)",
        "not_claimed": "that the list was fixed before any modelling: the repository does not prove it (the gene files first appear in "
                       "the 2026-09-06 baseline commit)"}


def interval_methods_block():
    import inspect
    import benchmark as bm
    import regression_benchmark as rb
    bc, cb = inspect.signature(bm.bootstrap_ci).parameters, inspect.signature(rb.cell_line_bootstrap).parameters
    return {
        "classification_locked_test_intervals": {
            "label": "95% percentile-bootstrap interval", "method": "class-stratified nonparametric percentile bootstrap",
            "resamples": bc["n"].default, "seed": bc["seed"].default, "interval": "2.5th to 97.5th percentile of the resampled metric",
            "resampled_unit": "locked-test patients (MSI-H and non-MSI-H resampled separately)", "code": "src/benchmark.py::bootstrap_ci",
            "covers": "test-set sampling variation for a fixed fitted model and fixed predictions; not training variability"},
        "drug_response_locked_test_interval": {
            "label": "95% percentile-bootstrap interval", "method": "cluster (cell-line) nonparametric percentile bootstrap",
            "resamples": cb["n"].default, "seed": cb["seed"].default, "interval": "2.5th to 97.5th percentile",
            "resampled_unit": "whole cell lines with all their drug rows", "code": "src/regression_benchmark.py::cell_line_bootstrap",
            "covers": "variation over the 9 test cell lines for a fixed model"},
        "cv_plus_minus": {"what": "standard deviation (ddof=1) of the per-fold metric across the outer folds (5 folds x 2 repeats = 10)",
                          "note": "descriptive spread, not an interval; folds from repeated cross-validation overlap and are not independent"}}


def main():
    lab, lock_p = j(LAB / "lab_summary.json"), ROOT / "data" / "processed" / "locked_split_9fbc975e.json"
    lock, champs = j(lock_p), reg.champion_history()
    summ = {fs: j(LAB / fs / "summary.json") for fs in ("core", "core_plus_mmr", "extended_molecular")}
    models, others = [], []

    meta = {"core": (CORE_LABEL, "classification", "35 ferroptosis genes"),
            "core_plus_mmr": (MMR_LABEL, "classification_extended", "35 ferroptosis genes + MLH1, MSH2, MSH6, PMS2 (39 genes)"),
            "extended_molecular": (EXT_LABEL, "classification_extended",
                                   "35 ferroptosis + 4 mismatch-repair + 14 immune genes (53 features)")}
    why_map = {
        "core": "serving: the ensemble passed the predefined ensemble gate (CV, balanced accuracy, recall, stability, complementarity, "
                "locked test, leakage, reproducibility); its CV AUC gain over the single model is only about +0.004",
        "core_plus_mmr": "serving as the optional extended model: the pre-registered one-SE rule picks the simplest candidate within "
                         "1 SE of the best CV AUC",
        "extended_molecular": "NOT serving: benchmarked candidate; its CV AUC is within 1 SE of the 39-gene model, so the one-SE rule "
                              "keeps the simpler model. Promotion would need an explicit researcher decision"}
    for fs, (label, task, desc) in meta.items():
        s = summ[fs]
        e = entry_for(s["registry_entry"]["version"], s["registry_entry"]["name"])
        cv, te, ext = metrics(e)
        champ = champs.get(task) or {}
        serving = champ.get("version") == e["version"] and champ.get("name") == e["name"]
        is_ens = s["final"]["kind"] == "ensemble"
        models.append({
            "id": fs, "display_name": label, "result_status": AUTH, "feature_set": desc, "n_features": s["n_features"],
            "genes_added_to_ferroptosis_core": GENES_ADDED[fs] if GENES_ADDED[fs] is not None
            else "4 MMR genes + 14 immune genes (see configs/feature_sets.yaml)",
            "is_ferroptosis_only_classifier": fs == "core",
            "model_type": "ensemble" if is_ens else "single",
            "model_name": s["final"]["name"] if is_ens else e["name"],
            "ensemble_members": s["final"]["spec"].get("members") if is_ens else None,
            "ensemble_method": s["final"]["spec"].get("method") if is_ens else None,
            "calibration": e["calibration"], "decision_threshold": s["final"].get("decision_threshold"),
            "version": e["version"], "trained_at": e["trained_at"], "artifact": e["path"],
            "artifact_exists": (ROOT / e["path"]).exists(),
            "dataset": f"TCGA COAD/READ, {lock['n_dev'] + lock['n_test']} labelled patients (MSI-H = MANTIS >= 0.4)",
            "split": {"fingerprint": lock["fingerprint"], "n_dev": lock["n_dev"], "n_test": lock["n_test"],
                      "test_msi_h": lock["test_positives"]},
            "validation_protocol": "nested stratified CV (5 outer folds x 2 repeats, inner tuning) on the development set; one locked "
                                   "test; all preprocessing, tuning, calibration, weights and thresholds fit inside training folds",
            "cv": cv, "locked_test": te, "external_cptac2": ext,
            "metric_source": {"registry": f"models/registry.json::{e['name']}::{e['version']}",
                              "summary": f"results/ensemble_lab/{fs}/summary.json"},
            "sample_count": {"labelled_patients": lock["n_dev"] + lock["n_test"], "msi_h_total": lock["dev_positives"] + lock["test_positives"],
                             "development": lock["n_dev"], "locked_test": lock["n_test"], "locked_test_msi_h": lock["test_positives"]},
            "provenance": provenance_for(fs, s, lock),
            "observations": {"cv_vs_locked_test_discrimination": discrimination_gap(cv, te)},
            "status": "completed", "production": {"serving": serving, "slot": task, "reason": why_map[fs]},
            "majority_class_accuracy": lab["majority_baseline"]})

    # ---- comparators and non-authoritative results (never deleted, always labelled)
    b = j(BASE / "baseline_metrics.json")
    others.append({"id": "core_single_logreg_baseline", "display_name": "35-gene single logistic regression (reproduced baseline)",
                   "result_status": "reference_baseline", "cv_accuracy": b["cv_mean"]["accuracy"],
                   "cv_accuracy_std": b["cv_std"]["accuracy"], "locked_test_accuracy": b["locked_test"]["accuracy"],
                   "source": "results/baseline_35_gene/baseline_metrics.json",
                   "note": "Best single model on the same locked split. It was the production model until the ensemble passed the "
                           "lab gate; the ensemble differs from it by about one patient on the locked test."})
    for fs in summ:
        sc = summ[fs]["single_cv"]
        others.append({"id": f"{fs}_single_final_comparator", "display_name": f"{meta[fs][0]}: best single model (calibrated)",
                       "result_status": "comparator", "cv_accuracy": sc["accuracy_mean"],
                       "locked_test_accuracy": summ[fs]["single_test"]["accuracy"],
                       "source": f"results/ensemble_lab/{fs}/summary.json (single_cv / single_test)",
                       "note": "Gate reference used to decide whether an ensemble is adopted."})
        cand = pd.read_csv(LAB / fs / "locked_test_all_candidates.csv")
        top = cand.sort_values("accuracy", ascending=False).iloc[0]
        others.append({"id": f"{fs}_screening_table", "display_name": f"{meta[fs][0]}: all-candidates screening table",
                       "result_status": "screening_not_authoritative", "n_candidates": int(len(cand)),
                       "highest_row": {"name": top["name"], "locked_test_accuracy": float(top["accuracy"])},
                       "source": f"results/ensemble_lab/{fs}/locked_test_all_candidates.csv",
                       "note": "Every candidate was scored once on the locked test for REPORTING. Single-learner rows here are "
                               "UNCALIBRATED; model selection used cross-validation only. The best row of a table is not a result: "
                               "quoting it would be picking after looking."})
    old = [e for e in reg._load_registry() if e["task"] == "classification" and e["version"].startswith("v202609")
           and e["name"] == "LogisticRegression" and e["experiment"] == "exp1_msi_raw_genes"]
    if old:
        others.append({"id": "sept_2026_pipeline", "display_name": "Earlier pipeline (September 2026, 35 genes)",
                       "result_status": "historical_superseded", "n_test": old[-1]["metrics"].get("n_test"),
                       "n_train": old[-1]["metrics"].get("n_train"),
                       "note": "Predates the corrected MSI labels and the locked protocol (different samples, different split). "
                               "Kept for history; not comparable."})
    others.append({"id": "benchmark_single_registrations", "display_name": "Single-model benchmark registrations (35 genes)",
                   "result_status": "superseded_duplicate",
                   "note": "Repeated runs of the same logistic regression before the ensemble lab; v20261007_073737 is the run "
                           "behind the baseline figures."})
    others.append({"id": "whole_transcriptome_pre_correction", "display_name": "Earlier whole-transcriptome baseline",
                   "result_status": "historical_pre_correction_do_not_quote",
                   "source": "results/whole_transcriptome_baseline_results.csv",
                   "note": "Predates the corrected labels and the locked protocol. No figure from it is used anywhere as a current "
                           "result."})

    # ---- whole transcriptome: not completed
    assert not [e for e in reg._load_registry() if e["task"] == "classification_wt"], \
        "unexpected whole-transcriptome artifact: review before publishing"
    wt = {"id": "whole_transcriptome", "display_name": WT_LABEL, "status": "not_completed", "result_status": "not_completed",
          "reason": "The nested, leakage-safe re-run under the corrected labels was stopped before finishing (runtime). "
                    "No metrics exist or are claimed.",
          "metrics": None, "production": {"serving": False, "reason": "never served"}}

    # ---- drug response
    dr = reg.current_best("dose_response")
    rs = j(BENCH / "regression_summary.json")
    t, ev = dr["test_metrics"], rs["expression_value_added"]
    drug = {"id": "drug_response", "display_name": "Drug-response regression", "result_status": AUTH, "status": "completed",
            "model_type": "single regressor (Extra Trees); NOT a multi-model ensemble", "model_name": dr["name"],
            "version": dr["version"], "artifact": dr["path"], "artifact_exists": (ROOT / dr["path"]).exists(),
            "dataset": dr["dataset"],
            "response_target": f"{(dr.get('extra') or {}).get('target_column')} (ln IC50 of the cell line, in vitro; not a dose)",
            "features": "35 ferroptosis-gene expression values + drug identity", "n_cell_lines": rs["n_cell_lines_total"],
            "validation_protocol": "whole cell lines held out (grouped CV: 5 folds, grouped inner tuning); locked test of unseen "
                                   "cell lines",
            "cv": {"mae_mean": dr["cv_metrics"]["mae_mean"], "mae_std": dr["cv_metrics"]["mae_std"],
                   "baseline_per_drug_mean_mae": rs["baseline_drug_mean_cv"]["mae_mean"],
                   "improvement_over_baseline": ev["champion_vs_drug_mean_mae_gain_mean"] / rs["baseline_drug_mean_cv"]["mae_mean"],
                   "folds_where_expression_helped": f"{ev['folds_where_expression_helped']} of {ev['n_folds']}"},
            "locked_test": {"mae": t["mae"], "baseline_per_drug_mean_mae": t["baseline_drug_mean"]["mae"],
                            "improvement_over_baseline": t["mae_gain_vs_drug_mean"], "n_cell_lines": t["n_cell_lines"],
                            "mae_ci95": t["ci95"]["mae"]},
            "production": {"serving": champs["dose_response"]["version"] == dr["version"], "slot": "dose_response",
                           "reason": champs["dose_response"]["reason"]},
            "ensemble_check": {"status": "completed" if (BENCH / "regression_ensemble_check.json").exists() else "not_completed"},
            "validation_splits": {
                "cross_validation": {
                    "data": f"development cell lines only ({rs['n_cell_lines_total'] - t['n_cell_lines']} lines)",
                    "protocol": "5 folds grouped by cell line (a line is never in training and validation together), grouped inner tuning",
                    "mae": "mean over the 5 folds of the MAE on each fold's held-out cell lines (ln IC50 units)",
                    "paired_gain_over_baseline_per_fold": {"mean": ev["champion_vs_drug_mean_mae_gain_mean"],
                                                           "sd": ev["champion_vs_drug_mean_mae_gain_std"]}},
                "locked_test": {
                    "data": f"{t['n_cell_lines']} cell lines ({t['n_rows']} line-drug rows) held out before any modelling",
                    "relation_to_cv": "DISJOINT from the cross-validation data: the 9 lines are not a subset of the CV result, and the two "
                                      "numbers must not be mixed or presented as one result",
                    "locked_test_scoring_events_logged": len(j(ROOT / "data" / "processed" / "locked_split_gdsc_664e3bda.json")["test_evaluations"])}},
            "strength": "preliminary / weak",
            "honest_summary": "Preliminary, weak predictive result. Most of the accuracy comes from knowing WHICH DRUG it is; the per-drug mean "
                              "alone already scores close to it. Expression adds a modest gain in cross-validation (5.7%, better in 5 of 5 "
                              "folds, 35 development cell lines) and almost none on the separate 9-cell-line locked test (0.2%, well inside "
                              "its 95% interval). Use only as an exploratory computational in-vitro estimate.",
            "metric_source": f"models/registry.json::{dr['name']}::{dr['version']}; results/benchmark/regression_summary.json"}

    # ---- where each commonly-quoted number comes from
    def v(x):
        return round(float(x), 3)
    core_s, mmr_s, ext_s = summ["core"], summ["core_plus_mmr"], summ["extended_molecular"]
    cand = {fs: pd.read_csv(LAB / fs / "locked_test_all_candidates.csv").set_index("name") for fs in summ}
    tt = pd.read_csv(LAB / "core_plus_mmr" / "threshold_tradeoff.csv").set_index("objective")
    S = "screening_not_authoritative"
    recon = [
        {"value": v(b["locked_test"]["accuracy"]), "what": "LOCKED-TEST accuracy of the 35-gene single logistic regression (baseline)",
         "status": "reference_baseline", "source": "results/baseline_35_gene"},
        {"value": v(core_s["final"]["test"]["accuracy"]), "what": "LOCKED-TEST accuracy of the 35-gene ensemble (production core)",
         "status": AUTH, "source": "results/ensemble_lab/core/summary.json"},
        {"value": v(mmr_s["final"]["cv"]["accuracy_mean"]),
         "what": "CROSS-VALIDATION accuracy of the 39-gene +MMR model (a CV figure, not a test figure)", "status": AUTH,
         "source": "results/ensemble_lab/core_plus_mmr/summary.json"},
        {"value": v(mmr_s["final"]["test"]["accuracy"]), "what": "LOCKED-TEST accuracy of the 39-gene +MMR model (calibrated final)",
         "status": AUTH, "source": "results/ensemble_lab/core_plus_mmr/summary.json"},
        {"value": v(cand["core_plus_mmr"].loc["single:logreg_l2", "accuracy"]),
         "what": "locked-test accuracy of the UNCALIBRATED 39-gene logistic regression (screening row)", "status": S,
         "source": "core_plus_mmr/locked_test_all_candidates.csv"},
        {"value": v(ext_s["final"]["cv"]["accuracy_mean"]), "what": "CROSS-VALIDATION accuracy of the 53-feature ensemble",
         "status": AUTH, "source": "results/ensemble_lab/extended_molecular/summary.json"},
        {"value": v(ext_s["final"]["test"]["accuracy"]), "what": "LOCKED-TEST accuracy of the 53-feature ensemble",
         "status": AUTH, "source": "results/ensemble_lab/extended_molecular/summary.json"},
        {"value": v(ext_s["single_test"]["accuracy"]),
         "what": "locked-test accuracy of the calibrated 53-feature single logistic regression (gate comparator)",
         "status": "comparator", "source": "extended_molecular/summary.json (single_test)"},
        {"value": v(cand["extended_molecular"].loc["single:logreg_l2", "accuracy"]),
         "what": "locked-test accuracy of the UNCALIBRATED 53-feature logistic regression (screening row): the source of the "
                 "earlier '0.935'", "status": S, "source": "extended_molecular/locked_test_all_candidates.csv"},
        {"value": v(tt.loc["accuracy", "cv_accuracy"]),
         "what": "CV accuracy of the 39-gene model with the threshold optimised for accuracy: a second, unrelated '0.935'",
         "status": "threshold_trade_off_not_final", "source": "core_plus_mmr/threshold_tradeoff.csv"},
        {"value": v(t["mae"]), "what": "drug-response LOCKED-TEST MAE (ln IC50 units) on 9 cell lines", "status": AUTH,
         "source": "models/registry.json (dose_response)"}]
    why_differ = [
        "Different MODELS on the SAME locked split: the 35-gene single logistic regression (0.871) and the 35-gene ensemble (0.863) "
        "are different models; they differ by about one patient.",
        "Different ROWS of the SAME table: the all-candidates screening table scores every learner UNCALIBRATED (0.928 for the "
        "39-gene and 0.935 for the 53-feature logistic regression); the production models are calibrated with a CV-chosen "
        "threshold (0.914 / 0.921).",
        "CV vs test: 0.930 and 0.931 are cross-validation accuracies of the 39- and 53-feature models; their locked-test "
        "accuracies are 0.914 and 0.921.",
        "Earlier September-2026 registry entries (0.849 etc.) used different samples/labels (pre-correction, 119 test patients) "
        "and are historical.",
        "No result was overwritten: all current results use the same development/test split (fingerprint verified)."]

    reg_json = {
        "schema": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "generated_by": "scripts/build_final_registry.py",
        "rule": "Exactly one entry per model carries result_status = AUTHORITATIVE_FINAL_RESULT: the model chosen by the "
                "predefined, cross-validation-based selection and gate. A number is never authoritative because it is the "
                "highest in a table.",
        "locked_split": {
            "fingerprint": lock["fingerprint"], "n_dev": lock["n_dev"], "n_test": lock["n_test"],
            "test_msi_h": lock["test_positives"], "logged_test_scoring_events": len(lock["test_evaluations"]),
            "audit_trail": {
                "logged_locked_test_scoring_events": len(lock["test_evaluations"]),
                "events": [{"at_utc": u["at"], "tag": u["tag"]} for u in lock["test_evaluations"]],
                "extended_model_crash": {
                    "what_happened": "The first ensemble-lab run crashed in the extended-feature step on a floating-point rounding error "
                                     "(a probability of 1.0000000000000002 rejected by a metric function). The traceback is in "
                                     "results/ensemble_lab_full.log and points at the cross-validation-only calibration study.",
                    "locked_test_accessed_before_crash": False,
                    "fix": "probabilities are clipped to [0, 1] inside benchmark.clf_metrics (numerical safety only; no change to data, "
                           "features, models, selection rules or thresholds)",
                    "rerun": "the second run resumed from cached cross-validation results, selected the same candidate on cross-validation "
                             "alone (auto:soft_equal, as in both run logs), scored the locked test ONCE (logged) and then ran the leakage "
                             "canary for about four hours; the predictions file, summary and registry entry carry later timestamps only "
                             "because they are written at the end of that long step",
                    "unlogged_rescoring_found": False,
                    "stored_final_equals_logged_scoring": all(m["provenance"]["stored_final_equals_logged_scoring"] for m in models),
                    "used_to_tune": False},
                "caveats": [
                    "The core and 39-gene results were produced before the probability-clipping fix and were not recomputed; the fix can "
                    "only change a probability that exceeded 1 by rounding error (about 1e-16).",
                    "Every candidate model was scored once on the locked test for reporting (screening tables); selection used "
                    "cross-validation only.",
                    "The locked cell-line test of the drug-response model was logged twice (two benchmark runs of the same procedure)."]}},
        "production_decision": {
            "core": models[0]["production"], "extended_slot": models[1]["production"],
            "candidate_not_serving": {"model": EXT_LABEL, "reason": models[2]["production"]["reason"]},
            "changed_automatically": False,
            "note": "Production routing follows the stored champion record; nothing in this registry changes it."},
        "external_cohort": external_cohort_block(summ), "feature_set_provenance": feature_provenance_block(),
        "interval_methods": interval_methods_block(),
        "models": models, "whole_transcriptome": wt, "drug_response": drug, "other_results": others,
        "number_reconciliation": {"numbers": recon, "why_numbers_differ": why_differ}}
    ids = [m["id"] for m in models] + [drug["id"]]
    assert len(ids) == len(set(ids)) and all(m["result_status"] == AUTH for m in models)
    assert not [o for o in others if o["result_status"] == AUTH]
    problems = validate_registry(reg_json)
    assert not problems, problems
    (ROOT / "results" / "final_model_registry.json").write_text(json.dumps(reg_json, indent=2, default=float), encoding="utf-8")
    print("wrote results/final_model_registry.json")


if __name__ == "__main__":
    main()
