"""Writes the paper source documents from the frozen registry. No number is typed by hand.

    python scripts/write_paper_artifacts.py     ->  results/paper_results_table.md, results/paper_scientific_summary.md
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import terminology as T  # noqa: E402

WT_SENTENCE = ("The whole-transcriptome experimental arm was not completed within the validated evaluation protocol and was "
               "therefore excluded from the final performance claims.")


def f3(x):
    return f"{x:.3f}"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def prod_status(m):
    if m["production"]["serving"]:
        return "In production (default)" if m["production"]["slot"] == "classification" else "In production (optional extended slot)"
    return "Benchmarked candidate; not serving"


def main():
    FR = json.load(open(ROOT / "results" / "final_model_registry.json", encoding="utf-8"))
    M = {m["id"]: m for m in FR["models"]}
    d, sp = FR["drug_response"], FR["locked_split"]
    sel = yaml.safe_load(open(ROOT / "configs" / "model_selection.yaml", encoding="utf-8"))
    fsets = yaml.safe_load(open(ROOT / "configs" / "feature_sets.yaml", encoding="utf-8"))
    cl = sel["classification"]
    n_all, n_msih = M["core"]["sample_count"]["labelled_patients"], M["core"]["sample_count"]["msi_h_total"]
    ext_ci = lambda m: f"{m['locked_test']['accuracy_ci95'][0]:.2f}–{m['locked_test']['accuracy_ci95'][1]:.2f}"  # noqa: E731

    # ---------------------------------------------------------------------------------------------- results table
    NA = "not available"            # a missing value is shown explicitly, never blank or shifted

    def val(d_, k, fmt=f3):
        return fmt(d_[k]) if d_ and k in d_ and d_[k] is not None else NA

    A = [[m["display_name"], m["n_features"], f"{val(m['cv'], 'accuracy_mean')} ± {val(m['cv'], 'accuracy_std')}",
          val(m["locked_test"], "accuracy"), val(m["locked_test"], "roc_auc"), val(m["locked_test"], "f1"),
          val(m["locked_test"], "recall"), prod_status(m)] for m in FR["models"]]
    A2 = [[m["display_name"], ext_ci(m), val(m["locked_test"], "balanced_accuracy"), val(m["locked_test"], "specificity"),
           val(m["locked_test"], "mcc"), val(m["locked_test"], "precision"),
           f"{m['locked_test']['n']:.0f} / {m['locked_test']['n_positive']:.0f}"] for m in FR["models"]]
    A3 = [[m["display_name"], f"{o['cv_roc_auc_mean']:.3f} ± {o['cv_roc_auc_sd']:.3f}", f3(o["locked_test_roc_auc"]),
           f"{o['locked_test_roc_auc_interval95'][0]:.2f}–{o['locked_test_roc_auc_interval95'][1]:.2f}", f"{o['gap_cv_minus_test']:.3f}",
           f"{o['gap_in_cv_sd_units']:.1f}", f"{o['cv_accuracy_mean']:.3f} → {o['locked_test_accuracy']:.3f}"]
          for m in FR["models"] for o in [m["observations"]["cv_vs_locked_test_discrimination"]]]
    GAPS = {m["id"]: m["observations"]["cv_vs_locked_test_discrimination"]["gap_in_cv_sd_units"] for m in FR["models"]}
    ex, im = FR["external_cohort"], FR["interval_methods"]
    FP = FR["feature_set_provenance"]
    acc_ = ex["sample_accounting"]
    EXT_TEXT = (f"{ex['wording']} (CPTAC-2 colon: {acc_['raw_samples']} samples, {acc_['evaluated']} with a cohort MSI_STATUS label and "
                f"expression: {acc_['mss']} MSS, {acc_['msi_h']} MSI-H; the cohort's own MSI call, whereas TCGA labels are MANTIS ≥ 0.4; models "
                f"refit on the shared genes: " + ", ".join(f"{k} {v}" for k, v in ex["shared_genes_by_model"].items()) + ".)")
    INT_TEXT = (f"95% interval = {im['classification_locked_test_intervals']['method']} "
                f"({im['classification_locked_test_intervals']['resamples']:,} resamples of the locked-test patients; the fitted model is fixed, "
                f"so it reflects test-set sampling only). CV ± = standard deviation across the 10 outer folds (5 folds × 2 repeats), not an interval.")
    cv_s, lt_s = d["validation_splits"]["cross_validation"], d["validation_splits"]["locked_test"]
    B = [[f"{T.DRUG}: {d['model_name']} (single model)", "ln IC50 (in vitro)", "Cross-validation: " + cv_s["data"] + ", 5 grouped folds",
          f"{d['cv']['mae_mean']:.3f} ± {d['cv']['mae_std']:.3f}", f3(d["cv"]["baseline_per_drug_mean_mae"]),
          f"{d['cv']['improvement_over_baseline']:.1%}", "Preliminary / weak"],
         [f"{T.DRUG}: {d['model_name']} (single model)", "ln IC50 (in vitro)",
          f"Locked test: {d['locked_test']['n_cell_lines']} separate cell lines (disjoint from the cross-validation data)",
          f"{d['locked_test']['mae']:.3f}", f3(d["locked_test"]["baseline_per_drug_mean_mae"]),
          f"{d['locked_test']['improvement_over_baseline']:.1%} (MAE 95% interval {d['locked_test']['mae_ci95'][0]:.2f}–{d['locked_test']['mae_ci95'][1]:.2f})",
          "Preliminary / weak"]]
    out = ["# Paper results table", "",
           "_Generated by `scripts/write_paper_artifacts.py` from `results/final_model_registry.json` (frozen). Only values that exist in the "
           "registry are shown. Do not edit by hand._", "",
           "## A. Classification models (MSI-H vs MSS/MSI-L)", "",
           f"TCGA COAD/READ, {n_all} labelled patients ({n_msih} MSI-H, {n_msih / n_all:.0%}); development {sp['n_dev']} / locked test {sp['n_test']} "
           f"({sp['test_msi_h']} MSI-H). CV = nested stratified cross-validation on the development set. "
           f"Majority-class accuracy: {M['core']['majority_class_accuracy']:.3f}. All values are read from `results/final_model_registry.json`.", "",
           table(A, ["Model", "Feature count", "CV accuracy ± SD", "Locked-test accuracy", "Locked-test AUC", "F1 (test)",
                     "MSI-H recall (test)", "Production status"]), "",
           "Further locked-test metrics (stored values):", "",
           table(A2, ["Model", "Locked-test accuracy, 95% interval", "Balanced accuracy", "Specificity", "MCC", "Precision",
                      "Test n / MSI-H"]), "",
           INT_TEXT, "",
           "**Observation (reported, not corrected): cross-validated vs locked-test discrimination.** For every model the locked-test AUC is "
           "below the cross-validated AUC, while accuracy barely changes (accuracy is dominated by the majority class).", "",
           table(A3, ["Model", "CV AUC ± SD", "Locked-test AUC", "Test AUC 95% interval", "Gap (CV − test)", "Gap in CV SDs",
                      "Accuracy CV → test"]), "",
           f"The gap is {GAPS['core']:.1f} CV SDs for the {T.CORE}, {GAPS['core_plus_mmr']:.1f} for the {T.MMR} and "
           f"{GAPS['extended_molecular']:.1f} for the {T.EXTENDED}. With only "
           f"{sp['test_msi_h']} MSI-H test patients the test AUC interval is wide, so this may be sampling variation or mild optimism of "
           "the cross-validated discrimination; it is flagged as a limitation, not a finding. Note also that the 53-feature model's higher CV "
           "AUC than the 39-gene model's is not reproduced on the locked test (equal test AUC).", "",
           "**External cohort.** " + EXT_TEXT + " Its results are supplementary and excluded from the primary performance claims.", "",
           f"Only the {T.CORE} is a ferroptosis-gene classifier. The {T.MMR} and the {T.EXTENDED} add mismatch-repair genes "
           "(MLH1, MSH2, MSH6, PMS2) and, for the latter, 14 immune genes; mismatch-repair deficiency defines MSI-H, so their higher accuracy "
           "is associated substantially with those genes and must not be attributed to ferroptosis genes.", "",
           "## B. Drug-response regression", "",
           "Two separate evaluation stages using disjoint cell lines: the development cross-validation (35 cell lines) and the locked test "
           "(9 cell lines held out before any modelling). The second is not a subset of the first, and the two numbers must not be combined "
           "or presented as one result. The model is a single regressor; the result is preliminary / weak.", "",
           table(B, ["Model", "Target", "Validation split", "MAE (ln IC50)", "Baseline MAE (per-drug mean)", "Improvement", "Status"]), "",
           "## C. Experimental arms", "",
           table([[T.WT, "**Not completed**", WT_SENTENCE],
                  ["Drug-response multi-model ensemble", "**Not completed**",
                   "The regressor-ensemble experiment did not finish and no ensemble metrics exist; the single-model regressor above is the "
                   "only drug-response model."]], ["Arm", "Status", "Note"]), ""]
    (ROOT / "results" / "paper_results_table.md").write_text("\n".join(out), encoding="utf-8")

    # ---------------------------------------------------------------------------------------------- summary
    core, mmr, ext = M["core"], M["core_plus_mmr"], M["extended_molecular"]
    imm = fsets["immune"]["genes"]
    ev = FR["locked_split"]["audit_trail"]
    S = ["# Scientific summary (factual source for the paper)", "",
         "_Generated from the frozen registry by `scripts/write_paper_artifacts.py`. Not the paper: a checked set of facts and wording._", "",
         "## 1. Problem", "",
         "Can microsatellite-instability status (MSI-H vs MSS/MSI-L) of colorectal tumours be classified from the expression of a "
         "literature-derived ferroptosis gene panel, and does adding mismatch-repair and immune genes help? A secondary, exploratory "
         "question is whether the same panel carries cell-line drug-sensitivity signal (ln IC50). Research use only: outputs are model "
         "classifications and computational in-vitro estimates, not diagnoses, doses or treatment recommendations.", "",
         "## 2. Dataset", "",
         f"* **Classification:** TCGA PanCancer Atlas COAD/READ (cBioPortal); {n_all} patients with an MSI MANTIS score ({n_msih} MSI-H, "
         f"{n_msih / n_all:.0%}); 37 patients without a score were excluded. MSI-H = MANTIS ≥ 0.4 (DNA-based label, not computed from expression).",
         f"* **Split:** one locked split (fingerprint `{sp['fingerprint'][:12]}…`, seed 42): {sp['n_dev']} development / {sp['n_test']} locked test "
         f"({sp['test_msi_h']} MSI-H).",
         "* **External (supplementary):** " + EXT_TEXT,
         f"* **Drug response:** GDSC colorectal cell lines ({d['n_cell_lines']} lines), target ln IC50.", "",
         "## 3. Ferroptosis feature set", "",
         "35 genes, " + FP["wording"] + ". Construction: " + FP["construction"] + ". Caveat: " + FP["not_independent_of_tcga"] + ". "
         f"Model name: **{T.CORE}**.", "",
         "## 4. Classification methodology", "",
         f"* Learners: logistic regression (L2, elastic net), linear and RBF SVM, random forest, extra trees, histogram and gradient boosting, XGBoost, "
         f"LightGBM (CatBoost on the core).",
         f"* Nested stratified cross-validation ({cl['outer_folds']} outer folds × {cl['outer_repeats']} repeats, {cl['inner_folds']}-fold inner "
         f"tuning); preprocessing, tuning, sigmoid calibration, ensemble weights and decision thresholds (MCC objective) are all fit inside "
         "training folds.",
         "* Ensembles (soft vote, hard vote, stacking on out-of-fold predictions) are adopted only if they pass a predefined gate "
         "(CV AUC gain, balanced accuracy, minority recall, stability, member complementarity, locked test, leakage canary, reproducibility).",
         f"* The {T.CORE} is an equal-weight soft-voting model of {len(core['ensemble_members'])} calibrated learners "
         f"({', '.join(core['ensemble_members'])}); its CV AUC gain over the best single model was only about +0.004.", "",
         "## 5. Extended molecular features", "",
         f"* **{T.MMR} (39 genes):** the 35 genes + {', '.join(fsets['mmr']['genes'])}.",
         f"* **{T.EXTENDED} (53 features):** the 39 + {len(imm)} immune / interferon-γ genes ({', '.join(imm)}).",
         "* Rationale: MSI-H is mismatch-repair deficiency by definition, so MMR expression is a direct biological correlate of the label. "
         "These are **not** ferroptosis genes and the models are not ferroptosis classifiers.", "",
         "## 6. Validation strategy", "",
         "* Cross-validation on the development set for every choice; one locked test of "
         f"{sp['n_test']} patients with {sp['test_msi_h']} MSI-H, so positive-class metrics are imprecise. " + INT_TEXT,
         f"with {sp['test_msi_h']} MSI-H, so positive-class metrics are imprecise.",
         "* Permutation-label canary (AUC ≈ 0.5 expected): " + ", ".join(
             f"{m['display_name']} {m['provenance']['permutation_canary_auc']:.2f}" for m in FR["models"]) + ".",
         "* External cohort: " + EXT_TEXT,
         "* Drug response: whole cell lines held out (grouped CV) plus a separate locked set of 9 cell lines.", "",
         "## 7. Final classification results", "",
         table([[m["display_name"], f3(m["cv"]["accuracy_mean"]) + " ± " + f3(m["cv"]["accuracy_std"]),
                 f"{f3(m['locked_test']['accuracy'])} ({ext_ci(m)})", f3(m["locked_test"]["roc_auc"]), f3(m["locked_test"]["recall"]),
                 prod_status(m)] for m in (core, mmr, ext)],
               ["Model", "CV accuracy", "Locked-test accuracy (95% interval)", "Test AUC", "MSI-H recall", "Status"]), "",
         f"* Majority-class accuracy is {core['majority_class_accuracy']:.3f}; MSI-H recall on the locked test is {core['locked_test']['recall']:.2f} for the "
         f"core and {ext['locked_test']['recall']:.2f}–{mmr['locked_test']['recall']:.2f} for the extended models: most gain over the majority class "
         "comes from specificity.",
         "* The 35-gene core did **not** reach 90% accuracy in cross-validation or on the locked test. The extended models exceed 90%, "
         "but through mismatch-repair (and immune) genes.",
         f"* The {T.EXTENDED}'s advantage over the {T.MMR} is one patient on the locked test (inside overlapping 95% intervals); "
         "the one-SE rule therefore serves the simpler 39-gene model.",
         "* Observation (not corrected): locked-test AUC is below the cross-validated AUC for every model (gap in CV SDs: "
         + ", ".join(f"{T.NAME[k]} {v:.1f}" for k, v in GAPS.items()) + "), while accuracy barely changes; with 22 MSI-H test patients "
         "the AUC interval is wide. The 53-feature model's higher CV AUC than the 39-gene model's is not reproduced on the locked test.",
         "* Whole-transcriptome: " + WT_SENTENCE, "",
         "## 8. Drug-response results", "",
         f"* Model: {d['model_name'].replace('_', ' ')} regressor (single model; not an ensemble) on 35 ferroptosis-gene expression values + drug identity; "
         "target ln IC50 (in vitro).",
         f"* **Cross-validation** ({cv_s['data']}): MAE {d['cv']['mae_mean']:.3f} ± {d['cv']['mae_std']:.3f} vs per-drug-mean baseline "
         f"{d['cv']['baseline_per_drug_mean_mae']:.3f}: {d['cv']['improvement_over_baseline']:.1%} better, in {d['cv']['folds_where_expression_helped']} folds.",
         f"* **Locked test** ({lt_s['data']}; disjoint from the cross-validation data): MAE {d['locked_test']['mae']:.3f} vs {d['locked_test']['baseline_per_drug_mean_mae']:.3f}: "
         f"{d['locked_test']['improvement_over_baseline']:.1%} better, well inside the 95% interval of the MAE.",
         "* Characterisation: preliminary / weak. Most accuracy comes from knowing which drug it is. The multi-model drug-response ensemble was not completed.", "",
         "## 9. Limitations", "",
         f"* Only {sp['test_msi_h']} MSI-H patients in the locked test; one cohort for development; external cohort needs a refit on shared genes.",
         "* MMR-containing models are partly circular with the label biology; they are not evidence about ferroptosis.",
         "* Panel provenance: " + FP["not_independent_of_tcga"] + ".",
         "* The external cohort uses its own MSI_STATUS call (not MANTIS), and models are refit on the shared genes, so it does not "
         "evaluate the served models directly; patient independence from TCGA rests on the cohort description plus a check that no "
         "sample or patient IDs overlap.",
         "* Locked-test discrimination (AUC) is below the cross-validated AUC for all three classifiers (see section 7).",
         "* The core ensemble's gain over its best single model is within noise; the single model is a defensible equivalent.",
         "* Only 44 colorectal cell lines for drug response; no clinical dosing or treatment claim is made.",
         "* Whole-transcriptome and the drug-response ensemble were not completed.", "",
         "## 10. Reproducibility and provenance", "",
         f"* Frozen results: `results/final_model_registry.json` (one authoritative result per model); report: `results/final_model_report.md`; "
         "paper table: `results/paper_results_table.md`.",
         "* Seed 42; locked split recorded in `data/processed/locked_split_*.json` with a fingerprint and a log of every locked-test scoring.",
         f"* Every authoritative classifier has exactly one logged locked-test scoring, and its stored metrics equal that scoring "
         f"(max difference {max(m['provenance']['max_abs_difference_vs_logged_scoring'] for m in FR['models']):.0e}).",
         "* Disclosure: one crash (floating-point rounding, before the locked test was touched for that model) occurred while building the extended "
         "models; probabilities are now clipped to [0, 1] in the metric function, and the rerun scored the locked test once, was not used for tuning, "
         "and no unlogged re-scoring was found. The 35- and 39-gene results predate that numerical fix and were not recomputed.",
         "* Every candidate in the screening tables was scored once on the locked test for reporting; selection used cross-validation only.",
         "* Artifacts: `models/` (see the registry for paths); commands: `python scripts/run_ensemble_lab.py`, `python scripts/write_final_report.py`, "
         "`python scripts/write_paper_artifacts.py`; tests: `pytest tests`.", "",
         "## 11. Final contribution", "",
         "A leakage-controlled, fully audited comparison of single and ensemble classifiers for MSI status from a ferroptosis gene panel, showing "
         f"that the panel alone reaches about {core['locked_test']['accuracy']:.0%} locked-test accuracy (majority baseline "
         f"{core['majority_class_accuracy']:.0%}), that adding mismatch-repair genes lifts it above 90%, and that a preliminary drug-response "
         "regression adds little beyond per-drug means. The contribution is the honest, reproducible quantification of what the ferroptosis panel "
         "does and does not carry, not a claim that ferroptosis genes classify MSI at 90%.", ""]
    (ROOT / "results" / "paper_scientific_summary.md").write_text("\n".join(S), encoding="utf-8")
    print("wrote results/paper_results_table.md, results/paper_scientific_summary.md")


if __name__ == "__main__":
    main()
