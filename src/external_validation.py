"""
Genuine external (cross-cohort) validation of the TCGA-trained MSI
classifier on the independent CPTAC-2 Prospective Colon Cancer cohort
(Vasaikar et al., Cell 2019 -- different consortium, different patients,
different RNA-seq batch than TCGA COADREAD).

This module does NOT fit anything. It builds the external validation
dataset from the raw CPTAC snapshot (src/fetch_external_validation.py) and
scores it with the EXACT already-persisted, already-evaluated exp1 models
from models/registry.json -- the same models the Streamlit application
serves. There is no retraining, tuning, or feature selection against this
cohort at any point: doing so would defeat the purpose of an external
validation set.

Scale note: CPTAC's cBioPortal-served RNA-seq values are already on a
log2-like scale (observed range ~1.6-15.8, matching TCGA's log2(RSEM+1)
range), so no additional log2 transform is applied here -- see
data/external/README.md for the value-range check that established this.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluation as ev
import model_registry as reg
from config import RANDOM_SEED

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT / "data" / "external" / "cptac_coad_2019"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
COMPAT_EXPERIMENT = "exp1_msi_raw_genes_cptac28"


def build_external_validation_dataset(log_path=None):
    log = []

    def note(msg):
        log.append(msg)
        print(msg)

    clinical = pd.read_csv(EXTERNAL_DIR / "clinical_sample_long.csv")
    clinical_wide = clinical.pivot_table(
        index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first"
    )
    clinical_wide.columns.name = None
    clinical_wide = clinical_wide.reset_index()
    note(f"Raw: {len(clinical_wide)} CPTAC-2 COAD samples, "
         f"{clinical_wide['MSI_STATUS'].notna().sum()} with an MSI_STATUS label")

    expr_long = pd.read_csv(EXTERNAL_DIR / "mrna_expression_long.csv")
    gene_map = pd.read_csv(EXTERNAL_DIR / "gene_entrez_map.csv")[["entrezGeneId", "hugoGeneSymbol"]]
    expr_long = expr_long.merge(gene_map, on="entrezGeneId", how="left")
    expr_wide = expr_long.pivot_table(index="sampleId", columns="hugoGeneSymbol", values="value", aggfunc="first")
    expr_wide.columns.name = None
    expr_wide = expr_wide.reset_index()
    note(f"Expression matrix: {expr_wide.shape[0]} samples x {expr_wide.shape[1]-1} genes")

    df = clinical_wide.merge(expr_wide, on="sampleId", how="inner")
    note(f"After merging clinical + expression: {len(df)} samples")

    # MSI_STATUS in this cohort is MSS / MSI-H (no MSI-L bucket observed).
    # Match the pipeline's target taxonomy (MSI-H is the positive class
    # everywhere); do not silently drop or relabel anything else.
    valid = df["MSI_STATUS"].isin(["MSS", "MSI-H"])
    note(f"Samples with a usable MSI_STATUS label (MSS or MSI-H): {valid.sum()} / {len(df)} "
         f"(other values found: {sorted(set(df.loc[~valid, 'MSI_STATUS'].dropna()))})")
    df = df[valid].reset_index(drop=True)
    df["msi_label"] = df["MSI_STATUS"]  # kept as CPTAC's own label; see module docstring for the scale note

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "external_cptac_coad_2019_processed.csv"
    df.to_csv(out_path, index=False)
    note(f"Saved processed external validation dataset: {df.shape} -> {out_path}")

    meta = {
        "cohort": "CPTAC-2 Prospective Colon Cancer (Cell 2019)",
        "n_samples": len(df),
        "msi_label_counts": df["msi_label"].value_counts().to_dict(),
    }
    with open(PROCESSED_DIR / "external_cptac_coad_2019_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if log_path:
        Path(log_path).write_text("\n".join(log), encoding="utf-8")

    return df


def _gene_intersection(external_df: pd.DataFrame) -> list[str]:
    """The persisted exp1 models require exactly the 35-gene TCGA panel in a
    fixed column order; CPTAC's RNA-seq V2 profile does not report values
    for every one of those 35 genes (a real platform/profile coverage gap,
    the same kind already documented for TCGA-vs-literature-panel and
    GDSC-vs-array in docs/DATASET.md -- not a bug). Rather than imputing the
    missing ~20% of the feature panel with training-set means (which would
    bias 7/35 inputs for every external sample), a compatible model variant
    is trained on the INTERSECTION of genes available on both platforms --
    a defensible, upfront (not post-hoc-tuned) design decision."""
    with open(PROCESSED_DIR / "tcga_preprocessing_metadata.json") as f:
        tcga_genes = json.load(f)["ferroptosis_genes_used"]
    intersection = [g for g in tcga_genes if g in external_df.columns]
    missing = sorted(set(tcga_genes) - set(intersection))
    print(f"Gene panel available on both TCGA (RNA-seq V2) and CPTAC (RNA-seq V2) profiles: "
          f"{len(intersection)}/{len(tcga_genes)}. Missing from CPTAC: {missing}")
    return intersection


def train_compatible_variant(intersection_genes: list[str]):
    """Trains + persists the intersection-gene variant of Experiment 1 on
    TCGA (same models, same CV/tuning procedure as classification.py) so it
    can be scored on CPTAC without any missing-feature imputation. This is
    training on TCGA ONLY -- the external cohort is never touched here."""
    import classification as clf
    tcga = pd.read_csv(PROCESSED_DIR / "tcga_coadread_processed.csv")
    results_df, fitted_models, split, labels = clf.run_experiment(
        tcga, intersection_genes, "msi_label", "MSI-H", COMPAT_EXPERIMENT
    )
    version = reg.new_version()
    for name, model in fitted_models.items():
        row = results_df[results_df["model"] == name].iloc[0]
        reg.register_model(
            name=name, task="classification", experiment=COMPAT_EXPERIMENT, model_object=model,
            feature_columns=intersection_genes,
            metrics={k: row[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                          "n_train", "n_test"]},
            hyperparameters=row["best_params"], dataset="tcga_coadread_processed", random_seed=RANDOM_SEED,
            version=version, extra={"pos_label": labels["pos_label"], "neg_label": labels["neg_label"],
                                     "note": "intersection-gene variant of exp1, for CPTAC external validation"},
        )
    print(f"Trained + persisted {COMPAT_EXPERIMENT} variant (version {version}): "
          f"internal TCGA held-out ROC-AUC = {dict(zip(results_df['model'], results_df['roc_auc'].round(3)))}")
    return results_df


def run_external_validation(experiment: str = COMPAT_EXPERIMENT):
    """Scores the external CPTAC cohort with every persisted classification
    model for `experiment`, using each model's own required gene columns
    (no fitting happens here). Returns a results DataFrame and saves it,
    plus an ROC-curve figure, exactly like the internal evaluation does."""
    df = pd.read_csv(PROCESSED_DIR / "external_cptac_coad_2019_processed.csv")
    entries = reg.list_models(task="classification", experiment=experiment)
    model_names = sorted({e["name"] for e in entries})

    rows = []
    roc_inputs = {}
    for name in model_names:
        entry = reg.latest_model(task="classification", experiment=experiment, name=name)
        gene_cols = entry["feature_columns"]
        missing = [g for g in gene_cols if g not in df.columns]
        if missing:
            print(f"  {name}: SKIPPED -- {len(missing)} required gene(s) missing from external cohort: {missing}")
            continue

        pipeline = reg.load_model(entry)
        X = df[gene_cols]

        proba = pipeline.predict_proba(X)
        pos_idx = list(pipeline.classes_).index(1)
        y_proba = proba[:, pos_idx]
        pred_encoded = pipeline.predict(X)
        extra = entry.get("extra", {})
        pos_label = extra.get("pos_label", "MSI-H")
        neg_label = extra.get("neg_label", "MSS/MSI-L")

        # Harmonize the negative-class label string to the SAME value on both
        # sides: TCGA's training-time negative label is "MSS/MSI-L", CPTAC's
        # own MSI_STATUS vocabulary calls it "MSS" -- these mean the same
        # thing (not MSI-H) but sklearn's binary metrics need one consistent
        # string, or it (correctly) refuses as "multiclass".
        y_true_raw = df["msi_label"]
        y_true = np.where(y_true_raw == pos_label, pos_label, neg_label)
        y_pred_label = np.where(pred_encoded == 1, pos_label, neg_label)

        metrics = ev.evaluate_classification(y_true, y_pred_label, y_proba, pos_label)
        metrics.update({
            "experiment": f"{experiment}_EXTERNAL_cptac", "model": name,
            "n_external_samples": len(df), "internal_test_roc_auc": entry["metrics"].get("roc_auc"),
            "internal_train_version": entry["version"],
        })
        rows.append(metrics)
        roc_inputs[name] = (y_true, y_proba)

        cm_path = FIG_DIR / f"confusion_matrix_external_cptac_{name}.png"
        ev.plot_confusion_matrix(y_true, y_pred_label, sorted(set(y_true)),
                                  f"External validation (CPTAC-2) - {name}", cm_path)

    if not rows:
        raise RuntimeError(
            f"No model in experiment '{experiment}' could be scored on the external cohort -- "
            f"every model required a gene missing from CPTAC. Run train_compatible_variant() first."
        )

    if roc_inputs:
        ev.plot_roc_curves(roc_inputs, "MSI-H", "ROC - External validation (CPTAC-2 COAD, independent cohort)",
                            FIG_DIR / "roc_curve_external_validation_cptac.png")

    results_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "external_validation_results.csv", index=False)
    print(f"\nSaved external validation results -> {RESULTS_DIR / 'external_validation_results.csv'}")
    return results_df


def main():
    external_df = build_external_validation_dataset(
        log_path=ROOT / "docs" / "review_evidence" / "external_validation_preprocessing_log.txt"
    )
    intersection_genes = _gene_intersection(external_df)
    train_compatible_variant(intersection_genes)
    results = run_external_validation()
    print(results[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]])


if __name__ == "__main__":
    main()
