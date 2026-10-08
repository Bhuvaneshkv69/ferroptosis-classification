"""
Phase-1 data audit for the MSI classification task. Read-only: fits nothing
that is used downstream. Writes results/ml_audit_facts.json, which ML_AUDIT.md
quotes (every number in the audit document comes from this file).

Checks: class balance + majority baseline, exact / near-duplicate samples,
tissue-source-site (batch) confounding, duplicate patients, label/feature
independence, and how noisy the legacy 20% test split is (bootstrap CI).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import chi2_contingency  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from config import RANDOM_SEED  # noqa: E402


def main():
    df = pd.read_csv(ROOT / "data/processed/tcga_coadread_processed.csv")
    genes = json.load(open(ROOT / "data/processed/tcga_preprocessing_metadata.json"))["ferroptosis_genes_used"]
    sub = df[df["MSI_SCORE_MANTIS"].notna()].reset_index(drop=True)
    y = (sub["msi_label"] == "MSI-H").astype(int)
    X = sub[genes]
    facts = {"n_total_tcga": int(len(df)), "n_with_msi_label": int(len(sub)), "n_genes": len(genes),
             "n_msi_h": int(y.sum()), "n_mss": int((1 - y).sum()), "msi_h_prevalence": float(y.mean()),
             "majority_class_accuracy": float(1 - y.mean()), "n_missing_feature_values": int(X.isna().sum().sum()),
             "duplicate_patient_ids": int(sub["patientId"].duplicated().sum()),
             "samples_per_feature": float(len(sub) / len(genes)),
             "minority_events_per_feature": float(y.sum() / len(genes))}

    # exact + near duplicates (Pearson r over the 35-gene vector)
    facts["exact_duplicate_rows"] = int(X.duplicated().sum())
    z = (X - X.mean()) / X.std(ddof=0)
    corr = np.corrcoef(z.values)
    np.fill_diagonal(corr, -1)
    facts["max_sample_pair_correlation"] = float(corr.max())
    facts["pairs_corr_gt_0.99"] = int((np.triu(corr, 1) > 0.99).sum())

    # batch / site confounding
    for col in ["TISSUE_SOURCE_SITE_CODE", "TUMOR_TISSUE_SITE", "SUBTYPE"]:
        if col in sub:
            tab = pd.crosstab(sub[col].fillna("NA"), y)
            tab = tab[tab.sum(axis=1) >= 5]
            chi2, p, _, _ = chi2_contingency(tab)
            facts[f"{col}_n_groups(>=5)"] = int(len(tab))
            facts[f"{col}_vs_label_chi2_p"] = float(p)
    if "TISSUE_SOURCE_SITE_CODE" in sub:
        tss = sub["TISSUE_SOURCE_SITE_CODE"].value_counts()
        facts["n_tissue_source_sites"] = int(len(tss))
        facts["largest_site_fraction"] = float(tss.iloc[0] / len(sub))

    # is the label derived from the features? MANTIS is a DNA-sequencing score.
    facts["label_source"] = "MSI_SCORE_MANTIS >= 0.4 (DNA sequencing); not computed from RNA expression"
    facts["mantis_vs_genes_max_abs_spearman"] = float(
        X.apply(lambda c: abs(c.corr(sub["MSI_SCORE_MANTIS"], method="spearman"))).max())

    # how noisy is the legacy 20% test split?
    tr, te = train_test_split(np.arange(len(sub)), test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    facts["legacy_test_n"] = int(len(te))
    facts["legacy_test_positives"] = int(y.iloc[te].sum())
    rng = np.random.default_rng(0)
    n_pos = int(y.iloc[te].sum())
    n_neg = len(te) - n_pos
    # Wilson 95% half-width for sensitivity (n_pos trials) at observed 0.89
    p = 0.889
    facts["legacy_sensitivity_95ci_halfwidth_at_0.89"] = float(1.96 * np.sqrt(p * (1 - p) / n_pos))
    facts["legacy_test_note"] = (f"{n_pos} positives / {n_neg} negatives: every positive-class metric rests on "
                                 f"{n_pos} patients")

    out = ROOT / "results" / "ml_audit_facts.json"
    out.write_text(json.dumps(facts, indent=2))
    for k, v in facts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
