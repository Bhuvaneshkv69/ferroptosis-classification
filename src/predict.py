"""
Inference layer for the production ("champion") models. The champion is looked up in the registry
(models/champions.json) -- never hard-coded -- and every function here only SCORES: nothing is fitted.

Two scales exist in this project and are kept explicit:
  * classification model : TCGA RNA-seq, log2(RSEM + 1)
  * drug-response model  : GDSC microarray, log2 RMA
`harmonize_to_gdsc` is the (exploratory, unvalidated) bridge used only when one RNA-seq sample is sent to
both models; it is flagged as such wherever it is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import model_registry as reg

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
BENCH = ROOT / "results" / "benchmark"


class NoProductionModel(RuntimeError):
    pass


def load_production(task: str):
    """(registry entry, loaded bundle) for the current champion of `task`."""
    entry = reg.current_best(task)
    if entry is None:
        raise NoProductionModel(
            f"No production model for '{task}'. Run `python scripts/run_benchmark.py` "
            f"(classification) or `python scripts/run_benchmark.py --task regression`.")
    return entry, reg.load_model(entry)


def to_log2(df: pd.DataFrame, genes: list[str], already_log2: bool = False) -> pd.DataFrame:
    X = df[genes].apply(pd.to_numeric, errors="coerce").astype(float)
    return X if already_log2 else np.log2(X.clip(lower=0) + 1.0)


# ------------------------------------------------------------------ classification
def is_ensemble(bundle: dict) -> bool:
    return bundle.get("kind") == "ensemble" and hasattr(bundle["model"], "suite")


def classify(bundle: dict, X_log2: pd.DataFrame) -> pd.DataFrame:
    genes = bundle["features"]
    missing = [g for g in genes if g not in X_log2.columns]
    if missing:
        raise ValueError(f"This model needs {len(genes)} genes; {len(missing)} are missing from the input "
                         f"({', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}). "
                         "No other model is substituted: provide the full gene set or choose the matching model.")
    X = X_log2[genes]
    proba = bundle["model"].predict_proba(X)[:, 1]
    pred = bundle["model"].predict(X)
    out = pd.DataFrame({"model_probability": proba, "predicted_class": np.where(pred == 1, bundle["pos_label"],
                                                                                  bundle["neg_label"]),
                        "threshold": bundle["threshold"]}, index=X.index)
    if is_ensemble(bundle):
        votes = bundle["model"].suite.member_proba(X) >= bundle["model"].suite.member_thr_
        out["members_supporting_high"] = votes.sum(axis=1)
        out["n_members"] = votes.shape[1]
    return out


def member_table(bundle: dict, X_log2_row: pd.DataFrame) -> pd.DataFrame:
    """Per-member probability and class for ONE sample (only for ensemble bundles; computed, never assumed)."""
    suite = bundle["model"].suite
    P = suite.member_proba(X_log2_row[bundle["features"]].iloc[[0]])[0]
    return pd.DataFrame({"member": suite.names_, "probability": P, "threshold": suite.member_thr_,
                         "supports_high": P >= suite.member_thr_, "weight": suite.weights_})


def explain_local(bundle: dict, x_row: pd.Series, top_k: int = 10) -> pd.DataFrame:
    """Per-gene contribution of this sample's values to the model probability (SHAP, permutation algorithm,
    background = a stored sample of the development set). Model-agnostic, so it works for every winner."""
    genes, model, bg = bundle["features"], bundle["model"], bundle["background"][bundle["features"]]

    def f(d):
        return model.predict_proba(pd.DataFrame(d, columns=genes))[:, 1]

    try:
        import shap
        masker = shap.maskers.Independent(bg.to_numpy(), max_samples=25)
        sv = shap.Explainer(f, masker, algorithm="permutation")(
            x_row[genes].to_numpy(float).reshape(1, -1), max_evals=2 * len(genes) + 1, silent=True)
        contrib = np.asarray(sv.values)[0]
        method = "SHAP (permutation)"
    except Exception:  # fallback: single-gene occlusion at the development median
        base = f(x_row[genes].to_numpy(float).reshape(1, -1))[0]
        med = bg.median()
        contrib = np.array([base - f(np.where(np.arange(len(genes)) == i, med.to_numpy(),
                                              x_row[genes].to_numpy(float)).reshape(1, -1))[0]
                            for i in range(len(genes))])
        method = "single-gene occlusion"
    z = (x_row[genes].astype(float) - bg.mean()) / (bg.std() + 1e-9)
    out = pd.DataFrame({"gene": genes, "value_log2": x_row[genes].astype(float).to_numpy(), "z_vs_training": z.to_numpy(),
                        "contribution": contrib})
    out["direction"] = np.where(out["contribution"] >= 0, "toward MSI-High", "toward MSS / MSI-Low")
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top_k)
    out.attrs["method"] = method
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ drug response
def harmonize_to_gdsc(X_log2: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """Per-gene quantile mapping from the TCGA log2(RSEM+1) distribution onto the GDSC log2-RMA distribution
    (reference distributions are cohort-level and label-free). EXPLORATORY: assumes tumours and cell lines occupy
    comparable expression ranks; it has not been validated against paired measurements."""
    tcga = pd.read_csv(PROCESSED / "tcga_coadread_processed.csv")
    gdsc = pd.read_csv(PROCESSED / "gdsc_colorectal_dose_response_processed.csv").drop_duplicates("COSMIC_ID")
    out = pd.DataFrame(index=X_log2.index, columns=genes, dtype=float)
    for g in genes:
        ref_t = np.sort(tcga[g].dropna().to_numpy())
        q = (np.searchsorted(ref_t, X_log2[g].to_numpy(), side="left")
             + np.searchsorted(ref_t, X_log2[g].to_numpy(), side="right")) / (2.0 * len(ref_t))
        out[g] = np.quantile(gdsc[g].dropna().to_numpy(), np.clip(q, 0.0, 1.0))
    return out


def drug_table(bundle: dict) -> pd.DataFrame:
    return bundle["drugs"].sort_values("DRUG_NAME").reset_index(drop=True)


def predict_ln_ic50(bundle: dict, X_gdsc: pd.DataFrame, drug: str) -> pd.DataFrame:
    genes = bundle["genes"]
    d = bundle["drugs"].set_index("DRUG_NAME")
    if drug not in d.index:
        raise ValueError(f"'{drug}' is not in the compound panel the model was trained on.")
    row = d.loc[drug]
    X = X_gdsc[genes].copy()
    X["COSMIC_ID"] = -1
    X["DRUG_NAME"] = drug
    X["PATHWAY_NAME"] = row["PATHWAY_NAME"]
    X["MIN_CONC"], X["MAX_CONC"] = float(row["MIN_CONC"]), float(row["MAX_CONC"])
    ln = bundle["model"].predict(X)
    return pd.DataFrame({"ln_ic50": ln, "ic50_uM": np.exp(ln)}, index=X_gdsc.index)


def load_json(path: Path) -> dict | None:
    return json.load(open(path)) if path.exists() else None


def predict_all_drugs(bundle: dict, X_gdsc_row: pd.DataFrame) -> pd.DataFrame:
    """LN(IC50) for every compound in one vectorised call (one sample x every compound)."""
    genes, d = bundle["genes"], bundle["drugs"].reset_index(drop=True)
    X = pd.concat([X_gdsc_row[genes].iloc[[0]]] * len(d), ignore_index=True)
    X["COSMIC_ID"] = -1
    X["DRUG_NAME"] = d["DRUG_NAME"].to_numpy()
    X["PATHWAY_NAME"] = d["PATHWAY_NAME"].to_numpy()
    X["MIN_CONC"], X["MAX_CONC"] = d["MIN_CONC"].astype(float).to_numpy(), d["MAX_CONC"].astype(float).to_numpy()
    ln = bundle["model"].predict(X)
    return d.assign(ln_ic50=ln, ic50_uM=np.exp(ln))
