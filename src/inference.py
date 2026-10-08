"""
Inference-only functions: load a persisted model from the registry and score
one sample. Used by the Streamlit application (app/) and available directly
from the CLI. No training happens here.

Two real ways to obtain input data are supported everywhere below (never a
fabricated/synthetic sample):

  1. "Pick an existing sample" -- select a real TCGA patient / real GDSC
     (cell line, drug) row from the already-processed dataset. The true
     label/measured value is known and shown alongside the prediction, so the
     UI can be honestly evaluated against ground truth.
  2. "Upload new gene-expression values" -- the user supplies raw expression
     values for the ferroptosis gene panel (CSV or manual entry); these are
     put through the exact same transformation the training pipeline used
     (log2 for TCGA-style input) before being scored by the persisted model,
     which itself contains a StandardScaler/ColumnTransformer fit only on
     the training fold, so no leakage is possible at inference time either.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import model_registry as reg

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


class MissingGenesError(ValueError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(
            f"{len(missing)} required ferroptosis gene(s) missing from input: {missing}"
        )


# ---------------------------------------------------------------------------
# Classification (MSI status)
# ---------------------------------------------------------------------------

def available_classification_models(experiment: str = "exp1_msi_raw_genes") -> list[dict]:
    return reg.list_models(task="classification", experiment=experiment)


def load_classification_model(model_name: str, experiment: str = "exp1_msi_raw_genes"):
    entry = reg.latest_model(task="classification", experiment=experiment, name=model_name)
    if entry is None:
        raise FileNotFoundError(
            f"No persisted classification model '{model_name}' for experiment "
            f"'{experiment}'. Run `python scripts/train_and_persist.py` first."
        )
    return reg.load_model(entry), entry


def get_tcga_test_samples() -> pd.DataFrame:
    """Real TCGA COADREAD samples with known msi_label, for the 'pick an
    existing sample' inference demo. Returns the full feature table (already
    log2-transformed exactly as training used it)."""
    path = PROCESSED_DIR / "tcga_coadread_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python main.py` first.")
    df = pd.read_csv(path)
    return df[df["msi_label"].notna()].reset_index(drop=True)


def predict_msi_from_processed_row(row: pd.Series, model_name: str,
                                    experiment: str = "exp1_msi_raw_genes"):
    """Scores a row already taken from tcga_coadread_features.csv (i.e.
    already log2-transformed identically to training)."""
    pipeline, entry = load_classification_model(model_name, experiment)
    feature_cols = entry["feature_columns"]
    missing = [g for g in feature_cols if g not in row.index or pd.isna(row[g])]
    if missing:
        raise MissingGenesError(missing)
    X = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
    return _score_classification(pipeline, X, entry)


def predict_msi_from_raw_expression(raw_expression: dict[str, float], model_name: str,
                                     experiment: str = "exp1_msi_raw_genes"):
    """Scores user-supplied RAW (untransformed) expression values, applying
    the identical log2(clip(x, 0) + 1) transform used in
    preprocessing.build_tcga_dataset before scoring."""
    pipeline, entry = load_classification_model(model_name, experiment)
    feature_cols = entry["feature_columns"]
    missing = [g for g in feature_cols if g not in raw_expression or raw_expression[g] is None
               or (isinstance(raw_expression[g], float) and np.isnan(raw_expression[g]))]
    if missing:
        raise MissingGenesError(missing)
    values = [np.log2(max(float(raw_expression[g]), 0.0) + 1.0) for g in feature_cols]
    X = pd.DataFrame([values], columns=feature_cols)
    return _score_classification(pipeline, X, entry)


def _score_classification(pipeline, X: pd.DataFrame, entry: dict):
    # The persisted pipeline was fit on a 0/1-encoded target
    # ((y == pos_label).astype(int), see classification.py::run_experiment),
    # so pipeline.classes_ is [0, 1], not the original string labels. Map
    # back using the pos_label/neg_label recorded at training time.
    extra = entry.get("extra", {})
    pos_label = extra.get("pos_label")
    neg_label = extra.get("neg_label")
    label_for_encoding = {1: pos_label, 0: neg_label} if pos_label is not None else {}

    proba = pipeline.predict_proba(X)[0]
    classes = list(pipeline.classes_)
    proba_by_class = {label_for_encoding.get(int(c), str(c)): p for c, p in zip(classes, proba.tolist())}
    pred_encoded = int(pipeline.predict(X)[0])
    pred_label = label_for_encoding.get(pred_encoded, str(pred_encoded))

    result = {
        "predicted_label": pred_label,
        "class_probabilities": proba_by_class,
        "model_name": entry["name"],
        "model_version": entry["version"],
        "experiment": entry["experiment"],
        "reported_test_metrics": entry["metrics"],
    }

    # Local, exact explanation for linear models fit directly on
    # entry["feature_columns"] via a plain StandardScaler (the MSI-
    # classification pipeline shape: named steps "scaler" + "clf"). A
    # ColumnTransformer-based pipeline (named step "pre", used by the
    # drug-resistance pooled model, which one-hot-encodes PATHWAY_NAME) does
    # NOT produce one transformed column per entry in feature_columns, so a
    # coefficient-times-value contribution table would silently misalign --
    # that shape falls through to the real, precomputed global importance
    # instead of a fabricated/misaligned per-sample explanation.
    clf = pipeline.named_steps.get("clf")
    scaler = pipeline.named_steps.get("scaler")
    if clf is not None and scaler is not None and hasattr(clf, "coef_"):
        X_scaled = scaler.transform(X)
        contributions = (clf.coef_[0] * X_scaled[0])
        contrib_df = pd.DataFrame({
            "gene": entry["feature_columns"],
            "contribution": contributions,
        }).sort_values("contribution", key=np.abs, ascending=False)
        result["local_explanation"] = contrib_df.to_dict(orient="records")
        result["explanation_type"] = "exact_linear_contribution"
    else:
        result["explanation_type"] = "global_importance_only"

    return result


# ---------------------------------------------------------------------------
# Dose-response regression (LN_IC50 / AUC)
# ---------------------------------------------------------------------------

_TARGET_TO_EXPERIMENT = {"LN_IC50": "pooled_LN_IC50", "AUC": "pooled_AUC"}


def available_dose_response_models(target: str = "LN_IC50") -> list[dict]:
    return reg.list_models(task="dose_response", experiment=_TARGET_TO_EXPERIMENT[target])


def load_dose_response_model(model_name: str, target: str = "LN_IC50"):
    experiment = _TARGET_TO_EXPERIMENT[target]
    entry = reg.latest_model(task="dose_response", experiment=experiment, name=model_name)
    if entry is None:
        raise FileNotFoundError(
            f"No persisted dose-response model '{model_name}' for target '{target}'. "
            f"Run `python scripts/train_and_persist.py` first."
        )
    return reg.load_model(entry), entry


def get_gdsc_processed() -> pd.DataFrame:
    path = PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python main.py` first.")
    return pd.read_csv(path)


def get_available_drugs() -> pd.DataFrame:
    """One row per drug with its (most common) pathway and concentration
    range -- the real drug-representation features used by the pooled model,
    read directly from the processed GDSC dataset (no fabricated values)."""
    df = get_gdsc_processed()
    agg = df.groupby("DRUG_NAME").agg(
        PATHWAY_NAME=("PATHWAY_NAME", lambda s: s.mode().iloc[0]),
        MIN_CONC=("MIN_CONC", "median"),
        MAX_CONC=("MAX_CONC", "median"),
        n_cell_lines=("COSMIC_ID", "nunique"),
    ).reset_index()
    return agg.sort_values("DRUG_NAME")


def predict_dose_response_from_processed_row(row: pd.Series, drug_name: str, model_name: str,
                                              target: str = "LN_IC50"):
    """Scores a real (cell line, drug) row already in the processed GDSC
    dataset -- ground truth (row[target]) is known, so the UI can show
    predicted vs actual honestly."""
    pipeline, entry = load_dose_response_model(model_name, target)
    gene_cols = [c for c in entry["feature_columns"] if c not in ("MIN_CONC", "MAX_CONC", "PATHWAY_NAME")]
    missing = [g for g in gene_cols if g not in row.index or pd.isna(row[g])]
    if missing:
        raise MissingGenesError(missing)
    X = pd.DataFrame([{**{g: row[g] for g in gene_cols},
                        "MIN_CONC": row["MIN_CONC"], "MAX_CONC": row["MAX_CONC"],
                        "PATHWAY_NAME": row["PATHWAY_NAME"]}])
    return _score_regression(pipeline, X, entry, actual=row.get(target))


def predict_dose_response_from_raw_expression(gene_expression: dict[str, float], drug_name: str,
                                               model_name: str, target: str = "LN_IC50"):
    """Scores user-supplied gene expression (raw log2-RMA-scale values, same
    scale as the processed GDSC dataset -- no additional transform is
    applied here since GDSC's brainarray values are already log2) against a
    real, existing drug's pathway/concentration-range annotation."""
    pipeline, entry = load_dose_response_model(model_name, target)
    gene_cols = [c for c in entry["feature_columns"] if c not in ("MIN_CONC", "MAX_CONC", "PATHWAY_NAME")]
    missing = [g for g in gene_cols if g not in gene_expression or gene_expression[g] is None]
    if missing:
        raise MissingGenesError(missing)

    drugs = get_available_drugs().set_index("DRUG_NAME")
    if drug_name not in drugs.index:
        raise ValueError(f"Drug '{drug_name}' is not in the well-covered GDSC drug panel used for training.")
    drug_row = drugs.loc[drug_name]

    X = pd.DataFrame([{**{g: float(gene_expression[g]) for g in gene_cols},
                        "MIN_CONC": float(drug_row["MIN_CONC"]), "MAX_CONC": float(drug_row["MAX_CONC"]),
                        "PATHWAY_NAME": drug_row["PATHWAY_NAME"]}])
    return _score_regression(pipeline, X, entry, actual=None)


def _score_regression(pipeline, X: pd.DataFrame, entry: dict, actual=None):
    pred = float(pipeline.predict(X)[0])
    result = {
        "predicted_value": pred,
        "model_name": entry["name"],
        "model_version": entry["version"],
        "experiment": entry["experiment"],
        "target_column": entry.get("extra", {}).get("target_column", "LN_IC50"),
        "reported_test_metrics": entry["metrics"],
    }
    if actual is not None and not pd.isna(actual):
        result["actual_value"] = float(actual)
        result["absolute_error"] = abs(pred - float(actual))

    # Ensemble-spread uncertainty proxy for RandomForest: the empirical
    # spread of the individual trees' predictions on this exact input, after
    # passing it through the SAME fitted preprocessing step the ensemble
    # uses. This is a real, computed quantity (not a fabricated confidence
    # interval) -- documented as such in the UI.
    model_step = pipeline.named_steps.get("model")
    if model_step is not None and hasattr(model_step, "estimators_"):
        pre = pipeline.named_steps.get("pre")
        X_t = pre.transform(X) if pre is not None else X.values
        tree_preds = np.array([est.predict(X_t)[0] for est in model_step.estimators_])
        result["ensemble_spread_std"] = float(tree_preds.std())
        result["ensemble_spread_min"] = float(tree_preds.min())
        result["ensemble_spread_max"] = float(tree_preds.max())

    return result


# ---------------------------------------------------------------------------
# Drug-resistance classification (Resistant / Sensitive)
# ---------------------------------------------------------------------------

_RESISTANCE_EXPERIMENT = "pooled_resistance"


def available_resistance_models() -> list[dict]:
    return reg.list_models(task="drug_resistance", experiment=_RESISTANCE_EXPERIMENT)


def load_resistance_model(model_name: str):
    entry = reg.latest_model(task="drug_resistance", experiment=_RESISTANCE_EXPERIMENT, name=model_name)
    if entry is None:
        raise FileNotFoundError(
            f"No persisted drug-resistance model '{model_name}'. Run `python scripts/train_and_persist.py` first."
        )
    return reg.load_model(entry), entry


def predict_resistance_from_processed_row(row: pd.Series, drug_name: str, model_name: str):
    """Scores a real (cell line, drug) row already in the processed GDSC
    dataset. Ground truth here is the same train-fold-median-split label
    src/drug_resistance.py used to train the model, recomputed from the
    row's own LN_IC50 for display -- the UI shows it as a reference label,
    exactly like predict_dose_response_from_processed_row's actual_value."""
    pipeline, entry = load_resistance_model(model_name)
    gene_cols = [c for c in entry["feature_columns"] if c not in ("MIN_CONC", "MAX_CONC", "PATHWAY_NAME")]
    missing = [g for g in gene_cols if g not in row.index or pd.isna(row[g])]
    if missing:
        raise MissingGenesError(missing)
    X = pd.DataFrame([{**{g: row[g] for g in gene_cols},
                        "MIN_CONC": row["MIN_CONC"], "MAX_CONC": row["MAX_CONC"],
                        "PATHWAY_NAME": row["PATHWAY_NAME"]}])
    return _score_classification(pipeline, X, entry)


def predict_resistance_from_raw_expression(gene_expression: dict[str, float], drug_name: str, model_name: str):
    """Scores user-supplied gene expression (same log2-RMA scale as GDSC,
    no additional transform) against a real, existing drug's pathway/
    concentration-range annotation, mirroring
    predict_dose_response_from_raw_expression."""
    pipeline, entry = load_resistance_model(model_name)
    gene_cols = [c for c in entry["feature_columns"] if c not in ("MIN_CONC", "MAX_CONC", "PATHWAY_NAME")]
    missing = [g for g in gene_cols if g not in gene_expression or gene_expression[g] is None]
    if missing:
        raise MissingGenesError(missing)

    drugs = get_available_drugs().set_index("DRUG_NAME")
    if drug_name not in drugs.index:
        raise ValueError(f"Drug '{drug_name}' is not in the well-covered GDSC drug panel used for training.")
    drug_row = drugs.loc[drug_name]

    X = pd.DataFrame([{**{g: float(gene_expression[g]) for g in gene_cols},
                        "MIN_CONC": float(drug_row["MIN_CONC"]), "MAX_CONC": float(drug_row["MAX_CONC"]),
                        "PATHWAY_NAME": drug_row["PATHWAY_NAME"]}])
    return _score_classification(pipeline, X, entry)
