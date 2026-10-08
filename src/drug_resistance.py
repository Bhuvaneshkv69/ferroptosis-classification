"""
Drug-resistance classification module.

Complements the dose-response REGRESSION task (src/dose_response.py, which
predicts a continuous LN_IC50/AUC value) with a binary CLASSIFICATION task:
is a given (cell line, drug) pair "Resistant" or "Sensitive"?

Motivated by the ferroptosis-drug-resistance literature reviewed in
docs/REFERENCES.md (papers 6-7: ferroptosis and chemoresistance in
colorectal cancer in general, and ferroptosis-related oxaliplatin resistance
specifically) -- those papers motivate WHY this analysis is scientifically
worth building (ferroptosis-pathway activity is repeatedly reported as
associated with chemoresistance in the literature), but this project did not
have verified access to their exact reported thresholds or resistance-gene
panels, so no paper-specific numbers are reproduced here. Instead,
"Resistant"/"Sensitive" labels are defined transparently from THIS project's
own processed GDSC distribution: a (cell line, drug) pair is labeled
Resistant if its LN_IC50 is at or above that drug's TRAINING-FOLD median
LN_IC50 across the tested colorectal cell lines (Sensitive otherwise) -- a
standard, reproducible median-split definition used widely in GDSC-based ML
studies, computed strictly on the training fold so no test-fold cell line's
IC50 value influences its own label threshold (the same train-only-fitting
discipline used throughout this project -- see docs/DATA_LEAKAGE_AUDIT.md).

Two experiments, mirroring dose_response.py's structure:
  1. Pooled resistance model: all well-covered drugs, ferroptosis genes +
     drug pathway + concentration range as features, entire-cell-line
     held-out test split (GroupShuffleSplit/GroupKFold by COSMIC_ID). This
     is the model persisted to the registry and served by the application.
  2. Oxaliplatin-specific resistance model (paper 7's specific focus):
     ferroptosis genes only, leave-one-out CV (same small-n treatment as
     dose_response.py's representative single-drug models) -- reported as a
     diagnostic finding, not deployed for live inference (n is too small
     for a held-out split, exactly as already documented for the
     representative dose-response models).
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import evaluation as ev
from config import DATA_CONFIG, RANDOM_SEED

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

OXALIPLATIN = "Oxaliplatin"
POS_LABEL = "Resistant"
NEG_LABEL = "Sensitive"


def label_resistance(train_df: pd.DataFrame, apply_df: pd.DataFrame,
                      drug_col: str = "DRUG_NAME", value_col: str = "LN_IC50") -> tuple[pd.Series, dict]:
    """Median-split resistance label, fit on train_df only, applied to
    apply_df (which may be train_df itself, or a disjoint test set). Never
    computes a threshold from apply_df -- see module docstring."""
    thresholds = train_df.groupby(drug_col)[value_col].median().to_dict()
    global_threshold = float(train_df[value_col].median())
    label = apply_df.apply(
        lambda row: POS_LABEL if row[value_col] >= thresholds.get(row[drug_col], global_threshold) else NEG_LABEL,
        axis=1,
    )
    return label, {**thresholds, "_global_fallback": global_threshold}


def _pooled_pipeline(gene_cols: list[str], model) -> Pipeline:
    pre = ColumnTransformer([
        ("genes", StandardScaler(), gene_cols),
        ("conc", StandardScaler(), ["MIN_CONC", "MAX_CONC"]),
        ("pathway", OneHotEncoder(handle_unknown="ignore"), ["PATHWAY_NAME"]),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def run_pooled_resistance_experiment(df: pd.DataFrame, gene_cols: list[str]):
    feature_cols = gene_cols + ["MIN_CONC", "MAX_CONC", "PATHWAY_NAME"]

    gss = GroupShuffleSplit(n_splits=1, test_size=DATA_CONFIG["gdsc"]["test_size"], random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["COSMIC_ID"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_df["resistance_label"], thresholds = label_resistance(train_df, train_df)
    test_df["resistance_label"], _ = label_resistance(train_df, test_df)

    X_train, y_train = train_df[feature_cols], train_df["resistance_label"]
    X_test, y_test = test_df[feature_cols], test_df["resistance_label"]
    groups_train = train_df["COSMIC_ID"]

    grids = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5, None]},
        ),
    }

    rows = []
    fitted_models = {}
    roc_inputs, pr_inputs = {}, {}
    gkf = GroupKFold(n_splits=5)

    for name, (model, grid) in grids.items():
        pipe = _pooled_pipeline(gene_cols, model)
        cv_splits = list(gkf.split(X_train, (y_train == POS_LABEL).astype(int), groups_train))
        gs = GridSearchCV(pipe, grid, cv=cv_splits, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, (y_train == POS_LABEL).astype(int))
        best = gs.best_estimator_

        y_pred_bin = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, list(best.classes_).index(1)]
        y_pred_label = np.where(y_pred_bin == 1, POS_LABEL, NEG_LABEL)

        metrics = ev.evaluate_classification(y_test, y_pred_label, y_proba, POS_LABEL)
        metrics.update({"experiment": "pooled_resistance", "model": name, "best_params": json.dumps(gs.best_params_),
                         "n_train": len(X_train), "n_test": len(X_test),
                         "n_test_cell_lines": test_df["COSMIC_ID"].nunique()})
        rows.append(metrics)
        roc_inputs[name] = (y_test, y_proba)
        pr_inputs[name] = (y_test, y_proba)
        fitted_models[name] = best

        ev.plot_confusion_matrix(y_test, y_pred_label, [NEG_LABEL, POS_LABEL],
                                  f"pooled_resistance - {name}", FIG_DIR / f"confusion_matrix_pooled_resistance_{name}.png")

    ev.plot_roc_curves(roc_inputs, POS_LABEL, "ROC - drug resistance (pooled)",
                        FIG_DIR / "roc_curve_pooled_resistance.png")
    ev.plot_pr_curves(pr_inputs, POS_LABEL, "PR - drug resistance (pooled)",
                       FIG_DIR / "pr_curve_pooled_resistance.png")

    results_df = pd.DataFrame(rows)
    return results_df, fitted_models, {"pos_label": POS_LABEL, "neg_label": NEG_LABEL}, thresholds


def run_oxaliplatin_resistance_model(df: pd.DataFrame, gene_cols: list[str]):
    """Diagnostic-only (not deployed): oxaliplatin has the same ~44-cell-line
    sample size as dose_response.py's representative single-drug regressions,
    too small for a held-out split, so leave-one-out CV is used instead --
    identical treatment to the existing representative-drug regression
    models, applied here to the binary resistance task paper 7 focuses on."""
    sub = df[df["DRUG_NAME"] == OXALIPLATIN].reset_index(drop=True)
    if len(sub) < 15:
        return pd.DataFrame(), {}

    sub["resistance_label"], thresholds = label_resistance(sub, sub)
    X = sub[gene_cols]
    y = (sub["resistance_label"] == POS_LABEL).astype(int)
    loo = LeaveOneOut()

    rows = []
    for name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=4,
                                                 class_weight="balanced", random_state=RANDOM_SEED)),
    ]:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        y_proba = cross_val_predict(pipe, X, y, cv=loo, method="predict_proba", n_jobs=-1)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        y_pred_label = np.where(y_pred == 1, POS_LABEL, NEG_LABEL)
        metrics = ev.evaluate_classification(sub["resistance_label"], y_pred_label, y_proba, POS_LABEL)
        metrics.update({"drug": OXALIPLATIN, "model": name, "n": len(sub)})
        rows.append(metrics)

    return pd.DataFrame(rows), thresholds


def main():
    df = pd.read_csv(PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv")
    with open(PROCESSED_DIR / "gdsc_preprocessing_metadata.json") as f:
        meta = json.load(f)
    gene_cols = meta["ferroptosis_genes_used"]

    print("=== Pooled drug-resistance classification (median-split LN_IC50, train-fold-only threshold) ===")
    r_pooled, models_pooled, labels_pooled, thresholds_pooled = run_pooled_resistance_experiment(df, gene_cols)
    print(r_pooled[["model", "accuracy", "roc_auc", "f1"]])

    print("\n=== Oxaliplatin-specific resistance classification (ferroptosis genes only, leave-one-out CV) ===")
    r_oxali, thresholds_oxali = run_oxaliplatin_resistance_model(df, gene_cols)
    if not r_oxali.empty:
        print(r_oxali[["model", "accuracy", "roc_auc", "f1", "n"]])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    r_pooled.to_csv(RESULTS_DIR / "drug_resistance_results.csv", index=False)
    r_oxali.to_csv(RESULTS_DIR / "oxaliplatin_resistance_results.csv", index=False)

    ev.plot_model_comparison_bar(r_pooled, ["accuracy", "precision", "recall", "f1", "roc_auc"], "model",
                                  "Model comparison - drug resistance (pooled)",
                                  FIG_DIR / "model_comparison_drug_resistance.png")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    r_pooled.round(3).to_csv(TABLES_DIR / "drug_resistance_model_comparison.csv", index=False)

    with open(PROCESSED_DIR / "drug_resistance_thresholds.json", "w") as f:
        json.dump({"pooled_thresholds": thresholds_pooled, "oxaliplatin_threshold": thresholds_oxali}, f,
                   indent=2, default=str)

    print(f"\nSaved: {RESULTS_DIR/'drug_resistance_results.csv'}, "
          f"{RESULTS_DIR/'oxaliplatin_resistance_results.csv'}, figures in {FIG_DIR}")

    return {
        "pooled_resistance": (r_pooled, models_pooled, gene_cols, labels_pooled),
    }


if __name__ == "__main__":
    main()
