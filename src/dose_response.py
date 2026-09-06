"""
Dose-response / drug-sensitivity regression module.

Predicts GDSC2 fitted pharmacological summary statistics (LN(IC50) and AUC)
for colorectal-cancer cell lines from ferroptosis-related gene expression,
drug pathway, and screening-concentration-range features.

IMPORTANT SCOPE STATEMENT (see docs/LIMITATIONS.md):
This predicts an experimentally measured, in-vitro drug-SENSITIVITY summary
statistic for a cancer cell line (LN_IC50 / AUC from a dose-response curve
already fitted by the GDSC consortium), not a human patient's clinical
dosage. No claim of clinical dosing is made anywhere in this project.

Two experiments:
  1. Pooled model: all (drug, cell line) pairs for well-covered drugs, with
     drug pathway + concentration range + ferroptosis expression as features.
     Train/test split holds out entire cell lines (see src/preprocessing.py).
  2. Representative single-drug models: for four colorectal standard-of-care
     / ferroptosis-relevant drugs (5-Fluorouracil, Oxaliplatin, Irinotecan,
     Sorafenib), fit small per-drug models using ONLY ferroptosis gene
     expression (n ~= 44-46 cell lines), evaluated with leave-one-out CV
     because the per-drug sample size is too small for a held-out test split.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

import evaluation as ev
from preprocessing import split_gdsc
from config import DATA_CONFIG, RANDOM_SEED

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

REPRESENTATIVE_DRUGS = DATA_CONFIG["gdsc"]["representative_drugs"]


def _pooled_model_pipeline(gene_cols, model):
    pre = ColumnTransformer([
        ("genes", StandardScaler(), gene_cols),
        ("conc", StandardScaler(), ["MIN_CONC", "MAX_CONC"]),
        ("pathway", OneHotEncoder(handle_unknown="ignore"), ["PATHWAY_NAME"]),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def run_pooled_experiment(df, gene_cols, target_col):
    feature_cols = gene_cols + ["MIN_CONC", "MAX_CONC", "PATHWAY_NAME"]
    split = split_gdsc(df, feature_cols, target_col)
    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    groups_train = split["meta_train"]["COSMIC_ID"]

    models_grids = {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForest": (RandomForestRegressor(random_state=RANDOM_SEED),
                          {"model__n_estimators": [200, 400], "model__max_depth": [5, 10, None]}),
        "XGBoost": (XGBRegressor(random_state=RANDOM_SEED, n_jobs=1),
                    {"model__n_estimators": [100, 200], "model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]}),
    }

    rows = []
    preds_for_plot = {}
    fitted_models = {}
    gkf = GroupKFold(n_splits=5)

    for name, (model, grid) in models_grids.items():
        pipe = _pooled_model_pipeline(gene_cols, model)
        if grid:
            cv_splits = list(gkf.split(X_train, y_train, groups_train))
            gs = GridSearchCV(pipe, grid, cv=cv_splits, scoring="neg_mean_absolute_error", n_jobs=-1)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            best_params = gs.best_params_
        else:
            best = pipe.fit(X_train, y_train)
            best_params = {}

        y_pred = best.predict(X_test)
        metrics = ev.evaluate_regression(y_test, y_pred)
        metrics.update({"target": target_col, "model": name, "best_params": json.dumps(best_params),
                         "n_train": len(X_train), "n_test": len(X_test), "n_test_cell_lines": split["meta_test"]["COSMIC_ID"].nunique()})
        rows.append(metrics)
        preds_for_plot[name] = (y_test.values, y_pred)
        fitted_models[name] = best

        ev.plot_predicted_vs_actual(y_test, y_pred, f"{target_col} - {name} (held-out cell lines)",
                                     FIG_DIR / f"predicted_vs_actual_{target_col}_{name}.png")
        ev.plot_residuals(y_test, y_pred, f"{target_col} - {name} residuals",
                          FIG_DIR / f"residuals_{target_col}_{name}.png")

    results_df = pd.DataFrame(rows)
    return results_df, preds_for_plot, fitted_models


def run_representative_drug_models(df, gene_cols):
    rows = []
    for drug in REPRESENTATIVE_DRUGS:
        sub = df[df["DRUG_NAME"] == drug].reset_index(drop=True)
        if len(sub) < 15:
            continue
        X, y = sub[gene_cols], sub["LN_IC50"]
        loo = LeaveOneOut()

        for name, model in [
            ("LinearRegression", LinearRegression()),
            ("RandomForest", RandomForestRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_SEED)),
        ]:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            y_pred = cross_val_predict(pipe, X, y, cv=loo, n_jobs=-1)
            metrics = ev.evaluate_regression(y, y_pred)
            metrics.update({"drug": drug, "model": name, "n": len(sub)})
            rows.append(metrics)

        if drug == REPRESENTATIVE_DRUGS[0]:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(
                n_estimators=200, max_depth=4, random_state=RANDOM_SEED))])
            y_pred = cross_val_predict(pipe, X, y, cv=loo, n_jobs=-1)
            ev.plot_predicted_vs_actual(y, y_pred, f"LN_IC50 - {drug} (leave-one-out, ferroptosis genes only)",
                                         FIG_DIR / f"predicted_vs_actual_{drug}_LOOCV.png")

    return pd.DataFrame(rows)


def plot_illustrative_dose_response_curve(df, drug, path):
    """
    Reconstructs an illustrative 2-parameter logistic viability curve from the
    GDSC-fitted LN_IC50 for a small number of representative cell lines. This
    visualizes what the fitted IC50 parameter means; it is NOT derived from
    raw per-concentration viability measurements (the ~2GB raw GDSC2 screening
    file was not retrieved -- see docs/LIMITATIONS.md), and is labeled as such
    directly on the figure.
    """
    sub = df[df["DRUG_NAME"] == drug].drop_duplicates("CELL_LINE_NAME")
    sub = sub.reindex(sub["LN_IC50"].sort_values().index)
    picks = pd.concat([sub.head(2), sub.tail(2)]) if len(sub) >= 4 else sub

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in picks.iterrows():
        ic50 = np.exp(row["LN_IC50"])
        conc = np.logspace(np.log10(row["MIN_CONC"]), np.log10(row["MAX_CONC"]), 200)
        viability = 1 / (1 + (conc / ic50))
        ax.plot(conc, viability, label=f"{row['CELL_LINE_NAME']} (IC50={ic50:.2f}uM)")
    ax.set_xscale("log")
    ax.set_xlabel("Concentration (uM, log scale)")
    ax.set_ylabel("Illustrative relative viability")
    ax.set_title(f"Illustrative dose-response curves - {drug}\n(reconstructed from fitted IC50; not raw viability data)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv")
    with open(PROCESSED_DIR / "gdsc_preprocessing_metadata.json") as f:
        meta = json.load(f)
    gene_cols = meta["ferroptosis_genes_used"]

    print("=== Pooled dose-response regression: LN_IC50 ===")
    r_ic50, preds_ic50, models_ic50 = run_pooled_experiment(df, gene_cols, "LN_IC50")
    print(r_ic50[["model", "MAE", "RMSE", "R2", "pearson_r"]])

    print("\n=== Pooled dose-response regression: AUC ===")
    r_auc, preds_auc, models_auc = run_pooled_experiment(df, gene_cols, "AUC")
    print(r_auc[["model", "MAE", "RMSE", "R2", "pearson_r"]])

    print("\n=== Representative single-drug models (ferroptosis genes only, LOOCV) ===")
    r_drug = run_representative_drug_models(df, gene_cols)
    print(r_drug[["drug", "model", "n", "MAE", "R2", "pearson_r"]])

    results_df = pd.concat([
        r_ic50.assign(experiment="pooled_LN_IC50"),
        r_auc.assign(experiment="pooled_AUC"),
    ], ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "regression_results.csv", index=False)
    r_drug.to_csv(RESULTS_DIR / "representative_drug_results.csv", index=False)

    ev.plot_model_comparison_bar(r_ic50, ["MAE", "RMSE"], "model",
                                  "Model comparison - LN_IC50 (MAE/RMSE, lower is better)",
                                  FIG_DIR / "model_comparison_dose_response.png")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.round(3).to_csv(TABLES_DIR / "regression_model_comparison.csv", index=False)
    r_drug.round(3).to_csv(TABLES_DIR / "representative_drug_comparison.csv", index=False)

    for drug in REPRESENTATIVE_DRUGS:
        try:
            plot_illustrative_dose_response_curve(df, drug, FIG_DIR / f"dose_response_curve_{drug}.png")
        except Exception as e:
            print(f"  (skipped illustrative curve for {drug}: {e})")

    print(f"\nSaved regression results/figures/tables.")

    return {
        "pooled_LN_IC50": (r_ic50, models_ic50, gene_cols),
        "pooled_AUC": (r_auc, models_auc, gene_cols),
        "gene_cols": gene_cols,
    }


if __name__ == "__main__":
    main()
