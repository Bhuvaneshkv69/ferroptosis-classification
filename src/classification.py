"""
Classification module: predicts colorectal-cancer molecular subgroups from
ferroptosis-related gene expression.

Two independent classification tasks are run (see docs/METHODOLOGY.md for why
they are kept separate):

  1. MSI-status classification (PRIMARY, non-circular).
     Target: MSI-H vs MSS/MSI-L, derived from the MSI MANTIS score
     (threshold 0.4, Kautto et al., Oncotarget 2017) -- a clinically defined
     label that does NOT depend on the ferroptosis genes used as features.

  2. Ferroptosis survival-risk-group classification (SECONDARY, exploratory).
     Target: High/Low risk from the Cox-regression risk score built in
     src/feature_engineering.py. This label is a linear function of the same
     gene features used to predict it, so results are reported with an
     explicit circularity caveat and are not used as the project's headline
     claim.

For each task, two feature sets are compared (raw ferroptosis genes vs the
compact engineered signature) across four models (Logistic Regression, Random
Forest, SVM, XGBoost) with 5-fold stratified cross-validated hyperparameter
search on the training set only, then a single held-out test-set evaluation.
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import evaluation as ev
from config import RANDOM_SEED

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"


def _model_grid():
    return {
        "LogisticRegression": (
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_SEED))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5, None]},
        ),
        "SVM": (
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(
                probability=True, class_weight="balanced", random_state=RANDOM_SEED))]),
            {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["linear", "rbf"]},
        ),
        "XGBoost": (
            Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier(
                eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=1))]),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.05, 0.1]},
        ),
    }


def run_experiment(df, feature_cols, target_col, pos_label, experiment_name, test_size=0.2):
    """Runs all four models for one (feature set, target) combination."""
    sub = df[df[target_col].notna()].reset_index(drop=True)
    X, y = sub[feature_cols], sub[target_col].astype(str)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grids = _model_grid()

    rows = []
    roc_inputs, pr_inputs, cal_inputs = {}, {}, {}
    fitted_models = {}

    for name, (pipe, params) in grids.items():
        gs = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_trainval, (y_trainval == pos_label).astype(int))
        best = gs.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, list(best.classes_).index(1)]
        y_pred_label = np.where(y_pred == 1, pos_label, [c for c in sub[target_col].unique() if c != pos_label][0])

        metrics = ev.evaluate_classification(y_test, y_pred_label, y_proba, pos_label)
        metrics.update({"experiment": experiment_name, "model": name, "best_params": json.dumps(gs.best_params_),
                         "n_train": len(X_trainval), "n_test": len(X_test)})
        rows.append(metrics)

        roc_inputs[name] = (y_test, y_proba)
        pr_inputs[name] = (y_test, y_proba)
        cal_inputs[name] = (y_test, y_proba)
        fitted_models[name] = best

        cm_path = FIG_DIR / f"confusion_matrix_{experiment_name}_{name}.png"
        ev.plot_confusion_matrix(y_test, y_pred_label, sorted(y.unique()),
                                  f"{experiment_name} - {name}", cm_path)

    ev.plot_roc_curves(roc_inputs, pos_label, f"ROC - {experiment_name}", FIG_DIR / f"roc_curve_{experiment_name}.png")
    ev.plot_pr_curves(pr_inputs, pos_label, f"PR - {experiment_name}", FIG_DIR / f"pr_curve_{experiment_name}.png")
    ev.plot_calibration(cal_inputs, pos_label, f"Calibration - {experiment_name}",
                        FIG_DIR / f"calibration_{experiment_name}.png")

    neg_label = [c for c in sub[target_col].unique() if c != pos_label][0]
    results_df = pd.DataFrame(rows)
    return results_df, fitted_models, (X_trainval, y_trainval, X_test, y_test), {"pos_label": pos_label, "neg_label": neg_label}


def main():
    tcga = pd.read_csv(PROCESSED_DIR / "tcga_coadread_features.csv")
    with open(PROCESSED_DIR / "tcga_preprocessing_metadata.json") as f:
        meta = json.load(f)
    raw_genes = meta["ferroptosis_genes_used"]
    engineered_features = ["ferroptosis_score_mean", "ferroptosis_score_pc1"]

    all_results = []

    print("=== Experiment 1: MSI status, raw ferroptosis genes (baseline) ===")
    r1, models1, split1, labels1 = run_experiment(tcga, raw_genes, "msi_label", "MSI-H", "exp1_msi_raw_genes")
    all_results.append(r1)
    print(r1[["model", "accuracy", "roc_auc", "f1"]])

    print("\n=== Experiment 2: MSI status, engineered ferroptosis signature score ===")
    r2, models2, split2, labels2 = run_experiment(tcga, engineered_features, "msi_label", "MSI-H", "exp2_msi_engineered")
    all_results.append(r2)
    print(r2[["model", "accuracy", "roc_auc", "f1"]])

    print("\n=== Experiment 3 (exploratory/circular): ferroptosis risk group, raw genes ===")
    r3, models3, split3, labels3 = run_experiment(tcga, raw_genes, "ferroptosis_risk_group", "High risk", "exp3_riskgroup_raw_genes")
    all_results.append(r3)
    print(r3[["model", "accuracy", "roc_auc", "f1"]])

    results_df = pd.concat(all_results, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "classification_results.csv", index=False)

    # Experiment 4: model comparison chart for the primary (Experiment 1) task
    ev.plot_model_comparison_bar(
        r1, ["accuracy", "precision", "recall", "f1", "roc_auc"], "model",
        "Model comparison - MSI classification (raw genes)",
        FIG_DIR / "model_comparison_msi_classification.png",
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary = results_df.pivot_table(index=["experiment", "model"], values=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
    summary.round(3).to_csv(TABLES_DIR / "classification_model_comparison.csv")

    print(f"\nSaved: {RESULTS_DIR/'classification_results.csv'}, "
          f"{TABLES_DIR/'classification_model_comparison.csv'}, figures in {FIG_DIR}")

    return {
        "exp1": (r1, models1, split1, raw_genes, labels1),
        "exp2": (r2, models2, split2, engineered_features, labels2),
        "exp3": (r3, models3, split3, raw_genes, labels3),
    }


if __name__ == "__main__":
    main()
