"""
Trains and persists every model that the Streamlit application and the CLI
inference helpers (src/inference.py) load at runtime.

Deliberately reuses classification.main() and dose_response.main() as-is
(no separate "deployment training" code path) so the model saved to disk is
EXACTLY the model whose metrics are reported in results/*.csv and RESULTS.md
-- there is no risk of a silently different, unevaluated model being served
by the application.

Usage:
    python scripts/train_and_persist.py

Every run creates NEW versioned artifacts (timestamped) and appends to
models/registry.json; nothing already on disk is overwritten or deleted.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    import classification
    import dose_response
    import drug_resistance
    import model_registry as reg
    from config import RANDOM_SEED

    t0 = time.time()
    version = reg.new_version()
    print(f"Training + persisting version {version}")

    print("\n" + "=" * 70)
    print("Classification models (MSI status + risk group)")
    print("=" * 70)
    clf_out = classification.main()
    registered = []

    for exp_key, (results_df, models, split, feature_cols, labels) in clf_out.items():
        for model_name, fitted in models.items():
            row = results_df[results_df["model"] == model_name].iloc[0]
            metrics = {k: row[k] for k in
                       ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "n_train", "n_test"]}
            entry = reg.register_model(
                name=model_name,
                task="classification",
                experiment=row["experiment"],
                model_object=fitted,
                feature_columns=list(feature_cols),
                metrics=metrics,
                hyperparameters=row["best_params"],
                dataset="tcga_coadread_features",
                random_seed=RANDOM_SEED,
                version=version,
                extra={"pos_label": labels["pos_label"], "neg_label": labels["neg_label"],
                       "pos_label_encoding": 1, "neg_label_encoding": 0},
            )
            registered.append(entry)
            print(f"  registered {entry['task']}/{entry['experiment']}/{entry['name']} -> {entry['path']}")

    print("\n" + "=" * 70)
    print("Dose-response regression models (pooled LN_IC50 + AUC)")
    print("=" * 70)
    dr_out = dose_response.main()

    for exp_key in ["pooled_LN_IC50", "pooled_AUC"]:
        results_df, models, gene_cols = dr_out[exp_key]
        target = results_df["target"].iloc[0]
        for model_name, fitted in models.items():
            row = results_df[results_df["model"] == model_name].iloc[0]
            metrics = {k: row[k] for k in ["MAE", "RMSE", "R2", "pearson_r", "spearman_r", "n_train", "n_test"]}
            feature_cols = gene_cols + ["MIN_CONC", "MAX_CONC", "PATHWAY_NAME"]
            entry = reg.register_model(
                name=model_name,
                task="dose_response",
                experiment=exp_key,
                model_object=fitted,
                feature_columns=feature_cols,
                metrics=metrics,
                hyperparameters=row["best_params"],
                dataset="gdsc_colorectal_dose_response_processed",
                random_seed=RANDOM_SEED,
                version=version,
                extra={"target_column": target},
            )
            registered.append(entry)
            print(f"  registered {entry['task']}/{entry['experiment']}/{entry['name']} -> {entry['path']}")

    print("\n" + "=" * 70)
    print("Drug-resistance classification model (pooled, median-split LN_IC50)")
    print("=" * 70)
    dr_res_out = drug_resistance.main()
    r_res, models_res, gene_cols_res, labels_res = dr_res_out["pooled_resistance"]
    for model_name, fitted in models_res.items():
        row = r_res[r_res["model"] == model_name].iloc[0]
        metrics = {k: row[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                        "n_train", "n_test"]}
        feature_cols = gene_cols_res + ["MIN_CONC", "MAX_CONC", "PATHWAY_NAME"]
        entry = reg.register_model(
            name=model_name,
            task="drug_resistance",
            experiment="pooled_resistance",
            model_object=fitted,
            feature_columns=feature_cols,
            metrics=metrics,
            hyperparameters=row["best_params"],
            dataset="gdsc_colorectal_dose_response_processed",
            random_seed=RANDOM_SEED,
            version=version,
            extra={"pos_label": labels_res["pos_label"], "neg_label": labels_res["neg_label"],
                   "pos_label_encoding": 1, "neg_label_encoding": 0},
        )
        registered.append(entry)
        print(f"  registered {entry['task']}/{entry['experiment']}/{entry['name']} -> {entry['path']}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. {len(registered)} model artifacts registered under version {version}.")
    print(f"Registry: {reg.REGISTRY_PATH}")


if __name__ == "__main__":
    main()
