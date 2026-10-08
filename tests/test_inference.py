from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _prereqs_present():
    return (ROOT / "data" / "processed" / "tcga_coadread_features.csv").exists() and \
           (ROOT / "models" / "registry.json").exists()


pytestmark = pytest.mark.skipif(not _prereqs_present(),
                                 reason="Run `python main.py` and `python scripts/train_and_persist.py` first")


def test_predict_msi_from_processed_row_returns_a_real_label_and_probabilities():
    import inference as inf
    samples = inf.get_tcga_test_samples()
    row = samples.iloc[0]
    result = inf.predict_msi_from_processed_row(row, "LogisticRegression")
    assert result["predicted_label"] in ("MSI-H", "MSS/MSI-L")
    assert set(result["class_probabilities"]) == {"MSI-H", "MSS/MSI-L"}
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-6


def test_predict_msi_raises_missing_genes_error_when_a_required_gene_is_absent():
    import inference as inf
    samples = inf.get_tcga_test_samples()
    row = samples.iloc[0].copy()
    _, entry = inf.load_classification_model("LogisticRegression")
    dropped_gene = entry["feature_columns"][0]
    row = row.drop(labels=[dropped_gene])
    with pytest.raises(inf.MissingGenesError) as exc:
        inf.predict_msi_from_processed_row(row, "LogisticRegression")
    assert dropped_gene in exc.value.missing


def test_predict_msi_from_raw_expression_applies_log2_transform_consistently():
    import inference as inf
    _, entry = inf.load_classification_model("LogisticRegression")
    genes = entry["feature_columns"]
    raw = {g: 100.0 for g in genes}
    result = inf.predict_msi_from_raw_expression(raw, "LogisticRegression")
    assert result["predicted_label"] in ("MSI-H", "MSS/MSI-L")


def test_load_classification_model_raises_for_unknown_model_name():
    import inference as inf
    with pytest.raises(FileNotFoundError):
        inf.load_classification_model("NotARealModel")


def test_predict_dose_response_from_processed_row_matches_actual_column():
    import inference as inf
    gdsc = inf.get_gdsc_processed()
    row = gdsc.iloc[0]
    result = inf.predict_dose_response_from_processed_row(row, row["DRUG_NAME"], "RandomForest", target="LN_IC50")
    assert result["actual_value"] == pytest.approx(row["LN_IC50"])
    assert result["absolute_error"] == pytest.approx(abs(result["predicted_value"] - row["LN_IC50"]))


def test_predict_dose_response_rejects_unknown_drug():
    import inference as inf
    gdsc = inf.get_gdsc_processed()
    gene_cols = [c for c in gdsc.columns if c not in
                 ("COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME", "PATHWAY_NAME",
                  "PUTATIVE_TARGET", "MIN_CONC", "MAX_CONC", "LN_IC50", "AUC")]
    gene_expr = {g: 5.0 for g in gene_cols}
    with pytest.raises(ValueError):
        inf.predict_dose_response_from_raw_expression(gene_expr, "NotARealDrug12345", "RandomForest")


def test_random_forest_dose_response_prediction_has_ensemble_spread():
    import inference as inf
    gdsc = inf.get_gdsc_processed()
    row = gdsc.iloc[0]
    result = inf.predict_dose_response_from_processed_row(row, row["DRUG_NAME"], "RandomForest", target="AUC")
    assert "ensemble_spread_std" in result
    assert result["ensemble_spread_std"] >= 0
    assert result["ensemble_spread_min"] <= result["predicted_value"] <= result["ensemble_spread_max"]


def test_predict_resistance_from_processed_row_returns_resistant_or_sensitive():
    import inference as inf
    gdsc = inf.get_gdsc_processed()
    row = gdsc.iloc[0]
    result = inf.predict_resistance_from_processed_row(row, row["DRUG_NAME"], "RandomForest")
    assert result["predicted_label"] in ("Resistant", "Sensitive")
    assert set(result["class_probabilities"]) == {"Resistant", "Sensitive"}
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-6


def test_predict_resistance_raises_missing_genes_error():
    import inference as inf
    gdsc = inf.get_gdsc_processed()
    row = gdsc.iloc[0].copy()
    _, entry = inf.load_resistance_model("RandomForest")
    gene_cols = [c for c in entry["feature_columns"] if c not in ("MIN_CONC", "MAX_CONC", "PATHWAY_NAME")]
    dropped_gene = gene_cols[0]
    row = row.drop(labels=[dropped_gene])
    with pytest.raises(inf.MissingGenesError) as exc:
        inf.predict_resistance_from_processed_row(row, row.get("DRUG_NAME", gdsc.iloc[0]["DRUG_NAME"]), "RandomForest")
    assert dropped_gene in exc.value.missing


def test_predict_resistance_from_raw_expression_rejects_unknown_drug():
    import inference as inf
    _, entry = inf.load_resistance_model("RandomForest")
    gene_cols = [c for c in entry["feature_columns"] if c not in ("MIN_CONC", "MAX_CONC", "PATHWAY_NAME")]
    gene_expr = {g: 5.0 for g in gene_cols}
    with pytest.raises(ValueError):
        inf.predict_resistance_from_raw_expression(gene_expr, "NotARealDrug12345", "RandomForest")


def test_load_resistance_model_raises_for_unknown_model_name():
    import inference as inf
    with pytest.raises(FileNotFoundError):
        inf.load_resistance_model("NotARealModel")


def test_resistance_pipeline_falls_back_to_global_importance_not_misaligned_linear_contribution():
    """Regression test for the fix in inference._score_classification: the
    resistance pipeline uses a ColumnTransformer (named step "pre", one-hot
    encoding PATHWAY_NAME), not a plain StandardScaler (named step
    "scaler"), so a coefficient-times-value local explanation would be
    misaligned with entry["feature_columns"]. Must fall back to
    global_importance_only instead of crashing or fabricating a wrong
    explanation, even for LogisticRegression (which has .coef_)."""
    import inference as inf
    gdsc = inf.get_gdsc_processed()
    row = gdsc.iloc[0]
    result = inf.predict_resistance_from_processed_row(row, row["DRUG_NAME"], "LogisticRegression")
    assert result["explanation_type"] == "global_importance_only"
    assert "local_explanation" not in result
