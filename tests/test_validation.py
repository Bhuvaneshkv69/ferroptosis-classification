import pandas as pd

import validation as val


def test_required_gene_panel_matches_unified_signature():
    genes = val.load_required_gene_panel()
    assert len(genes) > 0
    assert len(genes) == len(set(genes))


def test_valid_expression_dataframe_passes():
    genes = val.load_required_gene_panel()
    df = pd.DataFrame([{g: 500.0 for g in genes}])
    result = val.validate_expression_dataframe(df, genes)
    assert result["is_valid"]
    assert result["missing_genes"] == []
    assert result["n_genes_usable"] == len(genes)


def test_missing_genes_are_detected_and_block_validity():
    genes = val.load_required_gene_panel()
    df = pd.DataFrame([{g: 500.0 for g in genes[3:]}])
    result = val.validate_expression_dataframe(df, genes)
    assert not result["is_valid"]
    assert set(result["missing_genes"]) == set(genes[:3])


def test_duplicate_gene_columns_are_detected():
    genes = val.load_required_gene_panel()
    df = pd.DataFrame([[500.0] * (len(genes) + 1)], columns=genes + [genes[0]])
    result = val.validate_expression_dataframe(df, genes)
    assert not result["is_valid"]
    assert genes[0] in result["duplicate_gene_columns"]


def test_non_numeric_values_are_detected():
    genes = val.load_required_gene_panel()
    df = pd.DataFrame([{g: 500.0 for g in genes}]).astype(object)
    df.loc[0, genes[0]] = "oops"
    result = val.validate_expression_dataframe(df, genes)
    assert not result["is_valid"]
    assert genes[0] in result["non_numeric_genes"]


def test_zero_variance_columns_are_a_warning_not_an_error_for_multi_sample_batch():
    genes = val.load_required_gene_panel()
    rows = [{g: 500.0 for g in genes} for _ in range(3)]
    rows[1][genes[0]] = 500.0  # identical across all 3 rows -> zero variance
    df = pd.DataFrame(rows)
    result = val.validate_expression_dataframe(df, genes)
    assert result["is_valid"]  # zero variance alone doesn't block validity
    assert genes[0] in result["zero_variance_genes"]


def test_empty_dataframe_is_invalid():
    result = val.validate_expression_dataframe(pd.DataFrame(), val.load_required_gene_panel())
    assert not result["is_valid"]
    assert result["n_samples"] == 0


def test_make_template_csv_has_exactly_the_required_columns():
    genes = val.load_required_gene_panel()
    tmpl = val.make_template_csv(genes)
    assert list(tmpl.columns) == genes
    assert len(tmpl) == 1
