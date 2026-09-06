import pandas as pd
import pytest

import data_loader as dl


def test_tcga_clinical_wide_has_msi_and_survival_columns():
    clin = dl.load_tcga_clinical_sample_wide()
    assert "MSI_SCORE_MANTIS" in clin.columns
    patient = dl.load_tcga_clinical_wide()
    assert "OS_MONTHS" in patient.columns
    assert "OS_STATUS" in patient.columns


def test_tcga_expression_wide_has_ferroptosis_genes_as_columns():
    genes = dl.load_ferroptosis_gene_list()["gene"].tolist()
    expr = dl.load_tcga_expression_wide()
    overlap = set(genes) & set(expr.columns)
    assert len(overlap) >= 30, f"expected most of {len(genes)} genes present, got {len(overlap)}"
    assert expr.shape[0] > 500


def test_gdsc_colorectal_cell_line_filter_is_non_empty():
    crc = dl.load_gdsc_colorectal_cell_lines()
    assert len(crc) > 0
    assert crc["COSMIC_ID"].notna().all()


def test_gdsc_dose_response_filter_matches_colorectal_label():
    dr = dl.load_gdsc_dose_response()
    assert (dr["TCGA_DESC"] == "COREAD").all()
    assert len(dr) > 0


def test_gdsc_expression_loader_returns_requested_genes_only():
    gmap = dl.load_gene_ensembl_map()
    ids = gmap[gmap["is_primary"]]["ensembl_gene_id"].dropna().head(3).tolist()
    expr = dl.load_gdsc_expression_for_genes(ids)
    assert set(expr.index).issubset(set(ids))
