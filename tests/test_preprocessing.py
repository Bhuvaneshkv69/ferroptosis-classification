import numpy as np
import pandas as pd
import pytest

from preprocessing import MSI_MANTIS_THRESHOLD, split_gdsc, split_tcga


@pytest.fixture(scope="module")
def tcga_processed():
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data/processed/tcga_coadread_processed.csv"
    if not path.exists():
        pytest.skip("Run src/preprocessing.py first to generate processed TCGA data")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def gdsc_processed():
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data/processed/gdsc_colorectal_dose_response_processed.csv"
    if not path.exists():
        pytest.skip("Run src/preprocessing.py first to generate processed GDSC data")
    return pd.read_csv(path)


def test_tcga_one_row_per_patient(tcga_processed):
    assert tcga_processed["patientId"].is_unique


def test_tcga_msi_label_matches_mantis_threshold(tcga_processed):
    labeled = tcga_processed.dropna(subset=["MSI_SCORE_MANTIS"])
    expected = np.where(labeled["MSI_SCORE_MANTIS"] >= MSI_MANTIS_THRESHOLD, "MSI-H", "MSS/MSI-L")
    assert (labeled["msi_label"].values == expected).all()


def test_tcga_split_has_no_row_overlap(tcga_processed):
    gene_cols = [c for c in tcga_processed.columns if c.isupper() and tcga_processed[c].dtype != object][:5]
    split = split_tcga(tcga_processed, "msi_label", gene_cols)
    train_idx = set(split["X_train"].index)
    val_idx = set(split["X_val"].index)
    test_idx = set(split["X_test"].index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_gdsc_group_split_holds_out_entire_cell_lines(gdsc_processed):
    gene_cols = [c for c in gdsc_processed.columns if c not in
                 ("COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME", "PATHWAY_NAME",
                  "PUTATIVE_TARGET", "MIN_CONC", "MAX_CONC", "LN_IC50", "AUC")][:5]
    split = split_gdsc(gdsc_processed, gene_cols, "LN_IC50")
    train_lines = set(split["meta_train"]["COSMIC_ID"])
    test_lines = set(split["meta_test"]["COSMIC_ID"])
    assert train_lines.isdisjoint(test_lines), "a cell line leaked across the GroupShuffleSplit"


def test_gdsc_no_duplicate_drug_cell_line_pairs(gdsc_processed):
    assert not gdsc_processed.duplicated(["DRUG_NAME", "COSMIC_ID"]).any()
