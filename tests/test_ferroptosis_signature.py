from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def test_literature_source_file_exists_and_has_required_columns():
    df = pd.read_csv(DATA_DIR / "ferroptosis_genes_literature.csv")
    required = {"gene", "category", "source_type", "paper_or_review", "journal", "year", "doi_or_url", "selection_reason"}
    assert required.issubset(df.columns)
    assert len(df) > 0


def test_every_gene_has_a_documented_source():
    df = pd.read_csv(DATA_DIR / "ferroptosis_genes_literature.csv")
    assert df["paper_or_review"].notna().all()
    assert df["gene"].notna().all()
    assert (df["gene"].str.strip() != "").all()


def test_unified_signature_is_deduplicated_union():
    import ferroptosis
    unified = ferroptosis.build_unified_signature()
    assert unified["gene"].is_unique
    lit = pd.read_csv(DATA_DIR / "ferroptosis_genes_literature.csv")
    assert set(unified["gene"]) == set(lit["gene"])


def test_cross_study_recurrent_genes_have_n_sources_greater_than_one():
    import ferroptosis
    unified = ferroptosis.build_unified_signature()
    recurrent = unified[unified["cross_study_recurrent"]]
    assert (recurrent["n_sources"] > 1).all()
