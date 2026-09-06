"""
Loads raw data snapshots (already retrieved by src/fetch_tcga.py and
src/fetch_gene_maps.py, plus the manually downloaded GDSC files in
data/raw/gdsc/) into tidy pandas DataFrames. Performs NO cleaning, imputation
or transformation -- see src/preprocessing.py for that.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TCGA_DIR = DATA_DIR / "raw" / "tcga"
GDSC_DIR = DATA_DIR / "raw" / "gdsc"

# GDSC uses two different spellings for "colorectal adenocarcinoma" across its
# two files: "COREAD" in GDSC2_fitted_dose_response.csv's TCGA_DESC column, and
# "COAD/READ" in Cell_Lines_Details.xlsx's "Cancer Type (matching TCGA label)"
# column. Both are documented here to avoid a silent empty-filter bug.
GDSC_DOSE_RESPONSE_LABEL = "COREAD"
GDSC_CELL_LINE_LABEL = "COAD/READ"


def load_ferroptosis_gene_list():
    return pd.read_csv(DATA_DIR / "unified_gene_signature.csv")


# ---------------------------------------------------------------------------
# TCGA COADREAD (patient-level)
# ---------------------------------------------------------------------------

def load_tcga_clinical_wide():
    long = pd.read_csv(TCGA_DIR / "clinical_patient_long.csv")
    wide = long.pivot_table(
        index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first"
    )
    wide.columns.name = None
    return wide.reset_index()


def load_tcga_clinical_sample_wide():
    """Sample-level clinical attributes (MSI scores, mutation count, sample type, ...)."""
    long = pd.read_csv(TCGA_DIR / "clinical_sample_long.csv")
    wide = long.pivot_table(
        index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first"
    )
    wide.columns.name = None
    return wide.reset_index()


def load_tcga_samples():
    return pd.read_csv(TCGA_DIR / "samples.csv")


def load_tcga_expression_wide():
    """Returns a (sample x gene symbol) matrix of RNA-seq V2 RSEM values."""
    long = pd.read_csv(TCGA_DIR / "mrna_expression_long.csv")
    gene_map = pd.read_csv(TCGA_DIR / "gene_entrez_map.csv")[["entrezGeneId", "hugoGeneSymbol"]]
    long = long.merge(gene_map, on="entrezGeneId", how="left")
    wide = long.pivot_table(index="sampleId", columns="hugoGeneSymbol", values="value", aggfunc="first")
    wide.columns.name = None
    return wide.reset_index()


# ---------------------------------------------------------------------------
# GDSC (colorectal cell lines)
# ---------------------------------------------------------------------------

def load_gdsc_cell_line_annotation():
    df = pd.read_excel(GDSC_DIR / "Cell_Lines_Details.xlsx", sheet_name="Cell line details")
    df = df.rename(
        columns={
            "Sample Name": "cell_line_name",
            "COSMIC identifier": "COSMIC_ID",
            "Cancer Type\n(matching TCGA label)": "tcga_desc",
            "Microsatellite \ninstability Status (MSI)": "msi_status",
            "GDSC\nTissue descriptor 1": "tissue_1",
            "GDSC\nTissue\ndescriptor 2": "tissue_2",
            "Gene Expression": "has_expression",
            "Drug\nResponse": "has_drug_response",
        }
    )
    df["COSMIC_ID"] = df["COSMIC_ID"].astype("Int64")
    return df[
        ["cell_line_name", "COSMIC_ID", "tcga_desc", "msi_status", "tissue_1", "tissue_2",
         "has_expression", "has_drug_response"]
    ]


def load_gdsc_colorectal_cell_lines():
    ann = load_gdsc_cell_line_annotation()
    return ann[ann["tcga_desc"] == GDSC_CELL_LINE_LABEL].reset_index(drop=True)


def load_gdsc_dose_response(tcga_desc=GDSC_DOSE_RESPONSE_LABEL):
    df = pd.read_csv(GDSC_DIR / "GDSC2_fitted_dose_response.csv")
    if tcga_desc is not None:
        df = df[df["TCGA_DESC"] == tcga_desc].reset_index(drop=True)
    return df


def load_gdsc_expression_for_genes(ensembl_ids, cosmic_ids=None):
    """
    Streams the large (~305MB) GDSC RMA expression matrix and returns only the
    requested Ensembl gene rows (and optionally only the requested COSMIC_ID
    columns), avoiding a full in-memory load of all ~17,700 genes x 1,018 lines.
    """
    path = GDSC_DIR / "sanger1018_brainarray_ensemblgene_rma.txt"
    header = pd.read_csv(path, sep="\t", nrows=0)
    all_cols = header.columns.tolist()  # ["ensembl_gene", "906826", "687983", ...]

    if cosmic_ids is not None:
        wanted = {str(int(c)) for c in cosmic_ids}
        use_cols = [all_cols[0]] + [c for c in all_cols[1:] if c in wanted]
    else:
        use_cols = all_cols

    ensembl_set = set(ensembl_ids)
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", usecols=use_cols, chunksize=2000):
        chunks.append(chunk[chunk["ensembl_gene"].isin(ensembl_set)])
    result = pd.concat(chunks, ignore_index=True)
    return result.set_index("ensembl_gene")


def load_gene_ensembl_map():
    return pd.read_csv(GDSC_DIR / "gene_ensembl_map.csv")
