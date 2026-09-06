# Dataset Documentation

This project uses **two independent, publicly available datasets** for two
deliberately separate tasks. They are never merged into a single model —
see `METHODOLOGY.md` for why mixing patient tumors with cell lines would be
scientifically invalid.

---

## Dataset A — TCGA Colorectal Adenocarcinoma (patient-level, classification)

| Field | Value |
|---|---|
| Name | Colorectal Adenocarcinoma (TCGA, PanCancer Atlas) |
| Accession / study ID | `coadread_tcga_pan_can_atlas_2018` |
| Access route | cBioPortal public REST API (`https://www.cbioportal.org/api`) |
| Underlying data | TCGA PanCancer Atlas (https://gdc.cancer.gov/about-data/publications/pancanatlas) |
| Citation | The Cancer Genome Atlas Research Network et al., *Cell* 2018 (PanCancer Atlas marker papers; PMIDs on file in the cBioPortal study record) |
| Biological context | Primary colon (COAD) and rectal (READ) adenocarcinoma tumors |
| Samples | 594 patients; 592 with matched RNA-seq expression; 555 with an MSI MANTIS score |
| Features retrieved | RNA-seq V2 (RSEM) expression for the 35-gene unified ferroptosis signature; patient- and sample-level clinical annotations (MSI scores, overall survival, stage, subtype, mutation count, etc.) |
| Data format | JSON via REST API, saved as CSV snapshots in `data/raw/tcga/` |
| Target variable(s) | (1) MSI-H vs MSS/MSI-L (MANTIS score ≥ 0.4, Kautto et al., *Oncotarget* 2017); (2) exploratory Cox-regression ferroptosis risk group (High/Low) |
| Why appropriate | Colorectal cancer is the project's stated primary application; TCGA COADREAD is the standard, most-cited public cohort for CRC molecular subtyping and is the same cohort used by all four ferroptosis-signature papers reviewed in this project |
| License / access | cBioPortal serves TCGA PanCancer Atlas data under open public access for research use; no login or data-use agreement required for this de-identified, already-public dataset |

Retrieval script: `src/fetch_tcga.py` (re-run with `python main.py --fetch`).
Raw snapshots: `data/raw/tcga/*.csv` + `fetch_metadata.json` (retrieval timestamp, exact genes/IDs requested).

---

## Dataset B — GDSC2 Colorectal Cancer Cell Lines (cell-line-level, dose-response)

| Field | Value |
|---|---|
| Name | Genomics of Drug Sensitivity in Cancer, GDSC2 dataset |
| Accession / source | Sanger Institute FTP: `ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/` |
| Files used | `current_release/GDSC2_fitted_dose_response_24Jul22.csv`, `current_release/Cell_Lines_Details.xlsx`, `release-7.0/sanger1018_brainarray_ensemblgene_rma.txt.gz` |
| Citation | Iorio F. et al., *Cell* 2016, "A Landscape of Pharmacogenomic Interactions in Cancer" (the release-7.0 expression array and cell-line annotation originate from this landmark GDSC study) |
| Biological context | Human colorectal-adenocarcinoma-derived cell lines (GDSC label `COAD/READ` / `COREAD`) screened against a large anti-cancer compound panel |
| Cell lines | 51 annotated as COAD/READ; 49 with expression data; 44 with both expression and drug-response data surviving preprocessing |
| Drugs | 268 compounds tested on ≥ 40 of the 46 screened colorectal lines (coverage filter, see `METHODOLOGY.md`); representative drugs analyzed individually: **5-Fluorouracil, Oxaliplatin, Irinotecan, Sorafenib** (three are backbone CRC chemotherapy agents; Sorafenib is a well-documented pharmacological ferroptosis inducer) |
| Response metric | `LN_IC50` (natural-log IC50, µM) and `AUC` (area under the fitted dose-response curve, 0–1) — both are the standard fitted summary statistics released by GDSC, not raw per-well viability |
| Data format | CSV / XLSX, gene expression matrix as tab-separated text (Ensembl gene ID × COSMIC cell-line ID) |
| Target variable(s) | `LN_IC50`, `AUC` (regression) |
| Why appropriate | GDSC2 is the largest public pharmacogenomic screen with paired expression + drug-response data for cancer cell lines, and its `COAD/READ` cell lines are the standard in-vitro model for colorectal drug-sensitivity studies |
| License / access | GDSC data are released for unrestricted academic/research use (Sanger Institute, Wellcome Sanger); no login required |

**Known data-coverage limitation:** the expression array (`sanger1018_brainarray_ensemblgene_rma.txt.gz`, a 2013-era Affymetrix custom-CDF platform) does not include probes for 4 of the 35 unified signature genes (`ATP6V1G2`, `NCOA4`, `SLC2A3`, `TXNIP`) — verified by direct lookup, not a mapping bug. 31/35 genes are used for all GDSC-side modeling. See `LIMITATIONS.md`.

**What was intentionally NOT downloaded:** `GDSC2_public_raw_data_24Jul22.csv` (~2 GB; per-well raw viability readings underlying the fitted curves) was not retrieved due to its size. This means the "dose-response curves" produced by this project (`figures/dose_response_curve_*.png`) are mathematically **reconstructed from the fitted IC50 parameter**, not plotted from raw viability measurements — clearly labeled as illustrative on the figures themselves and in `LIMITATIONS.md`.

Retrieval: files were downloaded directly from the Sanger FTP mirror (`data/raw/gdsc/`); this is a manual one-time step documented here since it requires no authentication but is not re-triggered by `--fetch` (see `README.md` for the exact URLs).

---

## Why these two datasets are never merged

TCGA COADREAD samples are **primary human tumors** profiled by RNA-seq;
GDSC cell lines are **immortalized in-vitro models** profiled by microarray.
Their expression values live on different technical scales, and a cell line's
IC50 to a drug is not a property any TCGA patient record contains. Pooling
them would conflate two different biological units of analysis and two
different measurement technologies. Instead:

- **Classification** (Experiments 1–3) runs entirely within Dataset A.
- **Dose-response regression** (pooled + representative-drug experiments) runs
  entirely within Dataset B.
- The only thing shared between the two pipelines is the **ferroptosis gene
  list** itself (`data/unified_gene_signature.csv`), used as the candidate
  feature set in both, independently fit and z-scored per dataset.

## Ferroptosis gene signature (feature source for both datasets)

See `data/gene_selection_method.md` for the full, literature-traceable
methodology. Summary: `data/unified_gene_signature.csv` contains 35 genes —
8 core mechanistic ferroptosis regulators (Jiang, Stockwell & Conrad, *Nat Rev
Mol Cell Biol* 2021) plus the union of 27 genes appearing in four published
2021 colon/colorectal-cancer ferroptosis prognostic-signature papers.
