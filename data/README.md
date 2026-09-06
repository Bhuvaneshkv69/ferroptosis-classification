# `data/` directory guide

| Directory | Contents | Produced by | Tracked in git? |
|---|---|---|---|
| `raw/tcga/` | Unmodified snapshots of the cBioPortal API response for `coadread_tcga_pan_can_atlas_2018` (clinical, sample, expression, gene-map CSVs) + `fetch_metadata.json` (retrieval timestamp). | `src/fetch_tcga.py` (`python main.py --fetch`) | Yes |
| `raw/gdsc/` | Manually downloaded GDSC2 snapshot: `GDSC2_fitted_dose_response.csv`, `Cell_Lines_Details.xlsx`, `sanger1018_brainarray_ensemblgene_rma.txt(.gz)`, `gene_ensembl_map.csv`. See `docs/DATASET.md` for exact source URLs. | Manual download (no GDSC API exists) + `src/fetch_gene_maps.py` for the gene map | Yes |
| `processed/` | Cleaned, transformed, feature-engineered datasets ready for modeling (`tcga_coadread_processed.csv`, `tcga_coadread_features.csv`, `gdsc_colorectal_dose_response_processed.csv`) plus their preprocessing-metadata JSON files (exact gene lists used, missingness stats, sample counts). | `src/preprocessing.py`, `src/feature_engineering.py` (`python main.py`) | Yes |
| `external/` | Reserved for a future independent validation cohort (e.g. a second public CRC gene-expression/MSI dataset) -- see `docs/FUTURE_WORK.md` item 1. Empty until that work is done; nothing here is currently used by the pipeline. | (future work) | N/A |
| `metadata/` | Reserved for cross-cutting metadata that doesn't belong to one specific processed file (e.g. a future data dictionary). Currently unused; per-dataset metadata instead lives next to each processed file in `processed/` (`*_metadata.json`) so it can never drift from the file it describes. | -- | N/A |
| `ferroptosis_genes_literature.csv` | One row per (gene, source publication) -- the raw literature evidence for every gene in the signature. | Manually compiled, see `gene_selection_method.md` | Yes |
| `unified_gene_signature.csv` | The deduplicated, unioned 35-gene ferroptosis signature used as the candidate feature set for both classification and dose-response regression. | `src/ferroptosis.py` (`python src/ferroptosis.py`) | Yes |
| `gene_selection_method.md` | Full methodology for how the 35-gene signature was built. | `src/ferroptosis.py` | Yes |

## Preprocessing requirements before a new raw file can enter the pipeline

- **TCGA-style clinical/expression data:** must be shaped as the cBioPortal `clinical-data` and
  `molecular-data` API responses are (long format: one row per (sample, attribute) or
  (sample, gene) pair) -- see `src/fetch_tcga.py` for the exact schema `src/data_loader.py` expects.
- **GDSC-style dose-response/expression data:** `GDSC2_fitted_dose_response.csv` must contain
  `COSMIC_ID, CELL_LINE_NAME, DRUG_ID, DRUG_NAME, PATHWAY_NAME, MIN_CONC, MAX_CONC, LN_IC50, AUC,
  TCGA_DESC`; the expression file must be a tab-separated (gene x cell-line) matrix with Ensembl gene
  IDs as the first column.
- **A brand-new expression file for inference** (Streamlit app "Data" or "Ferroptosis Classification" /
  "Drug Response" pages): a CSV with one row per sample and one column per gene symbol (the 35-gene
  panel in `unified_gene_signature.csv`); validated live by `src/validation.py` before any model sees it.

## Role in the pipeline

```
raw/  --[src/preprocessing.py]-->  processed/  --[src/classification.py, src/dose_response.py]-->
    results/, figures/, tables/  --[scripts/train_and_persist.py]-->  models/
```

## Licensing / access

- TCGA PanCancer Atlas data via cBioPortal: open public access for research use, no login/DUA required.
- GDSC2 data via Sanger Institute FTP: released for unrestricted academic/research use.
See `docs/DATASET.md` for full citation and accession detail.
