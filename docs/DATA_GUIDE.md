# Data Guide

This is a short index; the full detail lives in the documents linked below
so it is written and maintained in exactly one place each.

- **What datasets are used, why, exact accession IDs, citations, license
  terms:** `docs/DATASET.md`
- **What each `data/` subdirectory holds and the role of each file in the
  pipeline:** `data/README.md`
- **How the raw data was cleaned, transformed, and split (leakage-safe):**
  `docs/METHODOLOGY.md`, `docs/DATA_LEAKAGE_AUDIT.md`
- **How to obtain a new gene-expression file the application will accept:**
  `data/README.md` §"Preprocessing requirements," and the CSV template
  downloadable from the Data / Ferroptosis Classification / Drug Response
  pages of the running application.
- **Configurable dataset parameters** (MSI threshold, drug-coverage
  minimum, split sizes, random seed): `configs/data_config.yaml`.
- **How the ferroptosis gene panel itself was chosen:**
  `data/gene_selection_method.md`.
