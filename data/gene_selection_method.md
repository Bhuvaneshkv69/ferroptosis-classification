# Ferroptosis Gene Signature -- Selection Methodology

## Source
`data/ferroptosis_genes_literature.csv` -- one row per (gene, source publication) pair,
manually compiled from real, retrievable publications (title, journal, year and
DOI/URL recorded per row). No gene in this project's signature was chosen
without a documented source.

## Two literature tiers

1. **Core mechanistic ferroptosis regulators** (8 genes) -- from the
   mechanistic review Jiang, Stockwell & Conrad, *"Ferroptosis: mechanisms,
   biology and role in disease"*, Nature Reviews Molecular Cell Biology, 2021.
   These genes (e.g. GPX4, SLC7A11, ACSL4) define the core lipid-peroxidation /
   iron-handling machinery of ferroptosis and are not colorectal-cancer specific.

2. **Colorectal/colon-cancer prognostic signature genes** (27 genes) -- the
   union of the *final, published* gene sets from four independent 2021
   ferroptosis-related prognostic-signature studies in colon/colorectal cancer,
   each built by the original authors via univariate Cox filtering of FerrDb-derived
   candidate genes followed by LASSO-Cox regression on TCGA COAD/READ data:
   - Qi et al., Frontiers in Genetics 2021 (9 genes)
   - Shi et al., Frontiers in Cell and Developmental Biology 2021 (4 genes)
   - Shao et al., Frontiers in Oncology 2021 (10 genes)
   - Wang et al., World Journal of Surgical Oncology 2021 (7 genes, from an
     initial FerrDb-sourced pool of 259 ferroptosis-related genes)

## Unified signature

The **union** of tier 1 and tier 2 forms `data/unified_gene_signature.csv`
(35 unique genes total). Genes reported independently in more than one
of the four CRC studies are flagged `cross_study_recurrent = True` and treated
as higher-confidence during interpretation:

(none)

## Why a union, not an intersection

The four CRC studies each used different DEG-filtering and LASSO random seeds,
so their final signatures barely overlap (only a few genes recur
out of 27 total) -- this is a known reproducibility limitation of the
LASSO-Cox ferroptosis-signature literature, discussed in `docs/LIMITATIONS.md`.
Taking the union keeps every literature-reported candidate available to our own
feature-selection step (see `src/feature_engineering.py`), rather than
arbitrarily privileging one paper's random seed over another's.

## What this signature is (and is not) used for

- It is the **candidate feature pool** for classification (ferroptosis-related
  gene expression in TCGA COADREAD) and for the drug-response regression module
  (ferroptosis-related gene expression in GDSC colorectal cell lines).
- It is **not** claimed to be an exhaustive or definitive ferroptosis gene set;
  FerrDb V2 alone lists >1000 regulators. This project intentionally works with
  a literature-anchored subset, documented here in full, rather than the entire
  FerrDb catalogue, to keep the feature set traceable to specific, checkable
  publications.

## Reproducing this file

```
python src/ferroptosis.py
```
regenerates `unified_gene_signature.csv` and this document deterministically
from `ferroptosis_genes_literature.csv`.
