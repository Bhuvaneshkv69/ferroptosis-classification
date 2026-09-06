"""
Builds the reproducible ferroptosis-related gene signature used throughout the
pipeline, starting strictly from documented literature sources
(data/ferroptosis_genes_literature.csv) rather than an arbitrary hard-coded list.

Selection method (documented in data/gene_selection_method.md):
  1. Start from two literature tiers:
       - core_mechanism genes: canonical ferroptosis regulators named in the
         Stockwell-lab mechanistic review (Jiang, Stockwell & Conrad, Nat Rev
         Mol Cell Biol 2021).
       - crc_prognostic_signature genes: genes that appear in the final,
         published LASSO-Cox ferroptosis-related prognostic signatures of four
         independent colon/colorectal cancer studies (2021).
  2. Take the union of both tiers as the "unified ferroptosis signature".
  3. Record cross-study recurrence (frequency) for interpretability -- a gene
     reported in >1 independent CRC study is flagged as higher-confidence.
  4. Downstream (preprocessing.py), genes are further intersected with what is
     actually measurable in the acquired TCGA / GDSC expression matrices; any
     gene dropped at that stage is logged, never silently substituted.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LITERATURE_PATH = DATA_DIR / "ferroptosis_genes_literature.csv"
UNIFIED_PATH = DATA_DIR / "unified_gene_signature.csv"
METHOD_DOC_PATH = DATA_DIR / "gene_selection_method.md"


def build_unified_signature():
    lit = pd.read_csv(LITERATURE_PATH)

    freq = lit.groupby("gene").size().rename("n_sources")
    categories = lit.groupby("gene")["category"].agg(lambda s: ";".join(sorted(set(s))))
    papers = lit.groupby("gene")["paper_or_review"].agg(lambda s: " | ".join(sorted(set(s))))

    unified = (
        pd.DataFrame({"n_sources": freq, "category": categories, "supporting_sources": papers})
        .reset_index()
        .sort_values(["n_sources", "gene"], ascending=[False, True])
        .reset_index(drop=True)
    )
    unified["cross_study_recurrent"] = unified["n_sources"] > 1
    unified.to_csv(UNIFIED_PATH, index=False)
    return unified


def write_method_doc(unified: pd.DataFrame):
    n_core = (unified["category"].str.contains("core_mechanism")).sum()
    n_crc = (unified["category"].str.contains("crc_prognostic_signature")).sum()
    n_total = len(unified)
    recurrent = unified[unified["cross_study_recurrent"]]["gene"].tolist()

    doc = f"""# Ferroptosis Gene Signature -- Selection Methodology

## Source
`data/ferroptosis_genes_literature.csv` -- one row per (gene, source publication) pair,
manually compiled from real, retrievable publications (title, journal, year and
DOI/URL recorded per row). No gene in this project's signature was chosen
without a documented source.

## Two literature tiers

1. **Core mechanistic ferroptosis regulators** ({n_core} genes) -- from the
   mechanistic review Jiang, Stockwell & Conrad, *"Ferroptosis: mechanisms,
   biology and role in disease"*, Nature Reviews Molecular Cell Biology, 2021.
   These genes (e.g. GPX4, SLC7A11, ACSL4) define the core lipid-peroxidation /
   iron-handling machinery of ferroptosis and are not colorectal-cancer specific.

2. **Colorectal/colon-cancer prognostic signature genes** ({n_crc} genes) -- the
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
({n_total} unique genes total). Genes reported independently in more than one
of the four CRC studies are flagged `cross_study_recurrent = True` and treated
as higher-confidence during interpretation:

{', '.join(recurrent) if recurrent else '(none)'}

## Why a union, not an intersection

The four CRC studies each used different DEG-filtering and LASSO random seeds,
so their final signatures barely overlap ({', '.join(recurrent) if recurrent else 'only a few genes recur'}
out of {n_crc} total) -- this is a known reproducibility limitation of the
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
"""
    METHOD_DOC_PATH.write_text(doc, encoding="utf-8")


def main():
    unified = build_unified_signature()
    write_method_doc(unified)
    print(f"Unified ferroptosis gene signature: {len(unified)} genes -> {UNIFIED_PATH}")
    print(f"Methodology document -> {METHOD_DOC_PATH}")


if __name__ == "__main__":
    main()
