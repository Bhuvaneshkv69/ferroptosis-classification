# Literature Review, Research Gap, and Comparison with Existing Work

## What existing studies have done

Ferroptosis regulator genes are well characterized mechanistically (Jiang,
Stockwell & Conrad, *Nat Rev Mol Cell Biol* 2021). Separately, at least four
independent 2021 studies built ferroptosis-related gene-expression prognostic
signatures specifically for colon/colorectal cancer using TCGA data:

| Study | Journal, Year | Method | Task | Ferroptosis component | Drug-response component | Interpretability | External validation | Reported limitation |
|---|---|---|---|---|---|---|---|---|
| Qi et al. | Front. Genetics, 2021 | Univariate Cox + LASSO-Cox on TCGA COAD | 9-gene prognostic risk score, survival risk stratification | Yes (FerrDb-derived candidate pool) | No | Gene coefficients only | Internal (TCGA) split only | No independent cohort; risk label built from same features scored |
| Shi et al. | Front. Cell Dev. Biol., 2021/2022 | Cox regression | 4-gene prognostic signature, invasion/metastasis association | Yes | No | Gene coefficients only | Internal only | Small gene panel; no drug-response link |
| Shao et al. | Front. Oncology, 2021 | LASSO-Cox on TCGA CRC | 10-gene prognostic signature | Yes | No | Gene coefficients only | Internal only | No drug-response link |
| Wang et al. | World J. Surg. Oncol., 2021 | Univariate Cox (259 FerrDb genes) + LASSO-Cox | 7-gene prognostic signature | Yes (259-gene FerrDb pool) | No | Gene coefficients only | Internal only | No drug-response link; small final panel |
| **This project** | B.Tech Project, 2026 | RF/XGB/LR/SVM classification + RF/XGB/LR regression | MSI classification (primary, non-circular) + exploratory Cox risk group + drug-response (LN_IC50/AUC) regression | Yes (35-gene union of the above + core mechanistic genes) | **Yes** (GDSC2 colorectal cell lines, 4 models, 4 representative drugs) | Permutation importance + SHAP | Held-out split within each dataset (not yet an independent external cohort — see `FUTURE_WORK.md`) | Explicitly documented (`LIMITATIONS.md`); no external cohort yet |

## What none of the reviewed studies do

Every one of the four CRC ferroptosis-signature papers reviewed:
1. builds a **survival-risk classification** label from the same expression
   features it evaluates against (a circularity this project also reproduces,
   but explicitly flags rather than presenting as a headline result — see
   `docs/LIMITATIONS.md`), and
2. stops at prognosis/risk stratification — **none of them extends the
   ferroptosis gene panel to a quantitative drug-response / dose-response
   model.**

## Research gap

> Existing research separately investigates (a) ferroptosis-related molecular
> signatures and prognosis in colorectal cancer, using labels derived from the
> same expression data being modeled, and (b) drug-response / IC50 prediction
> in cancer cell lines using arbitrary (often whole-transcriptome or
> mutation-based) feature sets **not** anchored to a documented, literature-
> traceable ferroptosis gene panel. No reviewed study combines a
> literature-anchored ferroptosis gene signature with **both** a
> non-circular classification task **and** a quantitative dose-response
> regression task in one reproducible computational pipeline.

## How this project addresses the gap

1. Builds one literature-traceable ferroptosis gene signature (35 genes, full
   provenance in `data/gene_selection_method.md`) used consistently across
   both tasks.
2. Adds a **non-circular** classification target (MSI status, independently
   measured) alongside the literature-style circular risk score, so the
   project's headline result is not subject to the same circularity critique
   as the reviewed papers.
3. Extends the same gene panel to a **genuine quantitative dose-response
   regression** task (GDSC2 colorectal cell lines, `LN_IC50`/`AUC`), which none
   of the four reviewed CRC ferroptosis-signature papers attempt.
4. Reports interpretability (permutation importance + SHAP) for both tasks,
   with an explicit association-vs-causation caveat that the reviewed papers'
   simple Cox-coefficient reporting does not carry.
5. Documents every dataset, gene-selection, and modeling decision so the
   pipeline can be independently re-run end to end (`python main.py`).

## What remains genuinely unresolved (not claimed as solved here)

- No external validation cohort (see `FUTURE_WORK.md`).
- No whole-transcriptome comparison baseline.
- No wet-lab/functional validation of any implicated gene.

No claim of "absolute novelty" is made — this project's contribution is the
**combination and honest evaluation** of tasks that existing literature keeps
separate, not the invention of new ferroptosis biology or a new ML algorithm.
