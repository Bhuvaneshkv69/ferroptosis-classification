# Review-2 Presentation Content

## Title
Designing a Pipeline for Ferroptosis Classification and Dose-Response Prediction

## Abstract
Ferroptosis is an iron-dependent form of regulated cell death mechanistically
linked to colorectal cancer biology. Published studies have built
ferroptosis-related gene-expression prognostic signatures for colorectal
cancer, but each derives its risk label from the same expression features it
is validated against, and none extends its gene panel to quantitative
drug-response prediction. This project builds a reproducible computational
pipeline that (1) curates a literature-traceable, 35-gene ferroptosis
signature from one mechanistic review and four independent 2021 colorectal-
cancer prognostic-signature studies; (2) uses it to classify 594 TCGA
colorectal-adenocarcinoma patients by MSI status (a clinically independent,
non-circular target, ROC-AUC 0.917) and, separately, by an exploratory
Cox-regression-derived survival risk group; and (3) predicts quantitative
in-vitro drug-sensitivity (LN(IC50), AUC) for 44 GDSC2 colorectal cancer cell
lines across 268 drugs and four representative chemotherapy/ferroptosis-
relevant agents, achieving R²=0.617 on entirely held-out cell lines. All
results are interpreted with permutation importance and SHAP, and all
limitations — including the circularity of the survival-risk task, the lack
of an external validation cohort, and the in-vitro (not clinical) nature of
the drug-response predictions — are explicitly documented.

## Problem Statement
Existing ferroptosis-related colorectal cancer research is fragmented across
three separate threads — mechanistic biology, prognostic gene signatures, and
generic drug-response modeling — with no publicly documented pipeline that
combines a literature-anchored ferroptosis gene panel with both a
non-circular classification task and a genuine dose-response regression task
in one reproducible, tested, end-to-end system.

## Aim
To design, implement, and evaluate a reproducible machine-learning pipeline
that uses ferroptosis-related gene expression to classify colorectal cancer
molecular subgroups and predict quantitative drug-response in colorectal
cancer cell lines.

## Objectives
1. Curate a literature-traceable ferroptosis gene signature.
2. Build leakage-safe preprocessing pipelines for two independent public
   datasets (TCGA patients, GDSC cell lines).
3. Compare raw gene features against engineered signature-score features.
4. Classify colorectal tumors by MSI status and by ferroptosis survival-risk
   group using four ML model families.
5. Predict quantitative drug-response (LN_IC50, AUC) using three regression
   model families, on a pooled multi-drug design and four representative
   drugs individually.
6. Interpret all models with permutation importance and SHAP.
7. Document every dataset, methodological, and scope decision honestly.

## Literature Review (summary — full detail in `docs/RESEARCH_GAP.md`)
- Jiang, Stockwell & Conrad (2021) — core ferroptosis mechanism review.
- Qi et al. (2021, Front. Genetics) — 9-gene CRC prognostic signature.
- Shi et al. (2021, Front. Cell Dev. Biol.) — 4-gene CRC prognostic signature.
- Shao et al. (2021, Front. Oncology) — 10-gene CRC prognostic signature.
- Wang et al. (2021, World J. Surg. Oncol.) — 7-gene CRC prognostic signature.

## Research Gap
See `docs/RESEARCH_GAP.md` for the full comparison table. Summary: published
CRC ferroptosis signatures (a) validate a risk label built from the same
features being scored (circular), and (b) never extend to quantitative
drug-response prediction. This project addresses both gaps.

## Proposed Methodology
See `docs/METHODOLOGY.md` for the full pipeline diagram and per-stage detail:
gene curation → preprocessing (TCGA + GDSC, independently) → feature
engineering (raw vs. engineered) → classification (4 models × 3 experiments)
→ dose-response regression (3 models, pooled + 4 representative drugs) →
interpretability (permutation importance + SHAP).

## System Architecture
```
Literature -> unified gene signature (35 genes)
   -> TCGA COADREAD pipeline -> MSI / risk-group classification
   -> GDSC2 COAD/READ pipeline -> LN_IC50 / AUC regression
   -> shared evaluation + interpretability layer
```

## Dataset
TCGA COADREAD (PanCancer Atlas, via cBioPortal API, 594 patients) and GDSC2
colorectal cell lines (Sanger FTP, 44-51 cell lines, 268 drugs). Full detail,
citations, and license notes: `docs/DATASET.md`.

## Tools / Software / Hardware
Python 3.14, NumPy, pandas, scikit-learn, XGBoost, SHAP, lifelines,
matplotlib, scipy, requests, pytest. Standard laptop/desktop CPU — no GPU or
specialized hardware required; full pipeline runs in under 8 minutes on a
consumer machine. Full pinned versions: `requirements.txt`.

## Implementation Status
81% (13/16 planned components fully implemented, tested, and demonstrated).
Full breakdown: `IMPLEMENTATION_STATUS.md`.

## Preliminary Results
MSI classification ROC-AUC 0.917 (Logistic Regression); pooled dose-response
R²=0.617 (Random Forest, LN_IC50); representative single-drug models R²
0.21-0.32. Full results, tables, and discussion: `RESULTS.md`.

## Performance Analysis
Random Forest and XGBoost consistently outperform Linear Regression /
unregularized linear models wherever feature count approaches sample size
(notably the representative single-drug models, n≈44, p=31). Logistic
Regression outperforms tree ensembles for MSI classification, suggesting a
largely additive/linear relationship between ferroptosis gene expression and
MSI status in this cohort. Feature engineering (collapsing genes to a
compact score) *reduces* performance — a genuine negative finding. Full
analysis: `RESULTS.md`.

## Comparison with Existing Methods
See `docs/RESEARCH_GAP.md` comparison table: this project adds a non-circular
classification target and genuine dose-response regression, both absent from
the four reviewed CRC ferroptosis-signature papers.

## Expected Outcome
A reproducible, tested, open pipeline and this documentation set, usable as
the basis for a short IEEE-conference-style paper (not yet submitted — see
`docs/FUTURE_WORK.md` item 9). No publication, product deployment, or patent
is claimed at this stage.

## Individual Contribution (suggested — team to confirm actual ownership)
- **Mayank Bhardwaj (23BKT0106):** Ferroptosis gene curation and literature
  review (`data/ferroptosis_genes_literature.csv`,
  `data/gene_selection_method.md`, `docs/RESEARCH_GAP.md`); TCGA data
  acquisition and preprocessing (`src/fetch_tcga.py`, `src/preprocessing.py`
  TCGA half).
- **Sovan Sekar Sarma (23BKT0057):** Classification module and feature
  engineering (`src/feature_engineering.py`, `src/classification.py`);
  interpretability for the classification task.
- **Bhuvanesh K.V. (23BKT0138):** GDSC data acquisition and preprocessing,
  dose-response regression module (`src/dose_response.py`,
  `src/preprocessing.py` GDSC half); evaluation utilities
  (`src/evaluation.py`) and overall pipeline integration (`main.py`).

All three members should review and validate the full pipeline and
documentation, since it was authored in one integrated engineering session.

## References
See `README.md` §References and `data/ferroptosis_genes_literature.csv`.
