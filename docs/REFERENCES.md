# Literature basis and paper-to-pipeline mapping

The seven papers below were selected as this project's literature basis.
Bibliographic details were verified by web search against PubMed/PMC/journal
listings on 2026-09-17; please cross-check against your own reference
manager before final submission, since no independent verification beyond
search-engine metadata was performed.

This document states plainly, for each paper, **what is directly
implemented** in this pipeline vs. **what is conceptually related but not
built** — the project does not claim to reproduce any of these papers'
exact methods, gene lists, architectures, or numerical thresholds unless
explicitly said so below.

## The seven papers

1. Yue Q, Zhang Y, Wang F, Cao F, Duan X, Bai J. "Classification of
   colorectal carcinoma subtypes based on ferroptosis-associated molecular
   markers." *World Journal of Surgical Oncology*. 2022 Apr 12;20:115.
   doi:10.1186/s12957-022-02575-5.
2. Xu X, Zhang X, Lin Q, Qin Y, Liu Y, Tang W. "Integrated single-cell and
   bulk RNA sequencing analysis identifies a prognostic signature related to
   ferroptosis dependence in colorectal cancer." *Scientific Reports*. 2023
   Aug 4;13:12653. doi:10.1038/s41598-023-39412-y.
3. Nie J, Shan D, Li S, et al. "A Novel Ferroptosis Related Gene Signature
   for Prognosis Prediction in Patients With Colon Cancer." *Frontiers in
   Oncology*. 2021 May 11;11:654076. doi:10.3389/fonc.2021.654076.
4. Saranya KR, Vimina ER. "DRN-CDR: A cancer drug response prediction model
   using multi-omics and drug features." *Computational Biology and
   Chemistry*. 2024;112:108–. PMID:39191166.
5. Cao C, Zhao H, Wang J. "BANDRP: a bilinear attention network for
   anti-cancer drug response prediction based on fingerprint and
   multi-omics." *Briefings in Bioinformatics*. 2024 Nov;25(6):bbae493.
6. Cheng X, Zhao F, Ke B, Chen D, Liu F. "Harnessing Ferroptosis to
   Overcome Drug Resistance in Colorectal Cancer: Promising Therapeutic
   Approaches." *Cancers (Basel)*. 2023 Oct 30;15(21):5209.
   doi:10.3390/cancers15215209.
7. Zhong S, Wang Z, Yang J, Jiang D, Wang K. "Ferroptosis-related
   oxaliplatin resistance in multiple cancers: Potential roles and
   therapeutic implications." *Heliyon*. 2024 Sep 7;10(18):e37613.

## Honest paper-to-pipeline mapping

| # | Paper's contribution | What this project directly implements | What is NOT reproduced |
|---|---|---|---|
| 1 | Ferroptosis-associated molecular subtyping of CRC; high-risk vs. low-risk grouping (TCGA + GEO, consensus clustering, 682 patients) | The project's own literature-traceable 35-gene ferroptosis panel (`data/unified_gene_signature.csv`) and its Cox-regression-derived High/Low risk group (`src/feature_engineering.py`) — a real, train-fold-only-fit patient risk stratification, validated non-circularly by concordance index + log-rank test (see "Prognosis Prediction" below). | Paper 1's specific 4-subtype (C1–C4) consensus clustering, immune-microenvironment/CNV/mutation analysis, and its exact gene list are not reproduced — this project uses its own independently curated panel and a simpler two-group (not four-subtype) split. |
| 2 | Single-cell + bulk RNA-seq ferroptosis-dependence prognostic signature | The same Cox-based risk score above is this project's prognosis-prediction analogue — fit and validated on real TCGA overall-survival data (`OS_MONTHS`/`OS_EVENT`), reported honestly with the concordance index and log-rank p-value on a held-out test fold. | No single-cell RNA-seq data is used anywhere in this project (TCGA and GDSC are both bulk); the "FerrScore" single-cell dependence scoring method is not reproduced. |
| 3 | 5-gene (AKR1C1, ALOX12, FDFT1, ATP5MC3, CARS1) LASSO prognosis model for colon cancer | Notably, **4 of these 5 genes (AKR1C1, ALOX12, FDFT1, CARS1) are already members of this project's independently curated 35-gene panel** — a real, checkable overlap (`data/unified_gene_signature.csv`), not a coincidence claimed without evidence. The Cox risk-score model plays the same prognostic role paper 3's LASSO model does. | This project does not refit paper 3's exact 5-gene LASSO model or reproduce its specific AUC (0.668–0.686); ATP5MC3 is not in this project's panel. |
| 4 | DRN-CDR: deep ResNet regression over multi-omics (expression, mutation, methylation) + drug graph features (Uniform GCN) for IC50 prediction | The project's pooled dose-response regressors (`src/dose_response.py`: Linear/Random Forest/XGBoost over ferroptosis gene expression + drug pathway/concentration-range) address the same task — predicting a continuous IC50-like value from cell-line features and drug metadata — with classical ML rather than a deep ResNet, and gene-expression-only (no mutation/methylation) + pathway/concentration-range (no molecular-graph) drug features. | No deep ResNet, no Uniform Graph Convolution Network over drug structure, no mutation/methylation omics layers — explicitly out of scope (see `PROJECT_AUDIT.md`: "Molecular fingerprint/SMILES-based drug descriptors remain out of scope"). |
| 5 | BANDRP: bilinear attention network over drug fingerprints (SMILES) + multi-omics pathway-enrichment features | Conceptual motivation only: both projects predict drug sensitivity from a (cell-line features, drug features) pair. This project's drug representation is GDSC's own pathway/target/concentration-range annotation, not a SMILES fingerprint. | No bilinear attention network, no SMILES/fingerprint drug encoding, no genomic-mutation/methylation integration. |
| 6 | Review: ferroptosis-pathway activity associated with CRC chemoresistance; therapeutic strategies (ferroptosis inducers, iron/lipid-peroxidation interventions, combination therapy) | Directly motivates the new **drug-resistance classification module** (`src/drug_resistance.py`) added this session: a real, trained Resistant/Sensitive classifier per (cell line, drug) pair, from ferroptosis gene expression + drug pathway/concentration-range. | This project did not have verified access to paper 6's specific resistance thresholds or gene panel, so none are reproduced; "Resistant"/"Sensitive" is defined transparently from this project's own GDSC LN_IC50 distribution (train-fold-only per-drug median split), not from paper 6's numbers. No therapeutic-intervention modeling (inducers, combination therapy) is implemented — this project only classifies, it does not simulate treatment. |
| 7 | Ferroptosis (SLC7A11-GPX4 axis) specifically implicated in oxaliplatin resistance across cancers | Directly motivates the **oxaliplatin-specific resistance model** in `src/drug_resistance.py::run_oxaliplatin_resistance_model` — the same median-split Resistant/Sensitive task, restricted to Oxaliplatin, using only the ferroptosis gene panel (which includes `SLC7A11` and `GPX4`), leave-one-out CV (same small-n treatment already used for oxaliplatin's dose-response regression in `src/dose_response.py`). | No mechanistic (SLC7A11-GPX4 pathway activity, lncRNA regulation) modeling — this is a correlational classifier over expression values, not a causal or mechanistic model. Paper 7's specific reported findings/thresholds are not reproduced. |

## Complete flow, as actually implemented

```
Ferroptosis gene panel (data/unified_gene_signature.csv, this project's own
literature review -- 4 CRC prognostic-signature papers + 1 mechanistic
review, see data/gene_selection_method.md)
        |
        +-- TCGA COADREAD (patients) -----------------------------+
        |     |                                                    |
        |     +-- MSI classification (primary, non-circular)       |
        |     |                                                    |
        |     +-- Cox risk score -> Patient Risk Group              |
        |           -> Prognosis Prediction (concordance index +   |
        |              log-rank test on held-out patients;         |
        |              papers 1-3)                                 |
        |                                                          |
        +-- GDSC2 COAD/READ (cell lines) --------------------------+
              |
              +-- Dose-Response Prediction (LN_IC50 / AUC          |
              |    regression; papers 4-5 motivate this task,      |
              |    classical-ML implementation, not their deep-    |
              |    learning architectures)                         |
              |                                                    |
              +-- Drug Resistance classification (Resistant /      |
                   Sensitive, median-split; papers 6-7 motivate    |
                   this task, especially the oxaliplatin-specific  |
                   model)
```

This mirrors the "Patient Risk Group → Survival Prediction → Drug Response
→ Resistance Prediction → Dose-Response Estimation" flow the project's
literature basis was selected around, using this project's own real data,
models, and validation at every step — no step is a placeholder, and no
step claims a method it does not actually run.
