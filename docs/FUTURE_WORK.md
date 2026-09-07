# Future Work

Listed in rough priority order for Project-II / a follow-on IEEE paper.

**Update (this session, 2026-09-07):** items 1, 2, 3, 10, and 11 are now
**done** — see `RESULTS.md` §4b/4c/4d and `docs/DATA_LEAKAGE_AUDIT.md` for
full results and honest discussion of what each one found. Model
persistence, a model registry, input validation, and a report generator
were also added (none were on this list because the prior iteration did
not yet have an application to persist models for).

1. ~~**External validation cohort.**~~ **Done this session.** Validated the
   MSI classifier on CPTAC-2 Prospective Colon Cancer (Cell 2019), an
   independent cohort/consortium. External ROC-AUC (0.94–0.96) matched or
   exceeded internal TCGA held-out ROC-AUC, but decision-threshold accuracy
   did not transfer as cleanly (see `RESULTS.md` §4d for the honest
   discussion, including SVM's threshold miscalibration on the external
   cohort). `src/fetch_external_validation.py`, `src/external_validation.py`.
2. ~~**Whole-transcriptome baseline.**~~ **Done this session.** Fetched all
   ~19,200 protein-coding genes' expression for the same TCGA cohort;
   PCA(50, train-fold-only)-reduced classifiers **outperformed** the
   ferroptosis-panel-only classifier (best ROC-AUC 0.986 vs. 0.917) — an
   honest, expected finding that quantifies (rather than undermines) the
   ferroptosis panel's contribution (see `RESULTS.md` §4c).
   `src/fetch_whole_transcriptome.py`, `src/whole_transcriptome_baseline.py`.
3. ~~**Raw dose-response curves.**~~ **Done this session.** Downloaded and
   streamed `GDSC2_public_raw_data_24Jul22.csv` (2.1GB) to build real,
   measured dose-response curves (normalized to per-plate negative
   controls) for the 4 representative drugs, replacing the illustrative
   IC50-only reconstructions (kept alongside for comparison). Methodology
   is a documented simplification of GDSC's own `gdscIC50` R pipeline (no
   positional/edge-effect correction) — see `RESULTS.md` §4b and
   `docs/LIMITATIONS.md`. `src/process_raw_dose_response.py`.
4. **Regularized per-drug regression.** Replace plain Linear Regression with
   Ridge/Lasso/ElasticNet for the representative single-drug models, where
   n ≈ p made unregularized linear regression unstable (strongly negative
   R² observed — see `RESULTS.md`); also explore partial least squares.
5. **Ferroptosis-inducer-specific screen.** Extend the drug panel search to
   CTRP or DepMap PRISM datasets, which include Erastin/RSL3-class compounds
   absent from GDSC, to more directly test ferroptosis-inducer sensitivity.
6. **Multi-omics integration.** Incorporate TCGA copy-number, methylation, or
   mutation data (already available via the same cBioPortal study) alongside
   expression for a richer feature set.
7. **Formal survival analysis reporting.** Add Kaplan-Meier curves and
   log-rank tests for the Cox-derived risk groups (currently only the
   Cox model coefficients and median-split classification are reported) to
   more fully mirror and critique the published literature's validation style.
8. **Multiple-testing correction.** Apply Benjamini-Hochberg FDR correction to
   the univariate Cox gene screen instead of a raw p < 0.10 cutoff.
9. **Manuscript preparation.** Package the MSI-classification result (the
   project's strongest, non-circular finding) as a short IEEE-conference-style
   paper submission — "Status: Pending — not yet submitted, no publication
   claimed" (see `docs/REVIEW_2_CONTENT.md`, Expected Outcome).
10. ~~Web demo / interactive dashboard.~~ **Done this session** -- see `app/`.
11. ~~**Per-fold engineered-feature fitting.**~~ **Done this session.** See
    `docs/DATA_LEAKAGE_AUDIT.md` finding #1 for the fix and its measured
    (surprising, honestly reported) impact.
12. **Cross-cohort threshold recalibration.** The external validation
    (item 1) found that classification thresholds calibrated on TCGA do not
    transfer cleanly to CPTAC even though ranking (ROC-AUC) does; implement
    and evaluate a recalibration step (e.g. Platt scaling refit per cohort,
    or reporting only rank-based metrics for cross-cohort use).
13. **Broader cross-dataset generalization checks.** Systematically test
    classifier behavior under varying degrees of missing genes and sample
    counts (beyond the single CPTAC cohort and the missing-gene handling
    `src/validation.py` already performs at input time).

None of the above is claimed as completed anywhere else in this repository
except items explicitly marked done above, cross-referenced from
`PROJECT_STATUS.md` and `IMPLEMENTATION_STATUS.md`.
