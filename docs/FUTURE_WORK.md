# Future Work

Listed in rough priority order for Project-II / a follow-on IEEE paper.

1. **External validation cohort.** Validate the MSI classifier on an
   independent public CRC cohort (e.g. a GEO colorectal expression series
   with MSI annotation) to test generalization beyond TCGA. Currently
   "Pending implementation" — not attempted in this iteration due to time
   constraints on identifying a GEO series with matching MSI ground truth.
2. **Whole-transcriptome baseline.** Fetch full RNA-seq expression (not just
   the 35-gene ferroptosis panel) for a subset of TCGA COADREAD to establish
   a whole-transcriptome classification baseline, quantifying exactly how
   much signal the ferroptosis-specific panel captures relative to an
   unconstrained feature set.
3. **Raw dose-response curves.** Retrieve `GDSC2_public_raw_data_24Jul22.csv`
   (~2 GB) to replace the illustrative, IC50-reconstructed dose-response
   curves with true per-concentration viability curves and refit 4-parameter
   logistic models directly.
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
10. **Web demo / interactive dashboard.** A lightweight Streamlit/Flask app
    exposing model predictions and SHAP explanations for a user-supplied
    expression profile, as a tangible "product" deliverable.

None of the above is claimed as completed anywhere else in this repository.
