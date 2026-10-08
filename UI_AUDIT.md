# UI/UX Audit — Ferroptosis Research Workspace

**Date:** 2026-09-17
**Scope:** the Streamlit application (`app/`) only. The ML pipeline, data
processing, model registry, inference, validation, and reporting backends
(`src/*.py`) were reviewed for correctness but not rebuilt — they are the
real, tested engine this audit's UI changes sit on top of (76/76 tests
passing at the start of this session; see `IMPLEMENTATION_STATUS.md` /
`PROJECT_STATUS.md` for the pipeline's own history).

## 1. Starting condition

The application was **not** a placeholder or an academic dump of metrics.
A prior, uncommitted session had already rebuilt it from the original
8-page "Preprocessing / Classification / Drug Response / ..." numbered
Streamlit app into a cleaner 7-page workspace (`Overview`, `Data`,
`Analysis`, `Drug Response`, `Models`, `Results`, `Documentation`) with:

- A real design system (`app/theme.py`): restrained palette, compact stat
  tiles, status pills, no gradients/emoji-heavy decoration.
- A two-layer result pattern already in place on `Analysis` and
  `Drug Response`: a headline metric + short caption, then an expander for
  feature contributions ("Why this prediction?") and another for raw
  metrics/hyperparameters ("Model details" / "About this analysis").
- Honest language throughout: "Model probability" (not "confidence"),
  "reference label" (not "ground truth diagnosis"), explicit non-causal
  framing on every explainability panel ("association ... not proof of
  biological causality").
- Real CSV upload + validation (`src/validation.py`) with human-readable
  errors (missing genes listed by name, non-numeric columns named,
  duplicate columns flagged) and a downloadable template — no raw Python
  tracebacks surfaced to the user.
- A real analysis-session log (`results/analysis_sessions.json`) and a
  real HTML report generator (`src/reporting.py`), both driven by actual
  inference calls, not fabricated history.
- A persistent, small disclaimer in the sidebar on every page.
- No hard-coded predictions, metrics, or feature importances anywhere in
  `app/` — every page reads `data/processed/`, `results/`, `tables/`,
  `figures/`, or `models/registry.json` at render time.

This is a genuinely strong starting point. The gaps below are refinements,
not a rebuild.

## 2. Gaps found against the requested experience

| # | Gap | Where | Severity |
|---|---|---|---|
| 1 | No way to enter a sample's molecular values by hand — "manual/guided" input meant "upload a CSV," full stop. A user without a CSV file had no path into the model. | `analysis.py`, `drug_response.py` | High |
| 2 | No visible progress narration while a prediction runs (validate → prepare features → score → explain) — the button just produces a result instantly, which reads as opaque for a "guided" experience. | both analysis pages | Medium |
| 3 | Every result had technical detail (metrics, hyperparameters, contributions) but no single plain-language "What does this mean?" sentence separated from the technical expanders — a first-time reader had to infer the interpretation from a caption. | `analysis.py`, `drug_response.py` | Medium |
| 4 | Nav label "Analysis" reads as generic; doesn't signal "this is where you start." | `main.py` | Low |
| 5 | Home page CTA said "Open Analysis" rather than "Start New Analysis." | `overview.py` | Low |
| 6 | No `docs/CSV_FORMAT.md` — the required-columns/units/scale contract for CSV upload existed only implicitly in code comments (and differs between the classification path, which expects raw untransformed counts, and the drug-response path, which expects already-log2 GDSC-scale values — an easy point of confusion left undocumented). | docs | Medium |
| 7 | No link between the two analysis types after a result — a user who just classified a sample had to rediscover Drug Response via the sidebar rather than being offered the natural next step. | `analysis.py` | Low |
| 8 | Stale generated artifacts from prior manual test sessions (21 old HTML reports, a 20-entry session log dated 2026-09-07/08/09) were sitting untracked in `results/`, which would clutter a fresh demo's "Recent analyses" list and report archive. | `results/` | Low |
| 9 | `results/reports/*.html` and `results/analysis_sessions.json` were untracked but also not gitignored — ambiguous whether they're meant to persist across clones. | `.gitignore` | Low |

## 3. What was deliberately left unchanged

- **No question → prediction mapping was added.** The only new input path
  (manual gene-value entry) asks for exactly the features the persisted
  model's `feature_columns` require — nothing else. No lifestyle/symptom
  questionnaire exists anywhere in the app, and none was added, because
  the underlying models are gene-expression classifiers/regressors with no
  trained relationship to non-molecular variables.
- **No fabricated confidence intervals or calibration claims.** "Model
  probability" and "ensemble spread" labels are kept exactly as the prior
  session defined them, because that's what the persisted models actually
  output.
- **Results/Models pages (technical depth) were not simplified away** —
  per the brief, the technical detail stays, just one layer deeper than
  the headline result.
- **The dual-dataset separation (TCGA patients vs. GDSC cell lines) was
  not merged** for the manual-entry form; classification manual entry
  feeds the TCGA-trained model, drug-response manual entry feeds the
  GDSC-trained model, exactly matching how cohort/upload input already
  worked.

## 4. Changes implemented this session

1. **Manual "Research Sample Input" entry** added to both `Analysis` and
   `Drug Response` as a third input mode alongside "existing sample" and
   "upload CSV." Every field is one of the model's actual
   `feature_columns` (35 ferroptosis genes for classification; the same
   panel for dose response); each field defaults to the cohort's real
   population median (computed live from the processed data, not
   invented) and shows the population's 5th–95th percentile range as a
   help tooltip, so a user editing a handful of genes they care about
   doesn't have to guess plausible values for the other 30. An optional
   "Sample label" text field is explicitly captioned as "for your
   reference only — not used by the model" rather than silently implying
   it's a feature.
2. **Visible run progress** using `st.status(...)`, narrating the real
   steps already being executed (validating input → preparing features →
   running the model → generating the explanation) instead of an
   instantaneous result.
3. **Explicit "What does this mean?" panel** added directly under the
   headline result on both pages, before the technical expanders —
   2–3 plain-language sentences, distinct from the existing "Why this
   result?" (feature-contribution) panel.
4. **Cross-links between analyses**: a classification result now offers
   "Continue to Drug Response" for the same sample where applicable.
5. Nav label `Analysis` → `New Analysis`; Overview CTA `Open Analysis` →
   `Start New Analysis`.
6. **`docs/CSV_FORMAT.md`** added: required columns, the two distinct
   input scales (raw counts for classification vs. log2-RMA for dose
   response) and why they differ, missing-value handling, and a worked
   example.
7. Stale test-session artifacts removed from `results/`; both
   `results/analysis_sessions.json` and `results/reports/*.html` added to
   `.gitignore` as regenerable runtime output (matching how
   `models/*.joblib` is already treated).

## 5. Verification performed

- `pytest tests/ -q` — full suite, before and after changes.
- `streamlit.testing.v1.AppTest` smoke tests extended to exercise the new
  manual-entry input mode on both `Analysis` and `Drug Response` (a real
  prediction through the real persisted model, not a mock).
- Manual walkthrough of both complete workflows (CSV and guided/manual)
  against the running application, including the deliberately-broken
  cases (missing genes, non-numeric values, empty file) to confirm every
  error surfaces as a human-readable message, never a raw traceback.

See the end-of-session report for the full before/after test output and
the final acceptance-criteria checklist.

## 6. Addendum: literature-driven additions (same session, following request)

After the UI polish pass above, seven additional papers were provided as
this project's literature basis, with a requested "Complete Flow": Patient
Risk Group → Survival Prediction → Drug Response → Resistance Prediction →
Dose-Response Estimation. Full citations and an honest paper-by-paper
mapping (what is directly implemented vs. conceptually related but not
built) are in `docs/REFERENCES.md`. Three real, backend-and-UI additions
followed from that mapping, not just documentation:

1. **Non-circular Prognosis Prediction validation** (papers 1-3):
   `feature_engineering.evaluate_prognosis()` / `plot_kaplan_meier()` check
   the existing Cox ferroptosis risk score — previously only validated by
   the circular Experiment 3 classification accuracy — against real
   held-out patient survival directly (concordance index 0.692, log-rank
   p=0.039, n=118). New "Prognosis" tab on the Results page; Experiment 3's
   framing was corrected to point to this non-circular check rather than
   implying its classification accuracy is the prognostic evidence.
2. **New drug-resistance classification module** (papers 6-7):
   `src/drug_resistance.py` — a real, trained Resistant/Sensitive
   classifier (pooled, all drugs; plus an oxaliplatin-specific variant),
   with "Resistant"/"Sensitive" defined transparently from this project's
   own GDSC LN_IC50 distribution (train-fold-only median split), not from
   unverified paper-specific thresholds. Persisted to the model registry,
   served live from the Drug Response page, and given its own Results and
   Models tabs. The honest result (pooled: near-chance; oxaliplatin-only:
   real signal) is reported as-is, including in the UI.
3. **`docs/REFERENCES.md`** and a new Documentation → References tab
   linking every new addition back to the paper that motivated it.

Verification: `python src/classification.py`, `python src/drug_resistance.py`,
and `python scripts/train_and_persist.py` were run end-to-end (not just
imported); 21 new/updated automated tests were added
(`tests/test_drug_resistance.py`, additions to `tests/test_feature_engineering.py`,
`tests/test_inference.py`, `tests/test_app_smoke.py`) and pass alongside the
full existing suite (95/95). A real regression bug this work surfaced and
fixed: `inference._score_classification`'s linear-explanation path assumed
every classification pipeline uses a plain `StandardScaler` step; the new
resistance pipeline's `ColumnTransformer` (one-hot-encoded drug pathway)
would have produced a silently misaligned per-gene contribution table for
LogisticRegression. Fixed to fall back to the existing global-importance
explanation for that pipeline shape, with a regression test.
