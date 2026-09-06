# Project Status

**Project:** Designing a Pipeline for Ferroptosis Classification and Dose-Response Prediction
**Team:** Mayank Bhardwaj (23BKT0106), Sovan Sekar Sarma (23BKT0057), Bhuvanesh K.V. (23BKT0138)
**Faculty Guide:** Dr. Adrija Bhattacharya
**Status date:** 2026-09-07 (this session's end-to-end system upgrade)

## History

- **Prior session:** built the analysis pipeline from scratch (data
  acquisition, preprocessing, classification, dose-response regression,
  interpretability, 29 tests, documentation) starting from an empty
  directory. See the git history / `IMPLEMENTATION_STATUS.md`'s original
  81% figure for that session's own accounting.
- **This session:** audited that pipeline (`PROJECT_AUDIT.md`), fixed one
  data-leakage issue found during the audit, and built the missing
  application/persistence/registry/reporting layer on top of it so the
  pipeline is a usable end-to-end system, not just a script + notebook.

## Phase-by-phase status (this session)

| Phase | Status | Evidence |
|---|---|---|
| 1. Audit existing repository | **DONE** | `PROJECT_AUDIT.md` |
| 2. Fix data-leakage issue found in audit | **DONE** | `docs/DATA_LEAKAGE_AUDIT.md`, `src/preprocessing.py` diff, pipeline re-run with identical metrics |
| 3. `configs/` + config-driven thresholds/seed | **DONE** | `configs/*.yaml`, `src/config.py`, wired into `preprocessing.py`/`classification.py`/`dose_response.py`/`interpretability.py` |
| 4. Model registry + persistence | **DONE** | `src/model_registry.py`, `scripts/train_and_persist.py`, `models/registry.json` (18 real persisted artifacts) |
| 5. Inference layer (load model, score one sample) | **DONE** | `src/inference.py`, `tests/test_inference.py` |
| 6. Input validation | **DONE** | `src/validation.py`, `tests/test_validation.py` |
| 7. Report generator | **DONE** | `src/reporting.py`, produces a real self-contained HTML report from live output files |
| 8. Streamlit application (8 pages) | **DONE** | `app/`, wired only to `src/*` — no hard-coded numbers in the UI layer |
| 9. New documentation set | **DONE** | `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/DATA_LEAKAGE_AUDIT.md`, `docs/DEMO_GUIDE.md`, `docs/API_OR_PIPELINE_GUIDE.md`, `docs/DATA_GUIDE.md`, `docs/SCIENTIFIC_METHODS.md`, `data/README.md` |
| 10. Tests for all new modules + full suite | **DONE** | `tests/test_model_registry.py`, `tests/test_inference.py`, `tests/test_validation.py`, `tests/test_app_smoke.py` (Streamlit `AppTest`, exercises 2 real predictions + real report generation) — **61/61 passing** |
| 11. End-to-end execution & verification | **DONE** | see "What was actually run" below |
| 12. External (cross-cohort) validation | **NOT IMPLEMENTED** | see `docs/FUTURE_WORK.md` item 1 — genuinely not attempted; would require identifying and integrating a second public CRC cohort, out of scope for this session |
| 13. Per-fold fitting of engineered TCGA features (Exp. 2/3) | **NOT IMPLEMENTED (known, documented)** | `docs/DATA_LEAKAGE_AUDIT.md` finding #1, `docs/FUTURE_WORK.md` item 11 — low severity, does not change either experiment's conclusion |
| 14. Raw per-well GDSC dose-response curves | **NOT IMPLEMENTED** | unchanged from prior session — the ~2GB raw file was never downloaded; existing curves are labeled "illustrative" |

## What was actually run, this session, in order

```
python main.py                       # full pipeline re-run after the leakage fix — 29/29 tests before, 29/29 after
python scripts/train_and_persist.py  # x2 (once before, once after adding pos_label/neg_label metadata) — 18 artifacts registered
pytest tests/ -q                     # 61/61 passing (final)
streamlit run app/main.py            # launched; verified serving on localhost:8501
pytest tests/test_app_smoke.py -v    # Streamlit AppTest: all 8 pages load without exception,
                                      #   2 real predictions executed end-to-end, 1 real report generated
```

## Rubric component mapping

See `docs/RUBRIC_MAPPING.md` for the original mapping (still accurate for
the analysis-pipeline components). New capabilities added this session map
to the rubric as: "≥75% implementation with demonstration" now additionally
includes a working application (was previously CLI/notebook-only);
"comparison with existing methods" unchanged (`docs/RESEARCH_GAP.md`);
"progress toward IEEE paper/product" — a usable application now exists as
the "product" progress item, honestly still short of a manuscript
submission (see `docs/REVIEW_2_CONTENT.md`).

## What is explicitly still NOT done

- No external (non-TCGA) validation cohort for the classifier.
- No whole-transcriptome baseline comparison.
- No raw per-well GDSC dose-response curves (illustrative curves only, clearly labeled).
- Engineered-feature fitting (Exp. 2/3) is not yet moved inside the CV loop
  (low-severity, documented, doesn't change conclusions).
- No manuscript submitted, no patent filed — see `docs/REVIEW_2_CONTENT.md`.

Full detail on all of the above: `docs/FUTURE_WORK.md`, `PROJECT_AUDIT.md`.
