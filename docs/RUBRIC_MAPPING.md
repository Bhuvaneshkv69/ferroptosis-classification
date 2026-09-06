# Rubric-to-Evidence Mapping

| Rubric | Evidence | File/Module | Result | Status |
|---|---|---|---|---|
| Project title, abstract, problem statement, objectives | Title/abstract/objectives written and consistent with implementation | `README.md`, `docs/REVIEW_2_CONTENT.md` | N/A | COMPLETE |
| Updated literature survey, clear research gap | 4 CRC ferroptosis-signature papers + 1 mechanistic review reviewed; gap articulated | `docs/RESEARCH_GAP.md` | Gap: no reviewed study combines non-circular classification + dose-response on one ferroptosis panel | COMPLETE |
| Proposed methodology and system architecture | Full pipeline diagram + per-stage description | `docs/METHODOLOGY.md` | N/A | COMPLETE |
| Dataset, tools, software, hardware | Accession IDs, sample counts, formats, license notes, tool versions | `docs/DATASET.md`, `requirements.txt` | 594 TCGA patients; 44-51 GDSC cell lines | COMPLETE |
| At least 75% implementation with demonstration | Per-module implemented/tested/demonstrated table | `IMPLEMENTATION_STATUS.md` | 13/16 = 81% | COMPLETE |
| Preliminary results, observations, performance analysis | Real metrics from real execution, discussed | `RESULTS.md`, `results/*.csv` | MSI ROC-AUC 0.917; dose-response R²=0.617 | COMPLETE |
| Comparison with existing methods/systems | Structured comparison table vs 4 published papers | `docs/RESEARCH_GAP.md` (table) | This project adds non-circular target + dose-response, absent elsewhere | COMPLETE |
| Progress toward IEEE paper/journal/product/patent | Honestly reported — not yet started beyond this pipeline | `docs/REVIEW_2_CONTENT.md` §Expected Outcome, `docs/FUTURE_WORK.md` item 9 | No submission made | PARTIAL (honestly scoped, not fabricated) |
| Updated documentation, source code, presentation, supporting evidence | Full doc set + preprocessing logs + test output | `docs/`, `docs/review_evidence/`, `README.md` | 29/29 tests pass | COMPLETE |
| Proper citations/references | Per-gene literature provenance + references list | `data/ferroptosis_genes_literature.csv`, `README.md` References | 9 primary sources cited | COMPLETE |
| Attendance ≥3 times/week | Not applicable to this deliverable — team/institutional record | N/A | N/A | N/A (outside pipeline scope) |
| Individual contribution of each member | Template with 3 named members and suggested task split | `docs/REVIEW_2_CONTENT.md` §Individual Contribution | Team to confirm actual split | PARTIAL (template provided, team must confirm) |
| Potential best-project quality | Real public data, real results, honest negative findings, full reproducibility, tests, interpretability, explicit limitations | Whole repository | N/A | COMPLETE (assessed by reviewer) |

## Notes on PARTIAL items

- **Publication/patent progress**: per the academic-honesty constraint in this
  project's build instructions, no IEEE submission, journal acceptance, or
  patent filing is claimed because none occurred. `docs/FUTURE_WORK.md` item 9
  proposes packaging the MSI-classification result as a short paper.
- **Individual contribution**: this was built as one continuous engineering
  session without the three team members present to divide work in real
  time; `docs/REVIEW_2_CONTENT.md` provides a defensible suggested split
  (data/ferroptosis curation, classification, dose-response) that the team
  should confirm reflects who actually drives each module going forward.
