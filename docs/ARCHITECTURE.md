# Architecture

## Directory layout

```
configs/            data_config.yaml, model_config.yaml -- centralized thresholds, paths, seed,
                     hyperparameter-grid documentation (single source of truth)
data/
  raw/{tcga,gdsc}/   unmodified snapshots (see data/README.md)
  processed/         cleaned + feature-engineered datasets + preprocessing metadata JSON
  external/          reserved for a future independent validation cohort
  metadata/          reserved for cross-cutting metadata
  unified_gene_signature.csv, ferroptosis_genes_literature.csv, gene_selection_method.md
src/
  config.py          loads configs/*.yaml once, exposes RANDOM_SEED etc. to every module
  data_loader.py      raw file I/O (no transformation)
  preprocessing.py    cleaning, transformation, target construction, leakage-safe splitting
  feature_engineering.py   engineered ferroptosis signature score + Cox risk score
  ferroptosis.py       builds the literature-traceable gene signature
  classification.py    4 models x 3 experiments, CV hyperparameter search, real evaluation
  dose_response.py     3 models x 2 pooled targets + 4 representative single-drug models
  evaluation.py         shared metric computation + plotting, used by both above
  interpretability.py   permutation importance + SHAP for both tasks
  model_registry.py     persists fitted Pipelines to models/, tracks version/metrics/hyperparams
  inference.py           load-model + score-one-sample functions (no training)
  validation.py           input validation for user-supplied expression data
  reporting.py             builds a self-contained HTML report from real output files
  fetch_tcga.py, fetch_gene_maps.py   raw data acquisition (network calls)
scripts/
  train_and_persist.py   runs classification.main()/dose_response.main() and registers every
                          fitted model -- the ONLY place models are written to models/
app/
  main.py                Streamlit entry point (Page 1: Home)
  common.py              shared bootstrap + cached data/model accessors for every page
  pages/1-7_*.py          Data, Preprocessing, Ferroptosis Classification, Drug Response,
                          Model Performance, Interpretability, Report
models/
  registry.json           metadata for every persisted model artifact
  <task>/<experiment>/<Model>__<version>.joblib   fitted sklearn Pipelines (gitignored, regenerable)
results/, tables/, figures/    CSV metrics, comparison tables, PNG plots (all from real runs)
tests/                    pytest suite (unit + integration + Streamlit AppTest end-to-end)
docs/                     this document + DATASET.md, METHODOLOGY.md, LIMITATIONS.md,
                          DATA_LEAKAGE_AUDIT.md, MODEL_CARD.md, DEMO_GUIDE.md,
                          API_OR_PIPELINE_GUIDE.md, RESEARCH_GAP.md, FUTURE_WORK.md
main.py                   full pipeline entry point (stages 1-7)
PROJECT_AUDIT.md          audit performed at the start of this session
PROJECT_STATUS.md         live phase-by-phase completion tracking
```

## Data flow

```
data/raw/  --[preprocessing.py]-->  data/processed/  --[feature_engineering.py]-->
    data/processed/*_features.csv
        |
        +--[classification.py / dose_response.py]--> results/, tables/, figures/
        |         (fitted models returned in-memory, same call evaluated)
        |
        +--[scripts/train_and_persist.py]--> models/<task>/<experiment>/*.joblib + models/registry.json
                |
                +--[src/inference.py]--> app/pages/{3,4}_*.py --> Streamlit UI
```

The application never re-trains anything. Every prediction shown in the UI
comes from `src/inference.py` loading a `models/registry.json` entry and
calling `.predict()`/`.predict_proba()` on the exact fitted `Pipeline`
object that was evaluated by `classification.py`/`dose_response.py` --
there is no separate "deployment model" that could silently diverge from
the reported metrics.

## Why a script-based core pipeline, not a class-heavy framework

Each stage (`preprocessing.py`, `classification.py`, `dose_response.py`,
`interpretability.py`) is a small number of top-level functions plus a
`main()`, runnable standalone (`python src/dose_response.py`) or via
`main.py`/`scripts/train_and_persist.py`. This keeps every stage
independently testable (see `tests/`) and independently re-runnable without
re-executing the whole pipeline, while the `sklearn.Pipeline` objects
themselves carry all fitted state (scalers, encoders) so no separate
"preprocessor artifact" bookkeeping is needed at inference time.

## Two independent modeling problems, one shared feature source

TCGA (patient-level, RNA-seq) and GDSC (cell-line-level, microarray) are
never pooled into one model -- see `docs/DATASET.md` §"Why these two
datasets are never merged." The only thing shared between the two halves of
the pipeline is the ferroptosis gene list itself
(`data/unified_gene_signature.csv`), independently fit/scaled per dataset.

## Configuration

`configs/data_config.yaml` and `configs/model_config.yaml` are the single
source of truth for thresholds (MSI MANTIS cutoff, drug-coverage minimum),
paths, the random seed, and (as documentation) the hyperparameter search
spaces actually defined in code. `src/config.py` loads them once;
`preprocessing.py`, `classification.py`, `dose_response.py`, and
`interpretability.py` all import `RANDOM_SEED` and the relevant thresholds
from there rather than hard-coding local constants.
