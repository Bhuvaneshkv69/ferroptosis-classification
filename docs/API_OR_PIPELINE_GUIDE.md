# Pipeline / CLI Guide

This project exposes its functionality as Python modules + CLI entry points
(no HTTP API is deployed -- the Streamlit application is the user-facing
interface and imports these modules directly). This guide documents every
command and the reusable functions behind them.

## CLI commands

| Command | What it does | Outputs |
|---|---|---|
| `python main.py` | Runs the full pipeline: gene signature → [optional fetch] → preprocessing → feature engineering → classification → dose-response → interpretability | `data/processed/`, `results/`, `figures/`, `tables/`, `docs/review_evidence/` |
| `python main.py --fetch` | Same, but re-downloads TCGA + gene-map data from cBioPortal/MyGene.info first (GDSC files must still be placed manually per `docs/DATASET.md`) | + fresh `data/raw/tcga/*` |
| `python src/ferroptosis.py` | Rebuilds the gene signature from `data/ferroptosis_genes_literature.csv` | `data/unified_gene_signature.csv`, `data/gene_selection_method.md` |
| `python src/preprocessing.py` | Preprocessing only, both datasets | `data/processed/*_processed.csv` + metadata JSON |
| `python src/feature_engineering.py` | Feature engineering only (requires `tcga_coadread_processed.csv` to already exist) | `data/processed/tcga_coadread_features.csv` |
| `python src/classification.py` | Classification only (requires processed+featured TCGA data) | `results/classification_results.csv`, `figures/*classification*`, `tables/classification_model_comparison.csv` |
| `python src/dose_response.py` | Dose-response regression only (requires processed GDSC data) | `results/regression_results.csv`, `results/representative_drug_results.csv`, `figures/*dose_response*` |
| `python src/interpretability.py` | Permutation importance + SHAP for both tasks | `results/feature_importance_*.csv`, `figures/shap_summary_*.png` |
| `python scripts/train_and_persist.py` | Trains + persists every deployable model to `models/` and `models/registry.json` | `models/<task>/<experiment>/*.joblib`, `models/registry.json` |
| `python src/fetch_external_validation.py` | Fetches the independent CPTAC-2 cohort from cBioPortal | `data/external/cptac_coad_2019/*` |
| `python src/external_validation.py` | Trains the 28-gene intersection variant on TCGA, scores it on CPTAC (no refitting on external data) | `results/external_validation_results.csv`, `figures/roc_curve_external_validation_cptac.png` |
| `python src/fetch_whole_transcriptome.py` | Fetches ~19,200 protein-coding genes' expression for the TCGA cohort (~5 min) | `data/raw/tcga/whole_transcriptome_expression_long.csv` |
| `python src/whole_transcriptome_baseline.py` | Trains the whole-transcriptome (PCA-50) MSI baseline, compares to the ferroptosis panel | `results/whole_transcriptome_baseline_results.csv`, `tables/ferroptosis_vs_whole_transcriptome_comparison.csv` |
| `python src/process_raw_dose_response.py` | Streams the 2.1GB raw GDSC file into real, measured dose-response curves (requires manually downloading `GDSC2_public_raw_data_24Jul22.csv` to `data/raw/gdsc/` first, see `docs/DATASET.md`) | `data/processed/gdsc_raw_dose_response_curves.csv`, `figures/dose_response_curve_real_*.png` |
| `python -m src.model_registry` (or `python src/model_registry.py`) | Lists every persisted model artifact | stdout |
| `pytest tests/ -q` | Runs the full test suite | pass/fail report |
| `streamlit run app/main.py` | Launches the interactive application | web UI at `localhost:8501` |

## Reusable functions (importable from `src/`)

```python
import sys; sys.path.insert(0, "src")

# Data
import data_loader as dl
dl.load_tcga_expression_wide()          # -> (sample x gene) DataFrame
dl.load_gdsc_dose_response()             # -> raw GDSC2 fitted dose-response rows

# Preprocessing (leakage-safe splits)
import preprocessing as pp
pp.split_tcga(df, target_col, feature_cols)   # stratified train/val/test
pp.split_gdsc(df, feature_cols, target_col)   # group split by COSMIC_ID

# Inference (load a persisted model, score one sample -- no training)
import inference as inf
inf.predict_msi_from_processed_row(row, model_name="LogisticRegression")
inf.predict_msi_from_raw_expression({"GPX4": 512.0, ...}, model_name="RandomForest")
inf.predict_dose_response_from_processed_row(row, drug_name, model_name="RandomForest", target="LN_IC50")
inf.predict_dose_response_from_raw_expression(gene_dict, drug_name, model_name="RandomForest")

# Validation
import validation as val
val.validate_expression_dataframe(df, required_genes)   # -> dict with is_valid/errors/warnings

# Model registry
import model_registry as reg
reg.list_models(task="classification", experiment="exp1_msi_raw_genes")
reg.latest_model(task="dose_response", experiment="pooled_LN_IC50", name="RandomForest")
reg.load_model(entry)   # -> fitted sklearn Pipeline

# Reporting
import reporting
reporting.generate_report()   # -> Path to a new self-contained HTML report
```

## Configuration

Edit `configs/data_config.yaml` (thresholds, paths, split sizes) or
`configs/model_config.yaml` (documented hyperparameter grids, which model is
"deployed" for interactive use) -- then re-run `python main.py` and
`python scripts/train_and_persist.py` so the change actually takes effect
(the application only ever reads what these scripts produced).
