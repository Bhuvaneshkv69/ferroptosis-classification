"""
Generates a single, self-contained HTML computational report from files the
pipeline actually produced -- results/*.csv, tables/*.csv, figures/*.png (
embedded as base64 so the report is one portable file), data/processed/*
metadata, and models/registry.json. No number in the generated report is
typed in by hand; everything is read from disk at generation time.
"""
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Template

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TEMPLATE = Template(r"""
<!doctype html><html><head><meta charset="utf-8">
<title>Ferroptosis Pipeline -- Computational Report</title>
<style>
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 1.6rem; } h2 { margin-top: 2.2rem; border-bottom: 2px solid #e2e2e2; padding-bottom: 0.3rem; }
table { border-collapse: collapse; width: 100%; margin: 0.8rem 0; font-size: 0.85rem; }
th, td { border: 1px solid #ddd; padding: 0.35rem 0.6rem; text-align: right; }
th { background: #f5f5f5; text-align: center; }
td:first-child, th:first-child { text-align: left; }
.meta { color: #666; font-size: 0.85rem; }
.warn { background: #fff8e1; border-left: 4px solid #f5b400; padding: 0.6rem 1rem; margin: 1rem 0; }
.figure { margin: 1rem 0; text-align: center; }
.figure img { max-width: 100%; border: 1px solid #ddd; }
.figcap { font-size: 0.8rem; color: #666; }
code { background: #f2f2f2; padding: 0.1rem 0.3rem; border-radius: 3px; }
</style></head><body>

<h1>Ferroptosis Classification &amp; Dose-Response Prediction &mdash; Computational Report</h1>
<p class="meta">Generated {{ generated_at }} from live pipeline outputs (results/, tables/, figures/, models/registry.json).
No number below is hand-entered.</p>

<div class="warn"><strong>Scope.</strong> This is a computational research report. It does not provide clinical
diagnosis, treatment recommendations, or patient-specific drug dosing. "Dose-response prediction" means predicting
an experimentally measured in-vitro drug-sensitivity summary statistic (LN_IC50 / AUC) for a cancer cell line, not
a human clinical dose. See docs/LIMITATIONS.md.</p></div>

<h2>1. Dataset</h2>
<table>
<tr><th>Dataset</th><th>Samples</th><th>Ferroptosis genes used</th><th>Genes missing from profile/array</th></tr>
<tr><td>TCGA COADREAD (patient-level)</td><td>{{ tcga_meta.n_samples }}</td><td>{{ tcga_meta.n_ferroptosis_genes_used }}</td>
<td>{{ tcga_meta.ferroptosis_genes_missing_from_profile | length }}</td></tr>
<tr><td>GDSC2 colorectal cell lines</td><td>{{ gdsc_meta.n_cell_lines }} lines / {{ gdsc_meta.n_rows }} (drug, line) rows</td>
<td>{{ gdsc_meta.ferroptosis_genes_used | length }}</td><td>{{ gdsc_meta.ferroptosis_genes_missing_from_array | length }}</td></tr>
</table>
<p class="meta">MSI-H/MSS threshold: MANTIS score &ge; {{ tcga_meta.msi_mantis_threshold }} (Kautto et al., Oncotarget 2017). Random seed: {{ tcga_meta.random_seed }}.</p>

<h2>2. Ferroptosis gene signature</h2>
<p>{{ gene_sig_n }} genes, literature-traceable (see data/gene_selection_method.md). Cross-study-recurrent genes: {{ recurrent_genes }}</p>

<h2>3. Classification results (MSI status, primary non-circular task)</h2>
{{ classification_table | safe }}
<div class="figure"><img src="data:image/png;base64,{{ figs.roc_exp1 }}"><div class="figcap">ROC curves, Experiment 1 (raw ferroptosis genes)</div></div>

<h2>4. Dose-response regression results (pooled, held-out cell lines)</h2>
{{ regression_table | safe }}
<div class="figure"><img src="data:image/png;base64,{{ figs.pred_vs_actual }}"><div class="figcap">Predicted vs. actual LN_IC50, Random Forest, held-out cell lines</div></div>

<h2>5. Interpretability (association, not causation)</h2>
<p>Top genes by permutation importance -- classification:</p>
{{ perm_clf_table | safe }}
<p>Top genes by permutation importance -- dose-response:</p>
{{ perm_reg_table | safe }}

<h2>6. Deployed model registry</h2>
<p>{{ n_registered_models }} model artifacts persisted. Deployed (interactive) models: classification &rarr;
<code>{{ deployed_clf }}</code>; dose-response &rarr; <code>{{ deployed_dr }}</code>.</p>

<h2>7. Limitations</h2>
<p>See <code>docs/LIMITATIONS.md</code> and <code>docs/DATA_LEAKAGE_AUDIT.md</code> for the full, honest list
(no external validation cohort; no whole-transcriptome baseline; illustrative, not raw, dose-response curves;
the ferroptosis risk-group classification experiment is circular by construction and excluded from interactive
prediction).</p>

<h2>8. Reproducibility</h2>
<p>Random seed {{ tcga_meta.random_seed }} everywhere. Regenerate all outputs above with:</p>
<pre>python main.py
python scripts/train_and_persist.py
pytest tests/ -q</pre>

</body></html>
""")


def _b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _df_to_html(df: pd.DataFrame) -> str:
    return df.round(3).to_html(index=False, border=0)


def generate_report(out_path: Path | None = None) -> Path:
    with open(PROCESSED_DIR / "tcga_preprocessing_metadata.json") as f:
        tcga_meta = json.load(f)
    with open(PROCESSED_DIR / "gdsc_preprocessing_metadata.json") as f:
        gdsc_meta = json.load(f)

    classification_results = pd.read_csv(RESULTS_DIR / "classification_results.csv")
    regression_results = pd.read_csv(RESULTS_DIR / "regression_results.csv")
    perm_clf = pd.read_csv(RESULTS_DIR / "feature_importance_classification_permutation.csv").head(10)
    perm_reg = pd.read_csv(RESULTS_DIR / "feature_importance_regression_permutation.csv").head(10)
    unified_genes = pd.read_csv(ROOT / "data" / "unified_gene_signature.csv")

    registry = []
    if (MODELS_DIR / "registry.json").exists():
        with open(MODELS_DIR / "registry.json") as f:
            registry = json.load(f)

    with open(ROOT / "configs" / "model_config.yaml", encoding="utf-8") as f:
        import yaml
        model_cfg = yaml.safe_load(f)
    deployed_clf = model_cfg["deployed_models"]["classification"]["primary_model"]
    deployed_dr = model_cfg["deployed_models"]["dose_response"]["ln_ic50"]["primary_model"]

    html = TEMPLATE.render(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        tcga_meta=tcga_meta,
        gdsc_meta=gdsc_meta,
        gene_sig_n=len(unified_genes),
        recurrent_genes=", ".join(unified_genes[unified_genes["cross_study_recurrent"]]["gene"].tolist()),
        classification_table=_df_to_html(
            classification_results[classification_results["experiment"] == "exp1_msi_raw_genes"]
            [["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
        ),
        regression_table=_df_to_html(
            regression_results[regression_results["experiment"] == "pooled_LN_IC50"]
            [["model", "MAE", "RMSE", "R2", "pearson_r", "spearman_r"]]
        ),
        perm_clf_table=_df_to_html(perm_clf),
        perm_reg_table=_df_to_html(perm_reg),
        figs={
            "roc_exp1": _b64(FIG_DIR / "roc_curve_exp1_msi_raw_genes.png"),
            "pred_vs_actual": _b64(FIG_DIR / "predicted_vs_actual_LN_IC50_RandomForest.png"),
        },
        n_registered_models=len(registry),
        deployed_clf=deployed_clf,
        deployed_dr=deployed_dr,
    )

    out_path = out_path or (RESULTS_DIR / "reports" /
                             f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = generate_report()
    print(f"Report written to {path}")
