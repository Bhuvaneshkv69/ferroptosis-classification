"""Read-only model status for the Train / Compare page. Everything comes from models/registry.json,
models/champions.json and results/ensemble_lab/lab_summary.json; nothing is typed in and nothing is trained."""
from __future__ import annotations

import json
from pathlib import Path

import model_registry as reg

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "results" / "ensemble_lab"
BENCH = ROOT / "results" / "benchmark"

import terminology as T  # noqa: E402

ROLES = [  # (feature set key, role label, registry task)
    ("core", T.WITH_GENES["core"], "classification"),
    ("core_plus_mmr", T.WITH_GENES["core_plus_mmr"], "classification_extended"),
    ("extended_molecular", T.WITH_GENES["extended_molecular"], "classification_extended"),
    ("whole_transcriptome", T.WT, "classification_wt"),
]


def _lab() -> dict:
    p = LAB / "lab_summary.json"
    return json.load(open(p)) if p.exists() else {}


def _latest(task: str, feature_set: str | None):
    hits = [e for e in reg.list_models(task)
            if feature_set is None or str(e.get("feature_pipeline", "")).split("|")[0] == feature_set]
    return hits[-1] if hits else None


def _row(role: str, entry: dict | None, serving: bool, why: str) -> dict:
    if entry is None:
        return {"Model": role, "Status": "Not completed", "Serving": "no", "Version": "-", "Trained": "-",
                "CV accuracy": "-", "Test accuracy": "-", "Test AUC": "-", "Artifact": "-", "Production eligibility": why}
    cv, te = entry.get("cv_metrics") or {}, entry.get("test_metrics") or {}
    art = (ROOT / entry["path"]).exists()
    return {"Model": role, "Status": "In production" if serving else "Completed, not serving",
            "Serving": "yes" if serving else "no", "Version": entry["version"], "Trained": entry["trained_at"][:16].replace("T", " "),
            "CV accuracy": f"{cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f}" if "accuracy_mean" in cv else "-",
            "Test accuracy": f"{te['accuracy']:.3f}" if "accuracy" in te else "-",
            "Test AUC": f"{te['roc_auc']:.3f}" if "roc_auc" in te else "-",
            "Artifact": "present" if art else "MISSING", "Production eligibility": why if art else "artifact missing"}


def classification_rows() -> list[dict]:
    lab = _lab()
    routing = lab.get("routing", {})
    prod = {t: reg.current_best(t) for t in ("classification", "classification_extended")}
    rows = []
    for fs, role, task in ROLES:
        e = _latest(task, fs)
        p = prod.get(task)
        serving = bool(e and p and e["version"] == p["version"] and e["name"] == p["name"])
        if e is None:
            why = "no verified artifact: experiment incomplete; no result is reported"
        elif serving:
            why = "serving; passed its promotion gate"
        elif fs == "extended_molecular":
            why = "benchmarked; not selected: " + str(routing.get("extended", {}).get("reason", "see Model Lab"))
        elif fs == "whole_transcriptome":
            why = "research benchmark only; never served"
        else:
            why = "registered; not the production choice for this slot"
        rows.append(_row(role, e, serving, why))
    return rows


def drug_row() -> dict:
    e = reg.current_best("dose_response")
    if e is None:
        return {"Model": T.DRUG + " (ln IC50)", "Status": "Not completed"}
    te, cv = e["test_metrics"], e["cv_metrics"]
    return {"Model": T.DRUG + " (ln IC50)", "Status": "In production", "Algorithm": T.drug_display(e) + ", not a multi-model ensemble",
            "Version": e["version"], "Trained": e["trained_at"][:16].replace("T", " "),
            "CV MAE": f"{cv['mae_mean']:.3f} ± {cv['mae_std']:.3f}", "Test MAE": f"{te['mae']:.3f}",
            "Held-out cell lines": te.get("n_cell_lines"),
            "Gain vs per-drug-mean baseline": f"{te['mae_gain_vs_drug_mean']:.1%}" if "mae_gain_vs_drug_mean" in te else "-",
            "Artifact": "present" if (ROOT / e["path"]).exists() else "MISSING",
            "Ensemble check": "completed" if (BENCH / "regression_ensemble_check.json").exists() else "Not completed"}


def artifact_locations() -> list[tuple[str, str, bool]]:
    items = [("Final model report", "results/final_model_report.md"), ("Ensemble lab summary", "results/ensemble_lab/lab_summary.json"),
             ("35-gene baseline", "results/baseline_35_gene/baseline_report.md"), ("Single-model benchmark", "results/benchmark/benchmark_summary.json"),
             ("Drug-response benchmark", "results/benchmark/regression_summary.json"), ("Model registry", "models/registry.json"),
             ("Production record", "models/champions.json"), ("Model selection write-up", "docs/MODEL_SELECTION.md")]
    return [(n, p, (ROOT / p).exists()) for n, p in items]
