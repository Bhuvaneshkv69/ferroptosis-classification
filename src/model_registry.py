"""
Minimal file-based model registry.

Persists fitted sklearn Pipelines (which already contain their own train-only
-fit preprocessing steps -- StandardScaler / OneHotEncoder / ColumnTransformer
-- so no separate "preprocessor artifact" needs to be tracked) to models/, and
records one JSON entry per persisted model in models/registry.json with:

    name, version, task, experiment, dataset, feature_columns,
    hyperparameters, metrics, random_seed, trained_at, path

An existing (name, version) pair is NEVER overwritten: register_model()
raises if the target file already exists, and a fresh version string
(timestamp-based, second-resolution) is used by scripts/train_and_persist.py
on every run, so re-running training always creates a new version and the
registry keeps full history rather than silently replacing prior artifacts.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


def _load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_registry(entries: list[dict]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, default=str)


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")


def register_model(
    *,
    name: str,
    task: str,
    experiment: str,
    model_object,
    feature_columns: list[str],
    metrics: dict,
    hyperparameters: dict,
    dataset: str,
    random_seed: int,
    version: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Persists `model_object` (a fitted sklearn Pipeline/estimator) via
    joblib and appends a metadata entry to the registry. Returns the entry."""
    version = version or new_version()
    subdir = MODELS_DIR / task / experiment
    subdir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}__{version}.joblib"
    path = subdir / filename

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model artifact: {path}")

    joblib.dump(model_object, path)

    entry = {
        "name": name,
        "version": version,
        "task": task,
        "experiment": experiment,
        "dataset": dataset,
        "feature_columns": list(feature_columns),
        "hyperparameters": hyperparameters,
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
        "random_seed": random_seed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
    }
    if extra:
        entry["extra"] = extra

    entries = _load_registry()
    entries.append(entry)
    _save_registry(entries)
    return entry


def list_models(task: str | None = None, experiment: str | None = None, name: str | None = None) -> list[dict]:
    entries = _load_registry()
    if task:
        entries = [e for e in entries if e["task"] == task]
    if experiment:
        entries = [e for e in entries if e["experiment"] == experiment]
    if name:
        entries = [e for e in entries if e["name"] == name]
    return entries


def latest_model(task: str, experiment: str, name: str) -> dict | None:
    matches = list_models(task=task, experiment=experiment, name=name)
    if not matches:
        return None
    return max(matches, key=lambda e: e["trained_at"])


def load_model(entry: dict):
    """Loads the joblib artifact referenced by a registry entry (as returned
    by latest_model()/list_models())."""
    path = ROOT / entry["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Registry references {path}, but the file does not exist. "
            f"Run `python scripts/train_and_persist.py` to (re)create it."
        )
    return joblib.load(path)


if __name__ == "__main__":
    entries = _load_registry()
    if not entries:
        print("Registry is empty. Run scripts/train_and_persist.py to populate it.")
    for e in entries:
        print(f"{e['task']}/{e['experiment']}/{e['name']} {e['version']} -> {e['path']}")


# ---------------------------------------------------------------------------
# Champion tracking + rich benchmark entries (added with the model-selection engine)
#
# registry.json stays append-only. Which model is "current best" for a task is
# recorded separately in models/champions.json, written ONLY by
# promote_champion() -- so the application discovers the production model from
# data, and a retrained model never replaces it unless the promotion gate in
# src/benchmark.py passes.
# ---------------------------------------------------------------------------
CHAMPIONS_PATH = MODELS_DIR / "champions.json"


def _load_champions() -> dict:
    if not CHAMPIONS_PATH.exists():
        return {}
    with open(CHAMPIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def register_benchmark_model(*, task: str, name: str, display_name: str, model_object, feature_columns: list[str],
                             feature_pipeline: str, dataset: str, hyperparameters: dict, cv_metrics: dict,
                             test_metrics: dict, external_metrics: dict | None, calibration: str,
                             extra: dict | None = None, version: str | None = None, seed: int = 42) -> dict:
    """Persists a fitted model bundle and appends a rich registry entry
    (name, version, feature pipeline, dataset, training date, CV / locked-test / external metrics,
    hyper-parameters, features, calibration status, model file)."""
    version = version or new_version()
    experiment = "msi_benchmark" if task == "classification" else f"{task}_benchmark"
    subdir = MODELS_DIR / task / experiment
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / f"{name}__{version}.joblib"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model artifact: {path}")
    joblib.dump(model_object, path)
    entry = {
        "name": name, "display_name": display_name, "version": version, "task": task, "experiment": experiment,
        "dataset": dataset, "feature_pipeline": feature_pipeline, "feature_columns": list(feature_columns),
        "hyperparameters": hyperparameters, "calibration": calibration,
        "cv_metrics": cv_metrics, "test_metrics": test_metrics, "external_metrics": external_metrics or {},
        "metrics": test_metrics,  # legacy key read by older code paths: always the LOCKED-TEST metrics
        "random_seed": seed, "trained_at": datetime.now(timezone.utc).isoformat(),
        "path": str(path.relative_to(ROOT)).replace("\\", "/"), "schema": 2,
    }
    if extra:
        entry["extra"] = extra
    entries = _load_registry()
    entries.append(entry)
    _save_registry(entries)
    return entry


def promote_champion(task: str, entry: dict, reason: str) -> None:
    if entry.get("task") != task:
        raise ValueError(f"cannot promote a '{entry.get('task')}' model into the '{task}' slot")
    if not (ROOT / entry["path"]).exists():       # last line of defence, independent of any caller's gate
        raise FileNotFoundError(f"refusing to promote: model artifact missing on disk: {entry['path']}")
    ch = _load_champions()
    previous = ch.get(task)
    ch[task] = {"name": entry["name"], "version": entry["version"], "experiment": entry["experiment"],
                "promoted_at": datetime.now(timezone.utc).isoformat(), "reason": reason,
                "previous": ({"name": previous["name"], "version": previous["version"]} if previous else None)}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHAMPIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(ch, f, indent=2)


def current_best(task: str) -> dict | None:
    """The registry entry currently serving `task`, looked up from champions.json (never hard-coded)."""
    rec = _load_champions().get(task)
    if not rec:
        return None
    for e in reversed(_load_registry()):
        if e["task"] == task and e["name"] == rec["name"] and e["version"] == rec["version"]:
            return e
    return None


def champion_history() -> dict:
    return _load_champions()
