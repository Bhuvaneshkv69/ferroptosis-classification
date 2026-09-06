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
