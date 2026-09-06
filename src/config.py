"""
Loads configs/data_config.yaml and configs/model_config.yaml once and exposes
their values to the rest of src/*. Centralizes thresholds, paths and random
seeds that used to be duplicated as Python constants in individual modules.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


DATA_CONFIG = _load("data_config.yaml")
MODEL_CONFIG = _load("model_config.yaml")

RANDOM_SEED = DATA_CONFIG["random_seed"]
