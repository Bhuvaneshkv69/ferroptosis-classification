from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def registry_entries():
    import model_registry as reg
    path = ROOT / "models" / "registry.json"
    if not path.exists():
        pytest.skip("Run scripts/train_and_persist.py first")
    return reg.list_models()


def test_registry_has_all_expected_task_experiment_combinations(registry_entries):
    combos = {(e["task"], e["experiment"]) for e in registry_entries}
    expected = {
        ("classification", "exp1_msi_raw_genes"),
        ("classification", "exp2_msi_engineered"),
        ("classification", "exp3_riskgroup_raw_genes"),
        ("dose_response", "pooled_LN_IC50"),
        ("dose_response", "pooled_AUC"),
    }
    assert expected.issubset(combos)


def test_every_registry_entry_has_required_metadata_fields(registry_entries):
    required = {"name", "version", "task", "experiment", "dataset", "feature_columns",
                "hyperparameters", "metrics", "random_seed", "trained_at", "path"}
    for e in registry_entries:
        assert required.issubset(e.keys())


def test_every_registered_artifact_file_actually_exists_on_disk(registry_entries):
    for e in registry_entries:
        assert (ROOT / e["path"]).exists(), f"registry references missing file: {e['path']}"


def test_latest_model_returns_most_recently_trained_entry(registry_entries):
    import model_registry as reg
    entry = reg.latest_model(task="classification", experiment="exp1_msi_raw_genes", name="LogisticRegression")
    assert entry is not None
    same = [e for e in registry_entries if e["task"] == "classification"
            and e["experiment"] == "exp1_msi_raw_genes" and e["name"] == "LogisticRegression"]
    assert entry["trained_at"] == max(e["trained_at"] for e in same)


def test_load_model_returns_a_fitted_sklearn_pipeline(registry_entries):
    import model_registry as reg
    entry = reg.latest_model(task="classification", experiment="exp1_msi_raw_genes", name="LogisticRegression")
    model = reg.load_model(entry)
    assert hasattr(model, "predict") and hasattr(model, "predict_proba")


class _DummyModel:
    """Module-level (picklable) stand-in for a fitted estimator, used only to
    test the registry's file-handling logic without a real training run."""
    def fit(self, X, y):
        return self


def test_register_model_refuses_to_overwrite_an_existing_version(tmp_path, monkeypatch):
    import model_registry as reg
    monkeypatch.setattr(reg, "ROOT", tmp_path)
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "registry.json")

    entry = reg.register_model(
        name="Dummy", task="unit_test", experiment="exp0", model_object=_DummyModel(),
        feature_columns=["a", "b"], metrics={"acc": 1.0}, hyperparameters={}, dataset="synthetic",
        random_seed=42, version="v_fixed",
    )
    assert entry["version"] == "v_fixed"
    with pytest.raises(FileExistsError):
        reg.register_model(
            name="Dummy", task="unit_test", experiment="exp0", model_object=_DummyModel(),
            feature_columns=["a", "b"], metrics={"acc": 1.0}, hyperparameters={}, dataset="synthetic",
            random_seed=42, version="v_fixed",
        )
