"""Production routing and incomplete-experiment handling (no model is trained here)."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "results" / "ensemble_lab" / "lab_summary.json"
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "champions.json").exists(), reason="no production models")


def test_every_production_model_uses_exactly_its_registered_features():
    import model_registry as reg
    sizes = {"classification": 35, "classification_extended": None}
    for task in ("classification", "classification_extended"):
        e = reg.current_best(task)
        assert e is not None
        b = reg.load_model(e)
        assert list(b["features"]) == list(e["feature_columns"])
        fs = e["feature_pipeline"].split("|")[0]
        expected = {"core": 35, "core_plus_mmr": 39, "extended_molecular": 53}[fs]
        assert len(b["features"]) == expected
        if sizes[task]:
            assert len(b["features"]) == sizes[task]


def test_core_input_is_never_silently_routed_to_the_larger_model():
    import model_registry as reg
    import predict as prd
    core = reg.load_model(reg.current_best("classification"))
    ext_e = reg.current_best("classification_extended")
    ext = reg.load_model(ext_e)
    X = pd.DataFrame([[1.0] * len(core["features"])], columns=core["features"])
    assert len(prd.classify(core, X)) == 1
    with pytest.raises(ValueError, match="missing"):
        prd.classify(ext, X)


def test_lab_summary_records_incomplete_arms_without_metrics():
    if not LAB.exists():
        pytest.skip("lab summary not generated")
    lab = json.load(open(LAB))
    done = {r["feature_set"] for r in lab["table"]}
    for nc in lab.get("not_completed", []):
        assert nc["status"] == "not_completed"
        assert nc["feature_set"] not in done
        assert not any(k in nc for k in ("accuracy", "cv_accuracy", "roc_auc", "test_accuracy"))
    assert {"core", "core_plus_mmr", "extended_molecular"} <= done


def test_final_report_has_no_historical_whole_transcriptome_claim():
    rep = ROOT / "results" / "final_model_report.md"
    if not rep.exists():
        pytest.skip("report not generated")
    text = rep.read_text(encoding="utf-8")
    assert "Not completed" in text or "not completed" in text
    import re
    for line in text.splitlines():
        if re.search(r"whole.transcriptome", line, re.I):
            assert not re.search(r"0\.9\d|9\d(\.\d)?%", line), line


# ----------------------------------------------------------------------------- Train / Compare page
def _train_page(monkeypatch, key=None):
    from streamlit.testing.v1 import AppTest
    if key is None:
        monkeypatch.delenv("FERROPTOSIS_ADMIN_KEY", raising=False)
    else:
        monkeypatch.setenv("FERROPTOSIS_ADMIN_KEY", key)
    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=90)
    at.run()
    at.switch_page("pages/train_compare.py").run()
    return at


def test_training_is_locked_by_default_and_page_shows_real_status(monkeypatch):
    at = _train_page(monkeypatch)
    assert not at.exception, [str(e) for e in at.exception]
    text = " ".join(x.value for x in list(at.warning) + list(at.markdown) + list(at.info))
    assert "restricted to authorised researchers" in text
    assert "run_benchmark.py" in text
    assert "Whole-transcriptome experiment: Not completed" in text
    assert not [b for b in at.button if b.label == "Start"]          # no training controls
    assert not [t for t in at.text_input if "access key" in t.label.lower()]
    assert at.dataframe, "status table missing"


def test_wrong_key_keeps_training_locked(monkeypatch):
    at = _train_page(monkeypatch, key="correct-horse")
    box = [t for t in at.text_input if "access key" in t.label.lower()][0]
    box.set_value("wrong").run()
    assert not at.exception
    assert not [b for b in at.button if b.label == "Start"]


def test_right_key_unlocks_controls_but_promotion_is_off_by_default(monkeypatch):
    at = _train_page(monkeypatch, key="correct-horse")
    [t for t in at.text_input if "access key" in t.label.lower()][0].set_value("correct-horse").run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [b for b in at.button if b.label == "Start"]
    cb = [c for c in at.checkbox if "promotion" in c.label.lower()][0]
    assert cb.value is False


def test_status_rows_come_from_registry_and_do_not_invent_whole_transcriptome():
    import model_status as ms
    import terminology as T
    rows = {r["Model"]: r for r in ms.classification_rows()}
    wt = rows[T.WT]
    assert wt["Status"] == "Not completed" and wt["Test accuracy"] == "-"
    core = rows[T.WITH_GENES["core"]]
    assert core["Serving"] == "yes" and core["Artifact"] == "present"
    ext53 = rows[T.WITH_GENES["extended_molecular"]]
    assert ext53["Serving"] == "no" and "not selected" in ext53["Production eligibility"]


def test_promotion_gate_rejects_feature_mismatch_and_missing_metrics():
    import benchmark as bm
    import benchmark_run as br
    cfg = bm.load_cfg()
    inc = {"feature_columns": ["A", "B"], "cv_metrics": {"roc_auc_mean": 0.8, "roc_auc_std": 0.03}, "test_metrics": {"roc_auc": 0.8}}
    good = {"cv_metrics": {"roc_auc_mean": 0.95, "roc_auc_std": 0.01}, "test_metrics": {"roc_auc": 0.95},
            "leakage_checks": {"passed": True}, "features": ["A", "B"]}
    assert br.evaluate_promotion(good, inc, cfg)[0] is True
    ok, why = br.evaluate_promotion({**good, "features": ["A", "B", "C"]}, inc, cfg)
    assert not ok and "feature set differs" in why[0]
    ok, why = br.evaluate_promotion({**good, "cv_metrics": {}}, inc, cfg)
    assert not ok and "missing" in why[0]


# ----------------------------------------------------------------------------- drug-response promotion gate
def _reg_candidate(tmp_path, with_artifact=True, **over):
    import model_registry as reg
    inc = reg.current_best("dose_response")
    art = tmp_path / "m.joblib"
    if with_artifact:
        art.write_bytes(b"x")
    base = inc["test_metrics"]["baseline_drug_mean"]["mae"]
    cand = {"name": "x", "task": "dose_response", "version": "v0", "dataset": inc["dataset"], "trained_at": "2026-01-01T00:00:00+00:00",
            "path": "m.joblib", "feature_columns": list(inc["feature_columns"]), "feature_pipeline": inc["feature_pipeline"],
            "extra": dict(inc.get("extra") or {}),
            "cv_metrics": {"mae_mean": inc["cv_metrics"]["mae_mean"] - 0.5, "mae_std": inc["cv_metrics"]["mae_std"]},
            "test_metrics": {"mae": base - 0.3, "baseline_drug_mean": {"mae": base}}}
    cand.update(over)
    return cand, inc


def test_regression_promotion_fails_when_the_artifact_is_missing(tmp_path):
    import benchmark as bm
    import regression_benchmark as rb
    cfg = bm.load_cfg()
    cand, inc = _reg_candidate(tmp_path, with_artifact=False)
    ok, why = rb.evaluate_regression_promotion(cand, inc, cfg, explicit_action=True, root=tmp_path)
    assert not ok and any("artifact does not exist" in w for w in why)
    cand, inc = _reg_candidate(tmp_path, with_artifact=True)       # control: the same candidate passes once the file exists
    ok, why = rb.evaluate_regression_promotion(cand, inc, cfg, explicit_action=True, root=tmp_path)
    assert ok, why


@pytest.mark.parametrize("change,needle", [
    ({"feature_columns": ["A"]}, "feature set differs"),
    ({"feature_pipeline": "other"}, "preprocessing"),
    ({"extra": {"target_column": "AUC"}}, "response target differs"),
    ({"cv_metrics": {}}, "metrics missing"),
    ({"dataset": ""}, "metadata missing"),
])
def test_regression_promotion_rejects_incompatible_or_incomplete_candidates(tmp_path, change, needle):
    import benchmark as bm
    import regression_benchmark as rb
    cand, inc = _reg_candidate(tmp_path, **change)
    ok, why = rb.evaluate_regression_promotion(cand, inc, bm.load_cfg(), explicit_action=True, root=tmp_path)
    assert not ok and any(needle in w for w in why), why


def test_regression_promotion_needs_explicit_action_and_a_baseline_win(tmp_path):
    import benchmark as bm
    import regression_benchmark as rb
    cfg = bm.load_cfg()
    cand, inc = _reg_candidate(tmp_path)
    assert not rb.evaluate_regression_promotion(cand, inc, cfg, explicit_action=False, root=tmp_path)[0]
    base = cand["test_metrics"]["baseline_drug_mean"]["mae"]
    cand["test_metrics"]["mae"] = base + 0.01          # worse than the per-drug mean, even though CV looks better
    ok, why = rb.evaluate_regression_promotion(cand, inc, cfg, explicit_action=True, root=tmp_path)
    assert not ok and any("baseline" in w for w in why)


def test_promote_champion_refuses_a_missing_artifact_and_leaves_production_unchanged():
    import model_registry as reg
    before = open(reg.CHAMPIONS_PATH, "rb").read()
    ghost = {**reg.current_best("dose_response"), "path": "models/dose_response/does_not_exist.joblib", "version": "vGhost"}
    with pytest.raises(FileNotFoundError):
        reg.promote_champion("dose_response", ghost, "test")
    with pytest.raises(ValueError):
        reg.promote_champion("classification", reg.current_best("dose_response"), "wrong slot")
    assert open(reg.CHAMPIONS_PATH, "rb").read() == before


# ----------------------------------------------------------------------------- authoritative final registry
FINAL = ROOT / "results" / "final_model_registry.json"


@pytest.fixture(scope="module")
def final_registry():
    if not FINAL.exists():
        pytest.skip("run scripts/build_final_registry.py")
    return json.load(open(FINAL, encoding="utf-8"))


def test_exactly_one_authoritative_result_per_model_and_nothing_else_claims_it(final_registry):
    ids = [m["id"] for m in final_registry["models"]] + [final_registry["drug_response"]["id"]]
    assert len(ids) == len(set(ids))
    assert all(m["result_status"] == "AUTHORITATIVE_FINAL_RESULT" for m in final_registry["models"])
    assert not [o for o in final_registry["other_results"] if o["result_status"] == "AUTHORITATIVE_FINAL_RESULT"]
    assert final_registry["whole_transcriptome"]["result_status"] == "not_completed"


def test_final_registry_matches_stored_summaries_and_production(final_registry):
    import model_registry as reg
    champs = reg.champion_history()
    for m in final_registry["models"]:
        s = json.load(open(ROOT / "results" / "ensemble_lab" / m["id"] / "summary.json"))
        assert m["locked_test"]["accuracy"] == pytest.approx(s["final"]["test"]["accuracy"], abs=1e-12)
        assert m["cv"]["accuracy_mean"] == pytest.approx(s["final"]["cv"]["accuracy_mean"], abs=1e-12)
        assert (ROOT / m["artifact"]).exists()
        task = m["production"]["slot"]
        assert m["production"]["serving"] == (champs[task]["version"] == m["version"])
    assert final_registry["production_decision"]["changed_automatically"] is False
    assert not final_registry["models"][2]["production"]["serving"]        # the 53-feature model is a candidate, not serving


def test_only_the_core_is_called_ferroptosis_only_and_whole_transcriptome_has_no_numbers(final_registry):
    flags = {m["id"]: m["is_ferroptosis_only_classifier"] for m in final_registry["models"]}
    assert flags == {"core": True, "core_plus_mmr": False, "extended_molecular": False}
    wt = final_registry["whole_transcriptome"]
    assert wt["status"] == "not_completed" and wt["metrics"] is None
    assert "0.99" not in json.dumps(wt) and "94" not in json.dumps(wt)


def test_drug_response_is_described_honestly(final_registry):
    d = final_registry["drug_response"]
    assert "NOT a multi-model ensemble" in d["model_type"]
    assert d["response_target"].startswith("LN_IC50")
    assert d["locked_test"]["improvement_over_baseline"] < 0.01
    assert d["ensemble_check"]["status"] == "not_completed"
    assert "weak" in d["honest_summary"].lower() and d["strength"] == "preliminary / weak"


def test_every_quoted_number_is_classified(final_registry):
    for n in final_registry["number_reconciliation"]["numbers"]:
        assert n["status"] and n["what"] and n["source"]
    screening = [n for n in final_registry["number_reconciliation"]["numbers"] if n["value"] == 0.935]
    assert {n["status"] for n in screening} == {"screening_not_authoritative", "threshold_trade_off_not_final"}


def test_terminology_is_consistent_across_surfaces():
    import terminology as T
    assert T.NAME == {"core": "Ferroptosis Core Classifier", "core_plus_mmr": "35-gene + MMR Extended Model",
                      "extended_molecular": "Extended Molecular Model",
                      "whole_transcriptome": "Whole-transcriptome model (experimental)"}
    assert T.FERROPTOSIS_ONLY["core"] and not any(v for k, v in T.FERROPTOSIS_ONLY.items() if k != "core")
    report = (ROOT / "results" / "final_model_report.md")
    if report.exists():
        text = report.read_text(encoding="utf-8")
        for bad in ("Ferroptosis core (35 genes)", "35 genes + mismatch repair", "Extended molecular model"):
            assert bad not in text, bad


# ----------------------------------------------------------------------------- scientific freeze
def test_frozen_registry_is_complete_and_has_one_logged_scoring_per_model(final_registry):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_final_registry as bfr
    assert bfr.validate_registry(final_registry) == []
    for m in final_registry["models"]:
        p = m["provenance"]
        assert p["locked_test_scoring_events_for_this_model"] == 1 and p["stored_final_equals_logged_scoring"]
        assert p["max_abs_difference_vs_logged_scoring"] < 1e-9
        assert m["sample_count"]["locked_test"] == 139 and m["sample_count"]["locked_test_msi_h"] == 22


def test_audit_trail_documents_the_crash_without_claiming_an_unlogged_rescore(final_registry):
    cr = final_registry["locked_split"]["audit_trail"]["extended_model_crash"]
    assert cr["locked_test_accessed_before_crash"] is False and cr["unlogged_rescoring_found"] is False
    assert cr["used_to_tune"] is False and cr["stored_final_equals_logged_scoring"] is True
    assert final_registry["locked_split"]["audit_trail"]["caveats"]


def test_drug_response_cv_and_locked_test_are_kept_distinct(final_registry):
    d = final_registry["drug_response"]
    assert d["validation_splits"]["cross_validation"]["data"] != d["validation_splits"]["locked_test"]["data"]
    assert "DISJOINT" in d["validation_splits"]["locked_test"]["relation_to_cv"]
    assert d["cv"]["improvement_over_baseline"] == pytest.approx(0.0567, abs=1e-3)
    assert d["locked_test"]["improvement_over_baseline"] == pytest.approx(0.0022, abs=1e-3)
    assert d["strength"] == "preliminary / weak"


def test_paper_documents_are_generated_from_the_registry(final_registry):
    tbl, summ = ROOT / "results" / "paper_results_table.md", ROOT / "results" / "paper_scientific_summary.md"
    if not (tbl.exists() and summ.exists()):
        pytest.skip("run scripts/write_paper_artifacts.py")
    t, s = tbl.read_text(encoding="utf-8"), summ.read_text(encoding="utf-8")
    for m in final_registry["models"]:
        assert f"{m['locked_test']['accuracy']:.3f}" in t and f"{m['cv']['accuracy_mean']:.3f}" in t and m["display_name"] in t
        assert f"{m['locked_test']['roc_auc']:.3f}" in t and f"{m['locked_test']['recall']:.3f}" in t
    d = final_registry["drug_response"]
    assert f"{d['cv']['mae_mean']:.3f}" in t and f"{d['locked_test']['mae']:.3f}" in t
    assert "was not completed within the validated evaluation protocol and was therefore excluded from the final performance claims" in t
    assert "Not completed" in t and "Preliminary / weak" in t
    for n in range(1, 12):
        assert f"## {n}. " in s
    for text in (t, s):
        assert "0.99" not in text.replace("0.997", "").replace("0.991", "") and "94%" not in text


# ----------------------------------------------------------------------------- paper wording / table alignment audit
def _md_table(text, header_start):
    """Rows of the markdown table whose header row starts with `header_start` -> list of {header: cell} dicts."""
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(header_start))
    head = [c.strip() for c in lines[i].strip("|").split("|")]
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append(dict(zip(head, [c.strip() for c in l.strip("|").split("|")])))
    return rows


def test_paper_table_cells_are_under_the_right_model_and_column(final_registry):
    t = (ROOT / "results" / "paper_results_table.md").read_text(encoding="utf-8")
    byname = {m["display_name"]: m for m in final_registry["models"]}
    a = _md_table(t, "| Model | Feature count")
    assert [r["Model"] for r in a] == [m["display_name"] for m in final_registry["models"]]
    for r in a:
        m = byname[r["Model"]]
        assert r["Feature count"] == str(m["n_features"])
        assert r["CV accuracy ± SD"] == f"{m['cv']['accuracy_mean']:.3f} ± {m['cv']['accuracy_std']:.3f}"
        assert r["Locked-test accuracy"] == f"{m['locked_test']['accuracy']:.3f}"
        assert r["Locked-test AUC"] == f"{m['locked_test']['roc_auc']:.3f}"
        assert r["F1 (test)"] == f"{m['locked_test']['f1']:.3f}"
        assert r["MSI-H recall (test)"] == f"{m['locked_test']['recall']:.3f}"
        assert (("In production" in r["Production status"]) == m["production"]["serving"])
    a2 = _md_table(t, "| Model | Locked-test accuracy, 95% interval")
    for r in a2:
        m = byname[r["Model"]]
        lo, hi = m["locked_test"]["accuracy_ci95"]
        assert r["Locked-test accuracy, 95% interval"] == f"{lo:.2f}–{hi:.2f}"
        for col, key in (("Balanced accuracy", "balanced_accuracy"), ("Specificity", "specificity"), ("MCC", "mcc"), ("Precision", "precision")):
            assert r[col] == f"{m['locked_test'][key]:.3f}"
        assert r["Test n / MSI-H"] == f"{m['locked_test']['n']:.0f} / {m['locked_test']['n_positive']:.0f}"
    a3 = _md_table(t, "| Model | CV AUC ± SD")
    for r in a3:
        o = byname[r["Model"]]["observations"]["cv_vs_locked_test_discrimination"]
        assert r["Locked-test AUC"] == f"{o['locked_test_roc_auc']:.3f}" and r["Gap (CV − test)"] == f"{o['gap_cv_minus_test']:.3f}"
    # the 35-gene and 53-feature rows must not be swapped or shifted
    core, ext = {r["Model"]: r for r in a}["Ferroptosis Core Classifier"], {r["Model"]: r for r in a}["Extended Molecular Model"]
    assert (core["Feature count"], ext["Feature count"]) == ("35", "53")
    assert core["Locked-test accuracy"] == "0.863" and ext["Locked-test accuracy"] == "0.921"


def test_wording_claims_are_no_stronger_than_the_evidence(final_registry):
    t = (ROOT / "results" / "paper_results_table.md").read_text(encoding="utf-8")
    s = (ROOT / "results" / "paper_scientific_summary.md").read_text(encoding="utf-8")
    rep = (ROOT / "results" / "final_model_report.md").read_text(encoding="utf-8")
    ex = final_registry["external_cohort"]
    assert ex["wording"] in t and ex["wording"] in s
    acc = ex["sample_accounting"]
    assert (acc["raw_samples"], acc["evaluated"], acc["msi_h"], acc["mss"]) == (110, 105, 24, 81)      # re-derived from the stored files
    assert ex["id_overlap_with_tcga"] == {"sample_ids": 0, "patient_ids": 0}
    for text in (t, s, rep):
        for banned in ("fixed before any modelling", "not selected on this dataset", "95% CI", "confidence interval"):
            assert banned not in text.replace("Not claimed: that the list was fixed before any modelling", ""), banned
        assert "bootstrap" in text
    assert "not independent of the TCGA cohort" in s and "stratified" in final_registry["interval_methods"]["classification_locked_test_intervals"]["method"]
    assert final_registry["feature_set_provenance"]["not_claimed"].startswith("that the list was fixed before any modelling")
