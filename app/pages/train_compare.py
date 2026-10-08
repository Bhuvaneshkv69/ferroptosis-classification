import hmac
import os
import sys
import threading
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403
import model_status as ms  # noqa: E402
from theme import page_header, section_label  # noqa: E402

page_header("Train / Compare Models", "Model status for everyone; rebuilding models is restricted to authorised researchers")

ADMIN_ENV = "FERROPTOSIS_ADMIN_KEY"
key = os.environ.get(ADMIN_ENV)


def status_panel():
    """Read-only view of what exists, from the registry and stored results. Shown in both modes."""
    section_label("Production models")
    cls = [r for r in ms.classification_rows()]
    prod = [r for r in cls if r["Serving"] == "yes"]
    c1, c2 = st.columns(2)
    core = next((r for r in prod if r["Model"].startswith("Ferroptosis Core")), None)
    ext = next((r for r in prod if not r["Model"].startswith("Ferroptosis Core")), None)
    c1.markdown("**Classification, default (ferroptosis genes only)**  \n" + (
        f"{core['Model']} · version {core['Version']} · test accuracy {core['Test accuracy']}" if core else "none registered"))
    c2.markdown("**Classification, optional extended model**  \n" + (
        f"{ext['Model']} · version {ext['Version']} · test accuracy {ext['Test accuracy']}" if ext else "none registered"))
    d = ms.drug_row()
    st.markdown("**Drug response (computational in-vitro estimate, not a dose)**  \n"
                + (f"{d['Algorithm']} · version {d['Version']} · held-out MAE {d['Test MAE']} ln-units" if d["Status"] != "Not completed"
                   else "none registered"))

    section_label("Completed model experiments")
    st.dataframe(pd.DataFrame(cls), hide_index=True, width="stretch")
    st.caption("The 35-gene core is a ferroptosis classifier. The other classification models add mismatch-repair (and, for the "
               "Extended Molecular Model, immune) genes and must not be described as ferroptosis classifiers. "
               "Accuracy figures are model-vs-label agreement; the majority-class baseline is 84%.")
    if any(r["Status"] == "Not completed" for r in cls):
        st.info("**Whole-transcriptome experiment: Not completed.** No metrics are shown and no earlier figure is substituted.",
                icon=":material/hourglass_empty:")

    section_label("Drug-response model")
    st.dataframe(pd.DataFrame([d]).T.rename(columns={0: "Value"}).astype(str), width="stretch")
    if d.get("Ensemble check") == "Not completed":
        st.info("**Drug-response ensemble: Not completed.** Only the single-model regressor is available.",
                icon=":material/hourglass_empty:")

    section_label("Where the evidence is")
    for name, path, exists in ms.artifact_locations():
        st.markdown(f"* {name}: `{path}`" + ("" if exists else "  *(not generated)*"))
    hist = reg.champion_history()
    if hist:
        with st.expander("Production history (champions.json)"):
            st.json(hist, expanded=False)


if not key:
    st.warning("**Model training is restricted to authorised researchers.**", icon=":material/lock:")
    st.markdown(
        "Retraining can replace the model behind every analysis, and a full benchmark is a long, compute-heavy run. "
        "It is therefore disabled unless the app is launched with the `" + ADMIN_ENV + "` environment variable set. "
        "Researchers can also run `python scripts/run_benchmark.py` (classification) or "
        "`python scripts/run_benchmark.py --task regression` from the command line; nothing is started from this page.")
    status_panel()
    st.stop()

entered = st.text_input("Researcher access key", type="password")
if not entered or not hmac.compare_digest(entered.encode(), key.encode()):
    st.info("Enter the researcher access key to unlock training controls. Model status is shown below.",
            icon=":material/lock:")
    status_panel()
    st.stop()


@st.cache_resource
def job_state() -> dict:
    return {"running": False, "log": [], "frac": 0.0, "result": None, "error": None, "label": ""}


J = job_state()
STEPS = ["Load training data", "Validate", "Build pipelines", "Cross-validate", "Tune candidates", "Compare",
         "Select best", "Evaluate locked test", "Save model", "Register model", "Generate report"]

st.success("Researcher mode. The command-line equivalent is `python scripts/run_benchmark.py`.", icon=":material/lock_open:")
section_label("Setup")
a, b, c = st.columns(3)
task = a.selectbox("Task", ["Classification (MSI status)", "Drug response (LN IC50)"])
mode = b.selectbox("Run", ["Quick smoke test (no locked test, no registration)", "Full benchmark (about an hour)"])
promote = c.checkbox("Allow promotion if every gate passes", False,
                     help="Off by default: the run is registered but the production model is left unchanged. When on, a new "
                          "model still replaces production only if its metrics exist, its features match the served slot, "
                          "the artifact exists, CV improves, the locked-test metric is not worse, folds are stable and every "
                          "leakage check passes. A higher accuracy alone never promotes a model.")
if promote and not mode.startswith("Quick"):
    st.warning("A full run with promotion enabled can change the production model if the gate is satisfied.")
dataset_path, upload_ok = None, True
if task.startswith("Classification"):
    up = st.file_uploader("Optional: your own training table (CSV)", type=["csv"],
                          help="Needs sampleId, patientId, msi_label and the 35 gene columns (log2 scale).")
    if up is not None:
        df = pd.read_csv(up)
        v = val.validate_training_dataframe(df)
        for w in v["warnings"]:
            st.warning(w)
        for e in v["errors"]:
            st.error(e)
        upload_ok = v["is_valid"]
        if upload_ok:
            dataset_path = ROOT / "data" / "processed" / "uploaded_training_table.csv"
            df.to_csv(dataset_path, index=False)
            st.success(f"{len(df)} rows validated.")
st.caption("Steps: " + " → ".join(STEPS))


def _worker(task_kind: str, quick: bool, promote_: bool, path):
    def progress(msg, frac=None):
        J["log"].append(msg)
        if frac is not None:
            J["frac"] = frac
    try:
        sys.path.insert(0, str(ROOT / "src"))
        if task_kind == "classification":
            import benchmark_run
            J["result"] = benchmark_run.run_classification_benchmark(quick=quick, progress=progress, promote=promote_,
                                                                      dataset_path=path)
        else:
            import regression_benchmark
            J["result"] = regression_benchmark.run_regression_benchmark(quick=quick, progress=progress,
                                                                        promote=promote_)
    except Exception as e:  # noqa: BLE001
        J["error"] = repr(e)
    finally:
        J["running"] = False


if st.button("Start", type="primary", disabled=J["running"] or not upload_ok):
    J.update(running=True, log=[], frac=0.0, result=None, error=None)
    threading.Thread(target=_worker, daemon=True, args=(
        "classification" if task.startswith("Class") else "regression", mode.startswith("Quick"), promote,
        dataset_path)).start()
    st.rerun()


@st.fragment(run_every=3)
def monitor():
    if J["running"]:
        st.progress(min(J["frac"], 0.99), text=(J["log"][-1] if J["log"] else "starting"))
        with st.expander("Log", expanded=False):
            st.code("\n".join(J["log"][-40:]))
    elif J["error"]:
        st.error(J["error"])
    elif J["result"] is not None:
        r = J["result"]
        st.success("Run finished.")
        if r.get("promotion"):
            p = r["promotion"]
            (st.success if p["promoted"] else st.warning)(
                ("Promoted to production. " if p["promoted"] else "Not promoted; production model unchanged. ")
                + "; ".join(p["reasons"]))
        if r.get("verdict"):
            st.write(f"**{r['verdict']['statement']}** · locked-test accuracy "
                     f"{pct(r['verdict']['locked_test_accuracy'], 1)}")
        st.write(f"Champion: **{mz.PLAIN_ENGLISH.get(r['champion'], r['champion'])}**")
        with st.expander("Full summary"):
            st.json(r, expanded=False)


section_label("Progress")
monitor()
section_label("Status")
status_panel()
