from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _prereqs_present():
    return (ROOT / "data" / "processed" / "gdsc_raw_dose_response_curves.csv").exists() and \
           (ROOT / "results" / "real_dose_response_curve_fits.csv").exists()


pytestmark = pytest.mark.skipif(
    not _prereqs_present(),
    reason="Run `python src/process_raw_dose_response.py` first (requires the 2.1GB raw GDSC file)",
)


@pytest.fixture(scope="module")
def curve_points():
    return pd.read_csv(ROOT / "data" / "processed" / "gdsc_raw_dose_response_curves.csv")


@pytest.fixture(scope="module")
def curve_fits():
    return pd.read_csv(ROOT / "results" / "real_dose_response_curve_fits.csv")


def test_real_curve_points_cover_all_four_representative_drugs(curve_points):
    expected = {"5-Fluorouracil", "Oxaliplatin", "Irinotecan", "Sorafenib"}
    assert expected.issubset(set(curve_points["DRUG_NAME"]))


def test_real_viability_values_are_non_negative(curve_points):
    assert (curve_points["mean_viability"] >= 0).all()


def test_real_curve_points_have_at_least_one_replicate(curve_points):
    assert (curve_points["n_replicates"] >= 1).all()


def test_curve_fits_produced_positive_ic50_and_hill_values(curve_fits):
    assert (curve_fits["IC50_uM_fit"] > 0).all()
    assert (curve_fits["hill_fit"] > 0).all()


def test_real_curve_figures_exist_for_every_drug():
    fig_dir = ROOT / "figures"
    for drug in ["5-Fluorouracil", "Oxaliplatin", "Irinotecan", "Sorafenib"]:
        assert (fig_dir / f"dose_response_curve_real_{drug}.png").exists()
