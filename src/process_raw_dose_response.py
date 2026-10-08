"""
Builds REAL (measured, not reconstructed) dose-response curves for the four
representative drugs from the raw per-well GDSC2 viability file, replacing
the illustrative IC50-reconstructed curves in dose_response.py.

Source: GDSC2_public_raw_data_24Jul22.csv (~2.1GB, Sanger FTP mirror,
downloaded once to data/raw/gdsc/ -- see docs/DATASET.md). This file is
intentionally NOT loaded fully into memory: it is streamed in chunks twice
(pass 1 to find the dosed wells for our drugs/cell-lines, pass 2 to find the
matching negative-control wells on the same plate), keeping peak memory to
the size of the small filtered subset, not the full 2.1GB/~28M-row file.

Methodology (a documented, simplified re-implementation, NOT a byte-for-byte
reproduction of GDSC's own `gdscIC50` R package, which additionally applies
positional/edge-effect correction -- see docs/LIMITATIONS.md):
  1. Keep raw-file rows for our 44 colorectal cell lines x 4 representative
     drugs (identified by DRUG_ID) -- these are the "dosed wells".
  2. For every BARCODE (physical assay plate) that contains a dosed well,
     also keep its TAG=="NC-0" (negative control, i.e. no drug) wells.
  3. Relative viability = INTENSITY / mean(NC-0 INTENSITY on the same
     BARCODE) -- normalizes for plate-to-plate signal-intensity differences.
  4. Average technical replicate wells at (DRUG_ID, COSMIC_ID, CONC).
  5. Fit a 2-parameter logistic viability curve (matching the model already
     used for the illustrative reconstruction, so the two are visually and
     mathematically comparable) via scipy.optimize.curve_fit on the REAL
     averaged points, per (drug, cell line).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "gdsc" / "GDSC2_public_raw_data_24Jul22.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

DRUG_ID_TO_NAME = {1073: "5-Fluorouracil", 1085: "Sorafenib", 1088: "Irinotecan", 1089: "Oxaliplatin"}
CHUNK_SIZE = 2_000_000
USECOLS = ["BARCODE", "COSMIC_ID", "CELL_LINE_NAME", "TAG", "DRUG_ID", "CONC", "INTENSITY"]


def _colorectal_cosmic_ids():
    df = pd.read_csv(PROCESSED_DIR / "gdsc_colorectal_dose_response_processed.csv")
    return set(df["COSMIC_ID"].unique().tolist())


def _stream_filtered(predicate, usecols=USECOLS, chunksize=CHUNK_SIZE):
    kept = []
    for i, chunk in enumerate(pd.read_csv(RAW_PATH, usecols=usecols, chunksize=chunksize, low_memory=False)):
        mask = predicate(chunk)
        if mask.any():
            kept.append(chunk[mask])
        if (i + 1) % 5 == 0:
            print(f"    ... scanned {(i + 1) * chunksize / 1e6:.0f}M rows")
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=usecols)


def build_real_curve_dataset(log_path=None):
    log = []

    def note(msg):
        log.append(msg)
        print(msg)

    cosmic_ids = _colorectal_cosmic_ids()
    drug_ids = set(DRUG_ID_TO_NAME.keys())
    note(f"Streaming raw file ({RAW_PATH.stat().st_size / 1e9:.2f} GB): pass 1 (dosed wells) ...")

    dosed = _stream_filtered(
        lambda c: c["COSMIC_ID"].isin(cosmic_ids) & c["DRUG_ID"].isin(drug_ids)
    )
    note(f"Pass 1: {len(dosed)} dosed-well rows found for {dosed['COSMIC_ID'].nunique()} cell lines "
         f"x {dosed['DRUG_ID'].nunique()} drugs")

    barcodes = set(dosed["BARCODE"].unique().tolist())
    note(f"Streaming raw file: pass 2 (negative-control wells on the {len(barcodes)} relevant plates) ...")
    controls = _stream_filtered(
        lambda c: c["BARCODE"].isin(barcodes) & (c["TAG"] == "NC-0")
    )
    note(f"Pass 2: {len(controls)} negative-control-well rows found")

    control_mean = controls.groupby("BARCODE")["INTENSITY"].mean().rename("control_intensity")
    dosed = dosed.merge(control_mean, on="BARCODE", how="left")
    n_no_control = dosed["control_intensity"].isna().sum()
    note(f"Dosed wells with no matching NC-0 control on their plate (dropped): {n_no_control}")
    dosed = dosed.dropna(subset=["control_intensity"])

    dosed["viability"] = dosed["INTENSITY"] / dosed["control_intensity"]
    dosed["DRUG_NAME"] = dosed["DRUG_ID"].map(DRUG_ID_TO_NAME)

    agg = (
        dosed.groupby(["DRUG_NAME", "COSMIC_ID", "CELL_LINE_NAME", "CONC"])
        .agg(mean_viability=("viability", "mean"), std_viability=("viability", "std"),
             n_replicates=("viability", "size"))
        .reset_index()
    )
    note(f"Averaged to {len(agg)} (drug, cell line, concentration) points across "
         f"{agg['CELL_LINE_NAME'].nunique()} cell lines")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "gdsc_raw_dose_response_curves.csv"
    agg.to_csv(out_path, index=False)
    note(f"Saved real (measured) dose-response points -> {out_path}")

    meta = {
        "source_file": str(RAW_PATH.name),
        "source_file_size_bytes": RAW_PATH.stat().st_size,
        "drugs": DRUG_ID_TO_NAME,
        "n_cell_lines": int(agg["CELL_LINE_NAME"].nunique()),
        "n_points": len(agg),
        "normalization": "relative viability = well INTENSITY / mean(NC-0 INTENSITY on same BARCODE)",
    }
    with open(PROCESSED_DIR / "gdsc_raw_dose_response_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if log_path:
        Path(log_path).write_text("\n".join(log), encoding="utf-8")

    return agg


def _logistic(conc, ic50, hill):
    return 1.0 / (1.0 + (conc / ic50) ** hill)


def fit_and_plot_real_curves(agg: pd.DataFrame):
    fit_rows = []
    for drug in DRUG_ID_TO_NAME.values():
        sub = agg[agg["DRUG_NAME"] == drug]
        # plot up to 4 representative cell lines (highest point-count) for legibility
        counts = sub.groupby("CELL_LINE_NAME").size().sort_values(ascending=False)
        picks = counts.head(4).index.tolist()

        fig, ax = plt.subplots(figsize=(7, 5))
        for cl in picks:
            s = sub[sub["CELL_LINE_NAME"] == cl].sort_values("CONC")
            if len(s) < 4:
                continue
            try:
                popt, _ = curve_fit(_logistic, s["CONC"], s["mean_viability"],
                                     p0=[s["CONC"].median(), 1.0], maxfev=5000,
                                     bounds=([1e-6, 0.05], [1e6, 20]))
                ic50_fit, hill_fit = popt
                fit_rows.append({"DRUG_NAME": drug, "CELL_LINE_NAME": cl, "IC50_uM_fit": ic50_fit,
                                  "hill_fit": hill_fit, "n_points": len(s)})
                conc_grid = np.logspace(np.log10(s["CONC"].min()), np.log10(s["CONC"].max()), 200)
                ax.plot(conc_grid, _logistic(conc_grid, *popt), "-", alpha=0.8)
            except RuntimeError:
                pass
            ax.errorbar(s["CONC"], s["mean_viability"], yerr=s["std_viability"], fmt="o", markersize=4,
                        label=cl, capsize=2)

        ax.set_xscale("log")
        ax.set_xlabel("Concentration (uM, log scale)")
        ax.set_ylabel("Relative viability (measured, normalized to NC-0 control)")
        ax.set_title(f"REAL measured dose-response curves - {drug}\n(raw per-well viability, "
                      f"{RAW_PATH.name})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"dose_response_curve_real_{drug}.png", dpi=150)
        plt.close(fig)
        print(f"  saved real curve figure for {drug}")

    fit_df = pd.DataFrame(fit_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fit_df.to_csv(RESULTS_DIR / "real_dose_response_curve_fits.csv", index=False)
    return fit_df


def main():
    agg = build_real_curve_dataset(
        log_path=ROOT / "docs" / "review_evidence" / "raw_dose_response_processing_log.txt"
    )
    fit_df = fit_and_plot_real_curves(agg)
    print(fit_df)
    print(f"\nSaved curve fits -> {RESULTS_DIR / 'real_dose_response_curve_fits.csv'}")


if __name__ == "__main__":
    main()
