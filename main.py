"""
End-to-end pipeline entry point.

    python main.py                 # run everything (uses cached raw data if present)
    python main.py --fetch         # also re-fetch raw data from cBioPortal/MyGene first
                                    # (GDSC files must be placed manually, see docs/DATASET.md)

Stages:
    1. Ferroptosis gene signature curation      (src/ferroptosis.py)
    2. [optional] Raw data acquisition          (src/fetch_tcga.py, src/fetch_gene_maps.py)
    3. Preprocessing                            (src/preprocessing.py)
    4. Feature engineering                      (src/feature_engineering.py)
    5. Classification (MSI + risk-group)        (src/classification.py)
    6. Dose-response regression                 (src/dose_response.py)
    7. Drug-resistance classification           (src/drug_resistance.py)
    8. Interpretability                         (src/interpretability.py)

Each stage writes its outputs to data/processed/, results/, figures/, and
tables/, so the pipeline can also be resumed from any intermediate stage by
running that stage's module directly (e.g. `python src/dose_response.py`).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="Ferroptosis classification & dose-response pipeline")
    parser.add_argument("--fetch", action="store_true", help="Re-fetch TCGA/gene-map raw data before running")
    args = parser.parse_args()

    t0 = time.time()
    stages_run = []

    print("=" * 70)
    print("STAGE 1/7: Ferroptosis gene signature curation")
    print("=" * 70)
    import ferroptosis
    ferroptosis.main()
    stages_run.append("ferroptosis_signature")

    if args.fetch:
        print("\n" + "=" * 70)
        print("STAGE 2/7: Raw data acquisition (TCGA + gene maps)")
        print("=" * 70)
        import fetch_tcga
        import fetch_gene_maps
        fetch_tcga.main()
        fetch_gene_maps.main()
        stages_run.append("data_acquisition")
    else:
        print("\n(Skipping --fetch: using cached raw data in data/raw/. "
              "Run with --fetch to re-download from cBioPortal/MyGene.info.)")
        required = [
            ROOT / "data/raw/tcga/mrna_expression_long.csv",
            ROOT / "data/raw/gdsc/GDSC2_fitted_dose_response.csv",
            ROOT / "data/raw/gdsc/sanger1018_brainarray_ensemblgene_rma.txt",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            print("ERROR: required raw data files are missing:")
            for m in missing:
                print(f"  - {m}")
            print("See docs/DATASET.md for exact download sources (GDSC files "
                  "must be downloaded manually from ftp.sanger.ac.uk; TCGA data "
                  "can be re-fetched with `python main.py --fetch`).")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("STAGE 3/7: Preprocessing (TCGA + GDSC)")
    print("=" * 70)
    import preprocessing
    preprocessing.build_tcga_dataset(log_path=ROOT / "docs/review_evidence/tcga_preprocessing_log.txt")
    preprocessing.build_gdsc_dose_response_dataset(log_path=ROOT / "docs/review_evidence/gdsc_preprocessing_log.txt")
    stages_run.append("preprocessing")

    print("\n" + "=" * 70)
    print("STAGE 4/7: Feature engineering")
    print("=" * 70)
    import pandas as pd
    import feature_engineering
    tcga_processed = pd.read_csv(ROOT / "data/processed/tcga_coadread_processed.csv")
    with open(ROOT / "data/processed/tcga_preprocessing_metadata.json") as f:
        tcga_meta = json.load(f)
    feature_engineering.run(tcga_processed, tcga_meta["ferroptosis_genes_used"])
    stages_run.append("feature_engineering")

    print("\n" + "=" * 70)
    print("STAGE 5/8: Classification (MSI status + ferroptosis risk group / prognosis)")
    print("=" * 70)
    import classification
    classification.main()
    stages_run.append("classification")

    print("\n" + "=" * 70)
    print("STAGE 6/8: Dose-response regression")
    print("=" * 70)
    import dose_response
    dose_response.main()
    stages_run.append("dose_response")

    print("\n" + "=" * 70)
    print("STAGE 7/8: Drug-resistance classification")
    print("=" * 70)
    import drug_resistance
    drug_resistance.main()
    stages_run.append("drug_resistance")

    print("\n" + "=" * 70)
    print("STAGE 8/8: Interpretability")
    print("=" * 70)
    import interpretability
    interpretability.main()
    stages_run.append("interpretability")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {elapsed/60:.1f} min. Stages run: {stages_run}")
    print("Outputs: data/processed/, results/, figures/, tables/, docs/review_evidence/")
    print("=" * 70)


if __name__ == "__main__":
    main()
