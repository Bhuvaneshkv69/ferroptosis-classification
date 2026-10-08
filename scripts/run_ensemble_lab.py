"""Runs the ensemble lab (see src/ensemble_lab.py).

    python scripts/run_ensemble_lab.py                      # all four feature sets (touches the locked test set once per set)
    python scripts/run_ensemble_lab.py --quick core         # CV-only smoke run, never touches the locked test set
    python scripts/run_ensemble_lab.py --sets core extended_molecular
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", nargs="*", default=None, help="CV-only smoke run on the given sets")
    ap.add_argument("--sets", nargs="*", default=None)
    a = ap.parse_args()
    import ensemble_lab
    if a.quick is not None:
        ensemble_lab.run_ensemble_lab(quick=True, feature_sets=a.quick or ["core"])
    else:
        ensemble_lab.run_ensemble_lab(feature_sets=a.sets)


if __name__ == "__main__":
    main()
