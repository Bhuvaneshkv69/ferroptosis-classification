"""Runs the classification model-selection benchmark (see src/benchmark_run.py).

    python scripts/run_benchmark.py            # full protocol (touches the locked test set once)
    python scripts/run_benchmark.py --quick    # smoke run, never touches the locked test set
    python scripts/run_benchmark.py --task regression   # drug-response (LN_IC50) benchmark
    python scripts/run_benchmark.py --promote   # explicit researcher action; without it the production model is never changed
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--promote", action="store_true", help="allow promotion if every gate passes (default: never promote)")
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    a = ap.parse_args()
    if a.task == "regression":
        import regression_benchmark
        regression_benchmark.run_regression_benchmark(quick=a.quick, promote=a.promote)
    else:
        import benchmark_run
        benchmark_run.run_classification_benchmark(quick=a.quick, promote=a.promote)


if __name__ == "__main__":
    main()
