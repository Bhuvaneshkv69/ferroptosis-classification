"""Structural safety checks shared by every promotion gate (classification and drug response).

A model is only PROMOTABLE if all of these hold, regardless of how good its score is. Score-based criteria
(CV improvement, locked-test not worse, stability, leakage) are separate and are applied on top by the caller.
Promotion also needs an explicit researcher action; a better metric alone never promotes anything.
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_META = ("dataset", "feature_pipeline", "trained_at", "path", "feature_columns")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def structural_failures(candidate: dict, incumbent: dict | None, *, required_cv: tuple, required_test: tuple,
                        root: Path = ROOT, check_pipeline: bool = True) -> list[str]:
    """Reasons the candidate (a registry-entry-shaped dict) cannot be promoted; empty list = structurally fine."""
    why = []
    missing = [k for k in REQUIRED_META if not candidate.get(k)]
    if missing:
        why.append(f"required metadata missing: {', '.join(missing)}")
    path = candidate.get("path")
    if path and not (root / path).exists():
        why.append(f"model artifact does not exist on disk: {path}")
    cv, te = candidate.get("cv_metrics") or {}, candidate.get("test_metrics") or {}
    bad = [k for k in required_cv if not _finite(cv.get(k))] + [k for k in required_test if not _finite(te.get(k))]
    if bad:
        why.append(f"required validation metrics missing or not finite: {', '.join(bad)}")
    if incumbent is not None:
        if list(candidate.get("feature_columns") or []) != list(incumbent.get("feature_columns") or []):
            why.append("feature set differs from the production model (the app feeds the production slot a fixed feature set)")
        if check_pipeline and candidate.get("feature_pipeline") != incumbent.get("feature_pipeline"):
            why.append("preprocessing / feature pipeline differs from the production model")
        tc = (candidate.get("extra") or {}).get("target_column")
        ti = (incumbent.get("extra") or {}).get("target_column")
        if tc is not None and ti is not None and tc != ti:
            why.append(f"response target differs from the production model ({tc} vs {ti})")
    return why
