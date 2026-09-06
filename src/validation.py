"""
Validation utilities for user-supplied gene-expression input (CSV upload in
the Streamlit application, or programmatic use). Detects the failure modes
explicitly called out in the project brief: duplicate genes, missing genes,
incompatible identifiers, zero-variance features, and malformed expression
data. Never silently drops or substitutes data -- every issue found is
returned to the caller to display, not swallowed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_required_gene_panel() -> list[str]:
    return pd.read_csv(ROOT / "data" / "unified_gene_signature.csv")["gene"].tolist()


def validate_expression_dataframe(df: pd.DataFrame, required_genes: list[str] | None = None) -> dict:
    """
    Validates a (samples x genes) expression DataFrame (genes as columns,
    one row per sample) intended for classification/dose-response inference.

    Returns a dict:
        is_valid: bool
        errors: list[str]      -- issues that block inference
        warnings: list[str]    -- issues that don't block inference but matter
        missing_genes: list[str]
        duplicate_gene_columns: list[str]
        zero_variance_genes: list[str]
        non_numeric_genes: list[str]
        n_samples: int
        n_genes_provided: int
        n_genes_usable: int
    """
    required_genes = required_genes or load_required_gene_panel()
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        return {
            "is_valid": False, "errors": ["Uploaded expression table is empty (0 rows)."],
            "warnings": [], "missing_genes": required_genes, "duplicate_gene_columns": [],
            "zero_variance_genes": [], "non_numeric_genes": [], "n_samples": 0,
            "n_genes_provided": 0, "n_genes_usable": 0,
        }

    columns = list(df.columns)
    duplicate_gene_columns = sorted({c for c in columns if columns.count(c) > 1})
    if duplicate_gene_columns:
        errors.append(f"Duplicate gene columns found (ambiguous which to use): {duplicate_gene_columns}")

    provided = set(columns)
    missing_genes = sorted(set(required_genes) - provided)
    if missing_genes:
        errors.append(
            f"{len(missing_genes)}/{len(required_genes)} required ferroptosis genes are missing "
            f"from the uploaded file: {missing_genes}"
        )

    usable_genes = [g for g in required_genes if g in provided and g not in duplicate_gene_columns]

    non_numeric_genes = []
    zero_variance_genes = []
    for g in usable_genes:
        col = df[g]
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.isna().sum() > col.isna().sum():
            non_numeric_genes.append(g)
            continue
        if len(df) > 1 and numeric.dropna().nunique() <= 1:
            zero_variance_genes.append(g)

    if non_numeric_genes:
        errors.append(f"Non-numeric / malformed values found in columns: {non_numeric_genes}")
    if zero_variance_genes:
        warnings.append(
            f"Zero-variance columns (identical value across all uploaded samples, uninformative "
            f"for a multi-sample batch, harmless for a single-sample prediction): {zero_variance_genes}"
        )

    any_negative = {g: bool((pd.to_numeric(df[g], errors="coerce") < 0).any())
                    for g in usable_genes if g not in non_numeric_genes}
    negative_genes = sorted([g for g, has_neg in any_negative.items() if has_neg])
    if negative_genes:
        warnings.append(
            f"Negative expression values found in {len(negative_genes)} gene column(s) "
            f"({negative_genes[:5]}{'...' if len(negative_genes) > 5 else ''}); these will be "
            f"clipped to 0 before the log2 transform, matching training-time preprocessing."
        )

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "missing_genes": missing_genes,
        "duplicate_gene_columns": duplicate_gene_columns,
        "zero_variance_genes": zero_variance_genes,
        "non_numeric_genes": non_numeric_genes,
        "n_samples": len(df),
        "n_genes_provided": len(provided),
        "n_genes_usable": len(usable_genes),
    }


def make_template_csv(required_genes: list[str] | None = None, example_values: dict | None = None) -> pd.DataFrame:
    """Builds a one-row example CSV (as a DataFrame) with the exact required
    gene columns, so a user knows precisely what format to upload."""
    required_genes = required_genes or load_required_gene_panel()
    example_values = example_values or {}
    row = {g: example_values.get(g, "") for g in required_genes}
    return pd.DataFrame([row])
