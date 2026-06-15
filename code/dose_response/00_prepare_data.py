"""
Step 0: Build a cleaned cross-sectional dataset for the GPS dose-response
analysis in this folder.

Input:  data/scorecard_with_longitudinal_metrics.csv
Output: data/dose_response/data.csv

For each row (a matched package_version), keeps the outcome (mttr), the 8
dose-variable candidates, and an extra confounder (scorecard_aggregate_score).
Drops rows with no GitHub match (match_method == 'no_match') since the
GitHub-derived covariates (stars, forks, commits, maintainers) are only
populated for matched rows, and drops rows with missing values in any of the
columns used downstream.

Heavily right-skewed count covariates are log1p-transformed (suffix `_log`)
for use as the GPS dose/confounders, since the GPS step assumes an
approximately normal conditional distribution of the dose given confounders.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
IN_CSV = DATA_DIR / "scorecard_with_longitudinal_metrics.csv"
OUT_DIR = DATA_DIR / "dose_response"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "data.csv"

# Columns to keep, post-rename.
KEEP_COLS = [
    "ecosystem", "package_name", "package_version", "github_repo",
    "mttr",
    "num_dependencies", "num_pinned_deps", "num_floating_deps",
    "release_frequency", "stars", "forks", "commits", "maintainers",
    "scorecard_aggregate_score",
]

# Count covariates that are heavily right-skewed -> log1p before GPS modeling.
LOG_COLS = [
    "num_dependencies", "num_pinned_deps", "num_floating_deps",
    "stars", "forks", "commits", "maintainers",
]


def main():
    print("Loading scorecard_with_longitudinal_metrics.csv …")
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"  rows: {len(df):,}  columns: {len(df.columns)}")

    # ── rename columns ──────────────────────────────────────────────────
    df = df.rename(columns={
        "mttr_all_severities": "mttr",
        "pins": "num_pinned_deps",
        "floats": "num_floating_deps",
        "commits_approx": "commits",
        "contributors_approx": "maintainers",
        "aggregate_score": "scorecard_aggregate_score",
    })

    # ── drop unmatched rows (no GitHub-derived covariates) ─────────────────
    n_before = len(df)
    df = df[df["match_method"] != "no_match"].copy()
    print(f"\nDropped {n_before - len(df):,} rows with match_method == 'no_match'; "
          f"{len(df):,} remain")

    df = df[[c for c in KEEP_COLS if c in df.columns]]

    # ── missingness report ──────────────────────────────────────────────
    print("\nMissingness before dropping:")
    print(df.isna().sum().to_string())

    n_before = len(df)
    df = df.dropna(subset=KEEP_COLS).copy()
    print(f"\nDropped {n_before - len(df):,} rows with missing values in "
          f"required columns; {len(df):,} remain")

    # ── log1p-transform skewed count covariates ─────────────────────────
    for col in LOG_COLS:
        df[f"{col}_log"] = np.log1p(df[col])

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(f"  rows: {len(df):,}  columns: {len(df.columns)}")

    # ── descriptive stats ────────────────────────────────────────────────
    print("\nDose variable descriptive stats (original scale):")
    print(df[["num_dependencies", "num_pinned_deps", "num_floating_deps",
              "release_frequency", "stars", "forks", "commits", "maintainers"]]
          .describe().T[["min", "50%", "max", "mean", "std"]])

    print("\nLog-transformed dose variable descriptive stats:")
    log_cols = [f"{c}_log" for c in LOG_COLS]
    print(df[log_cols].describe().T[["min", "50%", "max", "mean", "std"]])

    print("\nOutcome (mttr) descriptive stats:")
    print(df["mttr"].describe())


if __name__ == "__main__":
    main()
