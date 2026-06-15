"""
Step 0: Build a cleaned, renamed panel for the DiD analyses in this folder.

Input:  data/scorecard_with_longitudinal_metrics.csv
Output: data/did_analysis/panel.csv

Column renames (for readability):
  mttr_all_severities         -> mttr
  Dependency-Update-Tool_score-> sca_score  (also derives sca_used: 1/0/NaN)
  Pinned-Dependencies_score   -> pinned_dependencies_score
  pins                        -> num_pinned_deps
  floats                      -> num_floating_deps
  commits_approx              -> commits
  contributors_approx         -> maintainers
  aggregate_score             -> scorecard_aggregate_score
  published_at                -> release_date

Adds a `period` column: the 1-indexed rank of each row's release_date within its
(ecosystem, github_repo) group. This gives a regular per-unit time index for the
staggered DiD and panel FE regressions, despite irregular calendar spacing.
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
IN_CSV  = DATA_DIR / "scorecard_with_longitudinal_metrics.csv"
OUT_DIR = DATA_DIR / "did_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "panel.csv"

# Columns to keep in the output panel, using their *post-rename* names.
KEEP_COLS = [
    "ecosystem", "package_name", "package_version", "github_repo", "release_date",
    "mttr", "mttr_critical", "mttr_high", "mttr_medium", "mttr_low",
    "num_dependencies", "num_pinned_deps", "num_floating_deps",
    "version_releases", "release_frequency",
    "stars", "forks", "commits", "maintainers",
    "sca_score", "sca_used", "pinned_dependencies_score",
    "scorecard_aggregate_score", "match_method",
]


def strip_gh_prefix(url: str) -> str:
    """'https://github.com/owner/repo' -> 'owner/repo'"""
    return re.sub(r'^https?://github\.com/', '', str(url).strip().rstrip('/'))


def main():
    print("Loading scorecard_with_longitudinal_metrics.csv …")
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"  rows: {len(df):,}  columns: {len(df.columns)}")

    # ── rename / derive columns ────────────────────────────────────────────
    df = df.rename(columns={
        "mttr_all_severities": "mttr",
        "Dependency-Update-Tool_score": "sca_score",
        "Pinned-Dependencies_score": "pinned_dependencies_score",
        "pins": "num_pinned_deps",
        "floats": "num_floating_deps",
        "commits_approx": "commits",
        "contributors_approx": "maintainers",
        "aggregate_score": "scorecard_aggregate_score",
        "published_at": "release_date",
    })

    # Rows with no github_repo (match_method == 'no_match') have no entity id
    # and cannot belong to any per-repo panel — drop them.
    n_before_repo_filter = len(df)
    df = df[df["github_repo"].notna()].copy()
    print(f"\nDropped {n_before_repo_filter - len(df):,} rows with no github_repo "
          f"(match_method == 'no_match'); {len(df):,} remain")

    df["github_repo"] = df["github_repo"].apply(strip_gh_prefix)
    df["release_date"] = pd.to_datetime(df["release_date"], utc=True, errors="coerce")

    # sca_used: 1 if sca_score==10, 0 if sca_score==0, NaN otherwise (covers -1 and NaN)
    df["sca_used"] = df["sca_score"].map({10.0: 1, 0.0: 0})

    df = df[[c for c in KEEP_COLS if c in df.columns]]

    # ── sort, dedupe, period index ─────────────────────────────────────────
    df = df.sort_values(["ecosystem", "github_repo", "release_date"], kind="stable")
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nDropped {before - len(df):,} exact duplicate rows ({len(df):,} remain)")

    df["period"] = (
        df.groupby(["ecosystem", "github_repo"]).cumcount() + 1
    )

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved panel -> {OUT_CSV}")
    print(f"  rows: {len(df):,}  columns: {len(df.columns)}")
    print(f"  unique (ecosystem, github_repo): {df.groupby(['ecosystem','github_repo']).ngroups:,}")

    # ── summary stats ────────────────────────────────────────────────────
    print("\nperiod (releases per repo) distribution:")
    max_period = df.groupby(["ecosystem", "github_repo"])["period"].max()
    print(max_period.describe())

    print("\nsca_used value counts:")
    print(df["sca_used"].value_counts(dropna=False))

    sca_by_unit = df.groupby(["ecosystem", "github_repo"])["sca_used"].agg(
        lambda s: set(s.dropna().unique())
    )
    n_never = (sca_by_unit == {0.0}).sum()
    n_always = (sca_by_unit == {1.0}).sum()
    n_switch = (sca_by_unit == {0.0, 1.0}).sum()
    n_unusable = (sca_by_unit == set()).sum()
    print(f"\nSCA adoption status per unit (ecosystem, github_repo) — total {len(sca_by_unit):,}:")
    print(f"  never-treated (always 0)         : {n_never:,}")
    print(f"  always-treated (always 1)        : {n_always:,}")
    print(f"  switchers (both 0 and 1 observed): {n_switch:,}")
    print(f"  unusable (no valid sca_score)    : {n_unusable:,}")


if __name__ == "__main__":
    main()
