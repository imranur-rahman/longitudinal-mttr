"""
Step 5: Export a clean, human-readable analysis dataset.

Reads data/scorecard_with_longitudinal_metrics.csv (52 columns), keeps only the
18 columns used in the causal analyses (outcome, 9 continuous covariates, binary
treatment, identifiers), renames them to clear snake_case names, and drops the
73,741 rows with no GitHub match (match_method == 'no_match'), which lack the
GitHub-derived covariates.

Outputs:
  data/analysis_dataset.csv         — 18-column, 247,342-row clean dataset
  data/analysis_dataset_columns.csv — column dictionary (name, original, description)
"""

from pathlib import Path

import pandas as pd

IN_CSV = Path("data") / "scorecard_with_longitudinal_metrics.csv"
OUT_CSV = Path("data") / "analysis_dataset.csv"
OUT_DICT = Path("data") / "analysis_dataset_columns.csv"

# ── column mapping: (original_name, readable_name, description) ──────────────
COLUMN_MAP = [
    ("ecosystem",
     "ecosystem",
     "Package registry: 'npm' or 'pypi'"),
    ("package_name",
     "package_name",
     "Package identifier in the registry"),
    ("package_version",
     "package_version",
     "Version string (e.g. '1.2.3')"),
    ("published_at",
     "release_date",
     "Date this version was published on npm or PyPI"),
    ("github_repo",
     "github_repo",
     "GitHub repository URL (https://github.com/owner/repo)"),
    ("match_method",
     "scorecard_match_method",
     "How the Scorecard entry was matched to this version: "
     "'version_match', 'date_fallback', or 'no_match'"),
    ("mttr_all_severities",
     "mttr_days",
     "OUTCOME. Days to remediate vulnerable dependencies across all severities. "
     "0.0 means no vulnerable dependencies existed in the observation window."),
    ("num_dependencies",
     "num_direct_dependencies",
     "Number of direct dependencies declared by this package version"),
    ("pins",
     "num_pinned_dependencies",
     "Number of dependencies pinned to an exact version (e.g. '==1.2.3' or '1.2.3')"),
    ("floats",
     "num_floating_dependencies",
     "Number of dependencies with a version range (e.g. '^1.2.0' or '>=1.0')"),
    ("version_releases",
     "total_version_releases",
     "Total number of versions ever released by this package"),
    ("release_frequency",
     "release_frequency_days",
     "Average number of days between consecutive releases for this package"),
    ("stars",
     "github_stars",
     "GitHub star count at the time of the Scorecard run"),
    ("forks",
     "github_forks",
     "GitHub fork count"),
    ("commits_approx",
     "github_commits",
     "Approximate total commit count for the repository"),
    ("contributors_approx",
     "github_maintainers",
     "Approximate number of unique contributors (proxy for team size)"),
    ("Dependency-Update-Tool_score",
     "sca_tool_score",
     "BINARY TREATMENT. Scorecard SCA/dependency-update tool detection: "
     "10 = tool detected (e.g. Dependabot/Renovate), 0 = not detected, -1 = unknown"),
    ("aggregate_score",
     "scorecard_overall_score",
     "Overall OpenSSF Scorecard security score (0–10, higher = more secure practices)"),
]

ORIGINAL_COLS = [c[0] for c in COLUMN_MAP]
READABLE_COLS = [c[1] for c in COLUMN_MAP]
DESCRIPTIONS  = [c[2] for c in COLUMN_MAP]


def main():
    print("Loading scorecard_with_longitudinal_metrics.csv …")
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"  rows: {len(df):,}  columns: {len(df.columns)}")

    # ── drop unmatched rows (no GitHub covariates) ─────────────────────────
    n_before = len(df)
    df = df[df["match_method"] != "no_match"].copy()
    print(f"\nDropped {n_before - len(df):,} rows with match_method == 'no_match' "
          f"(no GitHub covariates available); {len(df):,} remain")

    # ── select and rename ──────────────────────────────────────────────────
    df = df[ORIGINAL_COLS].rename(columns=dict(zip(ORIGINAL_COLS, READABLE_COLS)))

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(f"  rows: {len(df):,}  columns: {len(df.columns)}")

    # ── column dictionary ──────────────────────────────────────────────────
    col_dict = pd.DataFrame({
        "readable_name": READABLE_COLS,
        "original_name": ORIGINAL_COLS,
        "description": DESCRIPTIONS,
    })
    col_dict.to_csv(OUT_DICT, index=False)
    print(f"Saved -> {OUT_DICT}")

    # ── print data dictionary to stdout ───────────────────────────────────
    print("\nColumn dictionary:")
    print(f"  {'Readable name':<35} {'Original name':<35} Description")
    print(f"  {'-'*35} {'-'*35} {'-'*40}")
    for _, row in col_dict.iterrows():
        desc = row["description"][:60] + "…" if len(row["description"]) > 60 else row["description"]
        print(f"  {row['readable_name']:<35} {row['original_name']:<35} {desc}")

    # ── quick completeness check ───────────────────────────────────────────
    print("\nColumn completeness (% non-null):")
    for col in READABLE_COLS:
        pct = df[col].notna().mean() * 100
        print(f"  {col:<35} {pct:.1f}%")


if __name__ == "__main__":
    main()
