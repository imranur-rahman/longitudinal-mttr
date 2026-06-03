"""
Step 0: Preprocess the scorecard CSV to extract unique inputs for downstream collectors.

Outputs (in data/collected/):
  unique_repos_dates.csv        — unique (repo_name, published_at) for BigQuery
  unique_package_versions.csv   — unique (ecosystem, package_name, package_version,
                                           tag_name, github_repo, published_at) for deps.dev
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "collected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORECARD_CSV = DATA_DIR / "scorecard" / "scorecard_local_metrics_final.csv"
MATCHED_CSV   = DATA_DIR / "matched_scorecard_mttr.csv"


def strip_gh_prefix(url: str) -> str:
    """'https://github.com/owner/repo' → 'owner/repo'"""
    return re.sub(r'^https?://github\.com/', '', str(url).strip().rstrip('/'))


def infer_version(tag: str) -> str:
    """Strip leading 'v' or 'V' from a git tag to get a registry-like version."""
    t = str(tag).strip()
    return t[1:] if t and t[0].lower() == 'v' else t


def main():
    print("Loading scorecard CSV …")
    sc = pd.read_csv(SCORECARD_CSV, low_memory=False)
    print(f"  scorecard rows: {len(sc):,}")

    print("Loading matched CSV for ecosystem lookup …")
    matched = pd.read_csv(MATCHED_CSV, low_memory=False,
                          usecols=['ecosystem', 'package_name', 'package_version',
                                   'source_file', 'tag_name', 'github_repo'])
    print(f"  matched rows:   {len(matched):,}")

    # Build ecosystem + package_version lookup keyed on (source_file, tag_name)
    matched['_key'] = list(zip(matched['source_file'], matched['tag_name']))
    lookup = (matched.drop_duplicates(subset=['_key'])
                     .set_index('_key')[['ecosystem', 'package_version']])

    sc['_key'] = list(zip(sc['source_file'], sc['tag_name']))
    sc = sc.join(lookup, on='_key', how='left')

    unmatched = sc['ecosystem'].isna().sum()
    print(f"  rows without ecosystem match: {unmatched:,} ({100*unmatched/len(sc):.1f}%)")

    # For unmatched rows, try to guess ecosystem from package name patterns
    # (simple heuristic — may be wrong; these rows are excluded from deps.dev calls)
    sc['package_version'] = sc['package_version'].fillna(sc['tag_name'].apply(infer_version))

    # Normalize github_repo to 'owner/repo' format for BigQuery
    sc['repo_name'] = sc['github_repo'].apply(strip_gh_prefix)

    # Normalize published_at to UTC ISO string
    sc['published_at_utc'] = (
        pd.to_datetime(sc['published_at'], utc=True, errors='coerce')
          .dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    )

    # ── Output 1: unique (repo_name, published_at) for BigQuery ──────────────
    repos_dates = (sc[sc['repo_name'].notna() & (sc['repo_name'] != '')]
                     .drop_duplicates(subset=['repo_name', 'published_at_utc'])
                     [['repo_name', 'published_at_utc']]
                     .rename(columns={'published_at_utc': 'published_at'}))
    repos_dates.to_csv(OUT_DIR / "unique_repos_dates.csv", index=False)
    print(f"\nunique_repos_dates.csv: {len(repos_dates):,} rows")
    print(f"  unique repos: {repos_dates['repo_name'].nunique():,}")

    # ── Output 2: unique (ecosystem, package_name, …) for deps.dev ───────────
    pkg_vers = (sc[sc['ecosystem'].notna()]
                  .drop_duplicates(subset=['ecosystem', 'package_name', 'package_version'])
                  [['ecosystem', 'package_name', 'package_version',
                    'tag_name', 'repo_name', 'published_at_utc']]
                  .rename(columns={'repo_name': 'github_repo',
                                   'published_at_utc': 'published_at'}))
    pkg_vers.to_csv(OUT_DIR / "unique_package_versions.csv", index=False)
    print(f"\nunique_package_versions.csv: {len(pkg_vers):,} rows")
    print(pkg_vers['ecosystem'].value_counts().to_string())


if __name__ == '__main__':
    main()
