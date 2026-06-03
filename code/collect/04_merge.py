"""
Step 4: Merge all collected metrics back onto the original scorecard CSV.

Joins:
  github_metrics.csv        on (repo_name, published_at)     → stars, forks,
                                                                commits_approx, contributors_approx
  depsdev_releases.csv      on (ecosystem, package_name,      → version_releases,
                                package_version)                release_frequency
  depsdev_requirements.csv  on (ecosystem, package_name,      → pins, floats
                                package_version)

Output: data/scorecard_with_longitudinal_metrics.csv
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR   = Path("data")
OUT_DIR    = DATA_DIR / "collected"
OUTPUT_CSV = DATA_DIR / "scorecard_with_longitudinal_metrics.csv"

SCORECARD_CSV   = DATA_DIR / "scorecard" / "scorecard_local_metrics_final.csv"
MATCHED_CSV     = DATA_DIR / "matched_scorecard_mttr.csv"
GH_METRICS_CSV  = OUT_DIR / "github_metrics.csv"
REL_CSV         = OUT_DIR / "depsdev_releases.csv"
REQ_CSV         = OUT_DIR / "depsdev_requirements.csv"


def strip_gh_prefix(url: str) -> str:
    return re.sub(r'^https?://github\.com/', '', str(url).strip().rstrip('/'))


def norm_ts(series: pd.Series) -> pd.Series:
    """Normalize timestamps to UTC-aware, then format as string for joining."""
    return pd.to_datetime(series, utc=True, errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S UTC')


def main():
    # ── Load base (matched scorecard has ecosystem + package_version) ──────────
    print("Loading matched scorecard CSV …")
    base = pd.read_csv(MATCHED_CSV, low_memory=False)
    print(f"  {len(base):,} rows")

    base['repo_name']      = base['github_repo'].apply(strip_gh_prefix)
    base['published_at_n'] = norm_ts(base['published_at'])

    # ── GitHub metrics ────────────────────────────────────────────────────────
    if GH_METRICS_CSV.exists():
        print("Loading github_metrics.csv …")
        gh = pd.read_csv(GH_METRICS_CSV)
        gh['published_at_n'] = norm_ts(gh['published_at'])
        gh = gh.drop(columns=['published_at'], errors='ignore')
        base = base.merge(
            gh,
            left_on=['repo_name', 'published_at_n'],
            right_on=['repo_name', 'published_at_n'],
            how='left',
        )
        print(f"  after GitHub join: {len(base):,} rows")
        print(f"  stars non-null: {base['stars'].notna().sum():,}")
    else:
        print(f"WARN: {GH_METRICS_CSV} not found — GitHub columns will be missing.")

    # ── deps.dev release metrics ──────────────────────────────────────────────
    if REL_CSV.exists():
        print("Loading depsdev_releases.csv …")
        rel = pd.read_csv(REL_CSV)
        base = base.merge(
            rel,
            on=['ecosystem', 'package_name', 'package_version'],
            how='left',
        )
        print(f"  version_releases non-null: {base['version_releases'].notna().sum():,}")
    else:
        print(f"WARN: {REL_CSV} not found — release columns will be missing.")

    # ── deps.dev requirements (pins/floats) ───────────────────────────────────
    if REQ_CSV.exists():
        print("Loading depsdev_requirements.csv …")
        req = pd.read_csv(REQ_CSV)
        base = base.merge(
            req,
            on=['ecosystem', 'package_name', 'package_version'],
            how='left',
        )
        print(f"  pins non-null: {base['pins'].notna().sum():,}")
    else:
        print(f"WARN: {REQ_CSV} not found — pins/floats columns will be missing.")

    # ── Clean up join helpers ─────────────────────────────────────────────────
    base = base.drop(columns=['repo_name', 'published_at_n'], errors='ignore')

    base.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved {len(base):,} rows × {len(base.columns)} columns to:\n  {OUTPUT_CSV}")

    # Coverage summary
    new_cols = ['stars', 'forks', 'commits_approx', 'contributors_approx',
                'version_releases', 'release_frequency', 'pins', 'floats']
    print("\nCoverage summary:")
    for col in new_cols:
        if col in base.columns:
            pct = 100 * base[col].notna().mean()
            print(f"  {col:<30} {pct:5.1f}% non-null")


if __name__ == '__main__':
    main()
