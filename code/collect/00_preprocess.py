"""
Step 0: Match npm/pypi releases against the scorecard dataset and extract
unique inputs for downstream collectors.

Pre-filter: only npm/pypi releases Oct 6 2023 – Oct 6 2025 (scorecard data window).

Scorecard enrichment: the Dependency-Update-Tool check is missing from
scorecard_local_metrics_final.csv, so it is sourced from
data/dependency_update_tool/package_data_final_with_scores_dut_fixed.csv
(a JSON "scorecard" blob per package_name + tag_name), deduplicated to the
most recent run per (package_name, tag_name), and left-merged onto the
scorecard data as Dependency-Update-Tool_score / Dependency-Update-Tool_reason
before matching begins.

Two-pass matching strategy:
  Pass 1 — normalized version match (package_name + normalized tag_name == package_version)
  Pass 2 — date fallback for remaining unmatched rows (package_name + date within ±1 day)

Version normalization rules applied to scorecard tag_name:
  Rule 1  v-prefix     : "v3.0.0"              → "3.0.0"      (strip leading lowercase v)
  Rule 2  V-prefix     : "V0.1.0"              → "0.1.0"      (strip leading uppercase V)
  Rule 3  pkg@version  : "@scope/pkg@1.2.3"    → "1.2.3"      (substring after last @)
  Rule 4  bare version : "3.0.0"               → "3.0.0"      (no change, starts with digit)
  Rule 5  unmatchable  : "MISSING_TAG", "latest", NaN, other  → None (skip in pass 1)

When multiple scorecard rows match the same (ecosystem, package_name,
package_version), one is picked to keep exactly one output row per pair:
  Pass 1 — most recent run_timestamp
  Pass 2 — smallest day_diff, then most recent run_timestamp

Outputs:
  data/matched_scorecard_mttr.csv
    All npm/pypi columns + all scorecard columns (incl. Dependency-Update-Tool_score/
    _reason) + match_method column.
    This is a LEFT join on (ecosystem, package_name, package_version): every npm/pypi
    (ecosystem, package, version) pair in scope is kept exactly once, even if no
    scorecard row matches (scorecard columns are left as NaN and match_method is
    "no_match").

  data/collected/unique_repos_dates.csv
    Unique (repo_name, published_at) for BigQuery — rows with a scorecard match only.

  data/collected/unique_package_versions.csv
    Unique (ecosystem, package_name, package_version, tag_name, github_repo,
    published_at) for deps.dev — rows with a scorecard match only.
"""

import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "collected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SC_START = pd.Timestamp("2023-10-06").date()
SC_END   = pd.Timestamp("2025-10-06").date()

MATCHED_OUT_CSV = DATA_DIR / "matched_scorecard_mttr.csv"


# ── version normalization ─────────────────────────────────────────────────────

def normalize_tag(tag) -> str | None:
    """Return a plain version string from a scorecard tag_name, or None if unmatchable."""
    if pd.isna(tag) or tag in ("MISSING_TAG", "latest", ""):
        return None
    tag = str(tag).strip()
    # Rule 3: pkg@version — "@scope/name@1.2.3" or "name@1.2.3"
    if "@" in tag:
        last_at = tag.rfind("@")
        candidate = tag[last_at + 1:]
        if re.match(r"[0-9]", candidate):
            return candidate
    # Rule 1 & 2: v/V prefix
    if re.match(r"^[vV][0-9]", tag):
        return tag[1:]
    # Rule 4: bare version
    if re.match(r"^[0-9]", tag):
        return tag
    # Rule 5: unmatchable (rel1.9.0, prerelease-v6.x, v.1.7.0, etc.)
    return None


def tag_norm_rule(tag) -> str:
    """Return which normalization rule was applied (for reporting)."""
    if pd.isna(tag) or tag in ("MISSING_TAG", "latest", ""):
        return "unmatchable"
    tag = str(tag).strip()
    if "@" in tag:
        last_at = tag.rfind("@")
        if re.match(r"[0-9]", tag[last_at + 1:]):
            return "Rule 3: pkg@version"
    if re.match(r"^v[0-9]", tag):
        return "Rule 1: v-prefix"
    if re.match(r"^V[0-9]", tag):
        return "Rule 2: V-prefix"
    if re.match(r"^[0-9]", tag):
        return "Rule 4: bare version"
    return "unmatchable"


def extract_dut_check(scorecard_json):
    """Return (score, reason, date) for the Dependency-Update-Tool check in a
    scorecard JSON blob, or (None, None, None) if missing/malformed."""
    if pd.isna(scorecard_json):
        return None, None, None
    try:
        blob = json.loads(scorecard_json)
    except (TypeError, ValueError):
        return None, None, None
    for check in blob.get("checks") or []:
        if check.get("name") == "Dependency-Update-Tool":
            return check.get("score"), check.get("reason"), blob.get("date")
    return None, None, None


def strip_gh_prefix(url: str) -> str:
    """'https://github.com/owner/repo' → 'owner/repo'"""
    return re.sub(r'^https?://github\.com/', '', str(url).strip().rstrip('/'))


def match_scorecard():
    print("Loading data …")

    scorecard = pd.read_csv(
        DATA_DIR / "scorecard" / "scorecard_local_metrics_final.csv",
        low_memory=False,
    )
    npm_full = pd.read_csv(
        DATA_DIR / "npm" / "npm_data_depsdev_per_release_results.csv",
        low_memory=False,
    )
    pypi_full = pd.read_csv(
        DATA_DIR / "pypi" / "pypi_data_depsdev_per_release_results.csv",
        low_memory=False,
    )

    dut = pd.read_csv(
        DATA_DIR / "dependency_update_tool" / "package_data_final_with_scores_dut_fixed.csv",
        low_memory=False,
    )

    print(f"  scorecard rows (raw) : {len(scorecard):>10,}")
    print(f"  npm rows      (raw)  : {len(npm_full):>10,}")
    print(f"  pypi rows     (raw)  : {len(pypi_full):>10,}")
    print(f"  dependency_update_tool rows (raw) : {len(dut):>10,}")

    # ── enrich scorecard with Dependency-Update-Tool check from dut dataset ────

    dut[["Dependency-Update-Tool_score", "Dependency-Update-Tool_reason", "dut_date"]] = (
        dut["scorecard"].apply(lambda s: pd.Series(extract_dut_check(s)))
    )
    dut["dut_date"] = pd.to_datetime(dut["dut_date"], utc=True, errors="coerce")

    dut_dedup = (
        dut.sort_values("dut_date", ascending=False)
           .drop_duplicates(subset=["package_name", "tag_name"], keep="first")
           [["package_name", "tag_name", "Dependency-Update-Tool_score", "Dependency-Update-Tool_reason"]]
    )

    scorecard = scorecard.merge(dut_dedup, on=["package_name", "tag_name"], how="left")
    print(f"  Dependency-Update-Tool scores matched onto scorecard : "
          f"{scorecard['Dependency-Update-Tool_score'].notna().sum():,} / {len(scorecard):,}")

    # ── pre-filter: drop pre-Oct-2023 npm/pypi releases ──────────────────────

    npm_full["package_release_date"]  = pd.to_datetime(npm_full["package_release_date"],  errors="coerce").dt.date
    pypi_full["package_release_date"] = pd.to_datetime(pypi_full["package_release_date"], errors="coerce").dt.date

    npm  = npm_full[(npm_full["package_release_date"]  >= SC_START) & (npm_full["package_release_date"]  <= SC_END)].copy()
    pypi = pypi_full[(pypi_full["package_release_date"] >= SC_START) & (pypi_full["package_release_date"] <= SC_END)].copy()

    print(f"\nAfter filtering to Oct 6 2023 – Oct 6 2025:")
    print(f"  npm  kept : {len(npm):>8,}  (dropped {len(npm_full)-len(npm):,})")
    print(f"  pypi kept : {len(pypi):>8,}  (dropped {len(pypi_full)-len(pypi):,})")

    combined = pd.concat([npm.assign(ecosystem="npm"), pypi.assign(ecosystem="pypi")],
                         ignore_index=True)

    total_pairs = len(combined)
    print(f"\nTotal (package, version) pairs to match: {total_pairs:,}")

    # ── prepare scorecard ─────────────────────────────────────────────────────

    scorecard["version_normalized"] = scorecard["tag_name"].apply(normalize_tag)
    scorecard["norm_rule"]          = scorecard["tag_name"].apply(tag_norm_rule)
    scorecard["published_date"]     = (
        pd.to_datetime(scorecard["published_at"], utc=True, errors="coerce").dt.date
    )

    print("\nScorecard tag_name normalization breakdown:")
    rule_counts = scorecard["norm_rule"].value_counts()
    for rule, cnt in rule_counts.items():
        print(f"  {rule:<30}: {cnt:>8,}  ({cnt/len(scorecard)*100:.1f}%)")

    sc_matchable = scorecard.dropna(subset=["version_normalized"])

    # ── pass 1: version-normalized match ─────────────────────────────────────

    print("\n── Pass 1: version-normalized match ──")

    m1 = combined.merge(
        sc_matchable,
        left_on=["package_name", "package_version"],
        right_on=["package_name", "version_normalized"],
        how="inner",
    )
    m1["match_method"] = "pass1_version"

    # dedupe to one row per (ecosystem, package_name, package_version): keep the
    # scorecard row with the most recent run_timestamp
    m1["_run_ts"] = pd.to_datetime(m1["run_timestamp"], utc=True, errors="coerce")
    m1 = (
        m1.sort_values("_run_ts", ascending=False)
          .drop_duplicates(subset=["ecosystem", "package_name", "package_version"], keep="first")
          .drop(columns=["_run_ts"])
    )

    unique_pairs_p1 = m1
    n_p1 = len(unique_pairs_p1)
    print(f"  Unique (package, version) pairs matched : {n_p1:,} / {total_pairs:,} ({n_p1/total_pairs*100:.1f}%)")
    print(f"    npm : {unique_pairs_p1[unique_pairs_p1.ecosystem=='npm'].shape[0]:,}")
    print(f"    pypi: {unique_pairs_p1[unique_pairs_p1.ecosystem=='pypi'].shape[0]:,}")
    print(f"  Total rows in join : {len(m1):,}")

    # date agreement check (informational)
    date_diff = (
        pd.to_datetime(m1["package_release_date"]) -
        pd.to_datetime(m1["published_date"])
    ).abs().dt.days
    print(f"  Date exact agreement (day==0) : {(date_diff==0).sum():,} / {len(m1):,} rows")
    print(f"  Date within ±1 day            : {(date_diff<=1).sum():,} / {len(m1):,} rows")

    # ── pass 2: date-based fallback ───────────────────────────────────────────

    print("\n── Pass 2: date-based fallback (±1 day) ──")

    matched_keys_p1 = set(zip(
        unique_pairs_p1["ecosystem"],
        unique_pairs_p1["package_name"],
        unique_pairs_p1["package_version"],
    ))
    combined["_key"] = list(zip(combined["ecosystem"], combined["package_name"], combined["package_version"]))
    unmatched = combined[~combined["_key"].isin(matched_keys_p1)].drop(columns=["_key"]).copy()
    combined = combined.drop(columns=["_key"])
    print(f"  Unmatched after pass 1: {len(unmatched):,}")

    sc_dated = scorecard.dropna(subset=["published_date"]).copy()
    sc_dated["published_date_dt"] = pd.to_datetime(sc_dated["published_date"])
    unmatched["release_date_dt"]  = pd.to_datetime(unmatched["package_release_date"])

    # cross-join on package_name then filter by date tolerance
    m2 = unmatched.merge(sc_dated, on="package_name", how="inner")
    m2["day_diff"] = (m2["release_date_dt"] - m2["published_date_dt"]).abs().dt.days
    m2 = m2[m2["day_diff"] <= 1].drop(columns=["release_date_dt", "published_date_dt"])
    m2["match_method"] = m2["day_diff"].apply(
        lambda d: "pass2_date_exact" if d == 0 else "pass2_date_tol1"
    )

    # dedupe to one row per (ecosystem, package_name, package_version): keep the
    # smallest day_diff, then the most recent run_timestamp
    m2["_run_ts"] = pd.to_datetime(m2["run_timestamp"], utc=True, errors="coerce")
    m2 = (
        m2.sort_values(["day_diff", "_run_ts"], ascending=[True, False])
          .drop_duplicates(subset=["ecosystem", "package_name", "package_version"], keep="first")
          .drop(columns=["_run_ts"])
    )

    unique_pairs_p2 = m2
    n_p2 = len(unique_pairs_p2)
    n_p2_exact = (unique_pairs_p2["match_method"] == "pass2_date_exact").sum()
    n_p2_tol1  = (unique_pairs_p2["match_method"] == "pass2_date_tol1").sum()

    print(f"  Additional (package, version) pairs matched : {n_p2:,}")
    print(f"    exact date (day diff = 0) : {n_p2_exact:,}")
    print(f"    ±1 day tolerance          : {n_p2_tol1:,}")
    print(f"    npm : {unique_pairs_p2[unique_pairs_p2.ecosystem=='npm'].shape[0]:,}")
    print(f"    pypi: {unique_pairs_p2[unique_pairs_p2.ecosystem=='pypi'].shape[0]:,}")
    print(f"  Total rows in join : {len(m2):,}")

    # ── pass 3: keep remaining unmatched npm/pypi pairs (left join) ───────────

    matched_keys_p1_p2 = matched_keys_p1 | set(zip(
        unique_pairs_p2["ecosystem"], unique_pairs_p2["package_name"], unique_pairs_p2["package_version"]
    ))
    combined["_key"] = list(zip(combined["ecosystem"], combined["package_name"], combined["package_version"]))
    m3 = combined[~combined["_key"].isin(matched_keys_p1_p2)].drop(columns=["_key"]).copy()
    combined = combined.drop(columns=["_key"])
    m3["match_method"] = "no_match"

    # ── combine pass 1 + pass 2 + pass 3 ──────────────────────────────────────

    output = pd.concat([m1, m2, m3], ignore_index=True)

    # clean up helper columns added during matching
    output = output.drop(columns=["version_normalized", "norm_rule", "published_date",
                                   "day_diff", "release_date_dt", "published_date_dt"],
                         errors="ignore")

    # ── summary ────────────────────────────────────────────────────────────────

    total_matched_pairs = n_p1 + n_p2
    unmatched_pairs = total_pairs - total_matched_pairs

    print("\n═══════════════════════════════════════════════════════════")
    print("SUMMARY  (npm/pypi releases Oct 6 2023 – Oct 6 2025)")
    print("═══════════════════════════════════════════════════════════")
    print(f"Total (package, version) pairs   : {total_pairs:>8,}")
    print(f"  Pass 1 – version match         : {n_p1:>8,}  ({n_p1/total_pairs*100:.1f}%)")
    print(f"  Pass 2 – date fallback (exact) : {n_p2_exact:>8,}  ({n_p2_exact/total_pairs*100:.1f}%)")
    print(f"  Pass 2 – date fallback (±1 day): {n_p2_tol1:>8,}  ({n_p2_tol1/total_pairs*100:.1f}%)")
    print(f"  Total matched pairs            : {total_matched_pairs:>8,}  ({total_matched_pairs/total_pairs*100:.1f}%)")
    print(f"  Unmatched pairs                : {unmatched_pairs:>8,}  ({unmatched_pairs/total_pairs*100:.1f}%)")
    print(f"Output rows (one per ecosystem/package/version) : {len(output):>8,}")
    print("═══════════════════════════════════════════════════════════")

    # unmatched breakdown
    pkg_not_in_sc = (~m3["package_name"].isin(scorecard["package_name"].unique())).sum()
    print(f"\nUnmatched breakdown:")
    print(f"  Package name not in scorecard    : {pkg_not_in_sc:,}")
    print(f"  Version/tag mismatch (pkg in sc) : {unmatched_pairs - pkg_not_in_sc:,}")

    output.to_csv(MATCHED_OUT_CSV, index=False)
    print(f"\nSaved matched dataset → {MATCHED_OUT_CSV}")
    print(f"  Rows: {len(output):,}  |  Columns: {len(output.columns)}")
    print(f"  Columns: {', '.join(output.columns.tolist())}")

    return output


def extract_unique_inputs(matched: pd.DataFrame):
    print("\n── Extracting unique inputs for downstream collectors ──")

    sc = matched[matched['match_method'] != 'no_match'].copy()
    print(f"  rows with scorecard match: {len(sc):,}")

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


def main():
    matched = match_scorecard()
    extract_unique_inputs(matched)


if __name__ == '__main__':
    main()
