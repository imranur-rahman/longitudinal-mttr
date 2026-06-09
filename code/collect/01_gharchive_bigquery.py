"""
Step 1: Collect historical GitHub metrics (stars, forks, commits, contributors)
from the GitHub Archive public dataset in BigQuery.

Two queries are run to minimize bytes scanned:
  Query A — stars & forks   (WatchEvent + ForkEvent, NO payload read)
  Query B — commits & contributors (PushEvent only, payload needed for distinct_size)

Prerequisites:
  - Authenticate: gcloud auth application-default login
  - BigQuery API enabled in your GCP project
  - Set GCP_PROJECT env var (or hardcode below)

Usage:
  python 01_gharchive_bigquery.py             # full run
  python 01_gharchive_bigquery.py --dry-run   # estimate cost, no data written

Output: data/collected/github_metrics.csv
  Columns: repo_name, published_at, stars, forks, commits_approx, contributors_approx
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound
    from google.auth.exceptions import RefreshError, TransportError
except ImportError:
    sys.exit("Install: pip install google-cloud-bigquery google-cloud-bigquery-storage pyarrow")

DATA_DIR = Path("data")
OUT_DIR  = DATA_DIR / "collected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
BQ_DATASET  = "longitudinal_mttr"
TEMP_TABLE  = f"{GCP_PROJECT}.{BQ_DATASET}.release_dates_tmp"

GH_START = "2011"
GH_END   = "2025"

UPLOAD_SCHEMA = [
    bigquery.SchemaField("repo_name",    "STRING"),
    bigquery.SchemaField("published_at", "TIMESTAMP"),
]

# ── Query A: stars & forks ────────────────────────────────────────────────────
# Reads only repo.name, type, created_at  — NO payload column.
# WatchEvent in GH Archive always represents a star ('started'); no payload
# filter needed.  ForkEvent also needs no payload.
SQL_STARS_FORKS = f"""
WITH sf_events AS (
  SELECT
    repo.name        AS repo_name,
    type,
    DATE(created_at) AS event_date
  FROM `githubarchive.year.*`
  WHERE _TABLE_SUFFIX BETWEEN '{GH_START}' AND '{GH_END}'
    AND repo.name IN (SELECT DISTINCT repo_name FROM `{{TEMP_TABLE}}`)
    AND type IN ('WatchEvent', 'ForkEvent')
),

daily_sf AS (
  SELECT
    repo_name,
    event_date,
    COUNTIF(type = 'WatchEvent') AS d_stars,
    COUNTIF(type = 'ForkEvent')  AS d_forks
  FROM sf_events
  GROUP BY repo_name, event_date
),

cumul_sf AS (
  SELECT
    repo_name,
    event_date,
    SUM(d_stars) OVER (PARTITION BY repo_name ORDER BY event_date
                       ROWS UNBOUNDED PRECEDING) AS cumul_stars,
    SUM(d_forks) OVER (PARTITION BY repo_name ORDER BY event_date
                       ROWS UNBOUNDED PRECEDING) AS cumul_forks
  FROM daily_sf
),

rd AS (
  SELECT repo_name, CAST(published_at AS DATE) AS release_date, published_at
  FROM `{{TEMP_TABLE}}`
)

SELECT
  rd.repo_name,
  rd.published_at,
  COALESCE(MAX(csf.cumul_stars), 0) AS stars,
  COALESCE(MAX(csf.cumul_forks), 0) AS forks
FROM rd
LEFT JOIN cumul_sf csf
  ON csf.repo_name = rd.repo_name
 AND csf.event_date <= rd.release_date
GROUP BY rd.repo_name, rd.published_at
"""

# ── Query B: commits & contributors ──────────────────────────────────────────
# Reads payload, but ONLY for PushEvent rows (type filter pushed down in BQ).
# distinct_size in the PushEvent payload counts new commits added by the push.
SQL_COMMITS_CONTRIBUTORS = f"""
WITH push_events AS (
  SELECT
    repo.name        AS repo_name,
    actor.login      AS actor_login,
    DATE(created_at) AS event_date,
    SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.distinct_size') AS INT64) AS n_commits
  FROM `githubarchive.year.*`
  WHERE _TABLE_SUFFIX BETWEEN '{GH_START}' AND '{GH_END}'
    AND repo.name IN (SELECT DISTINCT repo_name FROM `{{TEMP_TABLE}}`)
    AND type = 'PushEvent'
),

-- Running cumulative commit count per repo
daily_commits AS (
  SELECT repo_name, event_date,
    SUM(COALESCE(n_commits, 0)) AS d_commits
  FROM push_events
  GROUP BY repo_name, event_date
),
cumul_commits AS (
  SELECT repo_name, event_date,
    SUM(d_commits) OVER (PARTITION BY repo_name ORDER BY event_date
                         ROWS UNBOUNDED PRECEDING) AS cumul_commits
  FROM daily_commits
),

-- First time each contributor pushed to each repo
first_push AS (
  SELECT repo_name, actor_login,
    MIN(event_date) AS first_seen
  FROM push_events
  WHERE actor_login IS NOT NULL AND actor_login != ''
  GROUP BY repo_name, actor_login
),

rd AS (
  SELECT repo_name, CAST(published_at AS DATE) AS release_date, published_at
  FROM `{{TEMP_TABLE}}`
)

SELECT
  rd.repo_name,
  rd.published_at,
  COALESCE(MAX(cc.cumul_commits), 0) AS commits_approx,
  COUNT(DISTINCT
    CASE WHEN fp.first_seen <= rd.release_date THEN fp.actor_login END
  ) AS contributors_approx
FROM rd
LEFT JOIN cumul_commits cc
  ON cc.repo_name = rd.repo_name
 AND cc.event_date <= rd.release_date
LEFT JOIN first_push fp
  ON fp.repo_name = rd.repo_name
GROUP BY rd.repo_name, rd.published_at
"""


def fmt_bytes(b: int) -> str:
    if b >= 1e12:
        return f"{b/1e12:.2f} TB"
    if b >= 1e9:
        return f"{b/1e9:.2f} GB"
    return f"{b/1e6:.2f} MB"


def dry_run_cost(client: bigquery.Client, label: str, sql: str) -> int:
    """Run a dry-run query and print the estimated bytes/cost. Returns bytes."""
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(sql, job_config=cfg, project=GCP_PROJECT)
    b = job.total_bytes_processed
    free_tb = 1e12  # 1 TB free tier per month
    billable = max(0, b - free_tb)
    cost = billable / 1e12 * 5.0
    print(f"  {label}")
    print(f"    Bytes to scan : {fmt_bytes(b)}")
    print(f"    Estimated cost: ${cost:.2f}  (after 1 TB/month free tier)")
    return b


def upload_release_dates(client: bigquery.Client, df: pd.DataFrame) -> None:
    dataset_ref = bigquery.DatasetReference(client.project, BQ_DATASET)
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        client.create_dataset(bigquery.Dataset(dataset_ref))
        print(f"  Created dataset {BQ_DATASET}")

    df = df.copy()
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    job_cfg = bigquery.LoadJobConfig(
        schema=UPLOAD_SCHEMA,
        write_disposition="WRITE_TRUNCATE",
    )
    job = client.load_table_from_dataframe(df, TEMP_TABLE, job_config=job_cfg)
    job.result()
    print(f"  Uploaded {len(df):,} rows to {TEMP_TABLE}")


def run_query(client: bigquery.Client, label: str, sql: str) -> pd.DataFrame:
    print(f"Running {label} …")
    job = client.query(sql, project=GCP_PROJECT)
    result = job.result()
    billed = job.total_bytes_billed or 0
    df = result.to_dataframe()
    print(f"  Done: {len(df):,} rows, {fmt_bytes(billed)} billed")
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect GitHub metrics from GH Archive via BigQuery.")
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="Estimate bytes scanned and cost for both queries without running them.",
    )
    args = parser.parse_args()

    if not GCP_PROJECT:
        sys.exit("Set the GCP_PROJECT environment variable to your Google Cloud project ID.")

    input_csv = OUT_DIR / "unique_repos_dates.csv"
    if not input_csv.exists():
        sys.exit(f"Run 00_preprocess.py first — {input_csv} not found.")

    output_csv = OUT_DIR / "github_metrics.csv"
    if not args.dry_run and output_csv.exists():
        print(f"{output_csv} already exists. Delete it to re-run.")
        return

    print("Loading unique_repos_dates.csv …")
    rd = pd.read_csv(input_csv)
    print(f"  {len(rd):,} (repo, date) pairs across {rd['repo_name'].nunique():,} unique repos")

    client = bigquery.Client(project=GCP_PROJECT)

    # Substitute temp table name into SQL templates
    sql_a = SQL_STARS_FORKS.replace("{TEMP_TABLE}", TEMP_TABLE)
    sql_b = SQL_COMMITS_CONTRIBUTORS.replace("{TEMP_TABLE}", TEMP_TABLE)

    # ── Dry-run mode ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n── Dry-run cost estimate ────────────────────────────────────────")
        print("Note: the temp table must exist for BQ to validate the queries.")
        print("Uploading release dates to temp table first …")
        upload_release_dates(client, rd)

        b_a = dry_run_cost(client, "Query A  (stars & forks,  NO payload)", sql_a)
        b_b = dry_run_cost(client, "Query B  (commits & contributors, payload PushEvent only)", sql_b)

        total = b_a + b_b
        free_tb = 1e12
        total_cost = max(0, total - free_tb) / 1e12 * 5.0
        print(f"\n  ── Total ──")
        print(f"    Combined scan : {fmt_bytes(total)}")
        print(f"    Combined cost : ${total_cost:.2f}  (assuming free tier applies once)")

        client.delete_table(TEMP_TABLE, not_found_ok=True)
        return

    # ── Full run ──────────────────────────────────────────────────────────────
    print(f"\nUploading release dates to {TEMP_TABLE} …")
    upload_release_dates(client, rd)

    df_a = run_query(client, "Query A — stars & forks", sql_a)
    df_b = run_query(client, "Query B — commits & contributors", sql_b)

    # Merge the two results on (repo_name, published_at)
    # Both have the same row set (release_dates cross-joined), so a merge is safe.
    result = df_a.merge(df_b, on=["repo_name", "published_at"], how="outer")
    result[["stars", "forks", "commits_approx", "contributors_approx"]] = (
        result[["stars", "forks", "commits_approx", "contributors_approx"]].fillna(0).astype(int)
    )

    result.to_csv(output_csv, index=False)
    print(f"\nSaved {len(result):,} rows to {output_csv}")

    client.delete_table(TEMP_TABLE, not_found_ok=True)
    print("Deleted temp table.")


if __name__ == '__main__':
    try:
        main()
    except (RefreshError, TransportError):
        sys.exit(
            "GCP credentials are missing or expired.\n"
            "Fix: gcloud auth application-default login"
        )
