"""
Step 1: Collect historical GitHub metrics (stars, forks, commits, contributors)
from the GitHub Archive public dataset in BigQuery.

Prerequisites:
  - Google Cloud SDK installed and authenticated:
      gcloud auth application-default login
  - BigQuery API enabled in your GCP project
  - Set GCP_PROJECT environment variable (or edit GCP_PROJECT below)

Cost note:
  GH Archive tables are NOT partitioned by repo, so BigQuery scans the full
  yearly tables for all years in the date range. Expect ~5–20 TB scanned
  (≈$25–$100 at on-demand pricing). Run once and cache the output.

Output: data/collected/github_metrics.csv
  Columns: repo_name, published_at, stars, forks, commits_approx, contributors_approx
"""

import os
import sys
from pathlib import Path

import pandas as pd

try:
    from google.cloud import bigquery
except ImportError:
    sys.exit("Install google-cloud-bigquery: pip install google-cloud-bigquery")

DATA_DIR = Path("data")
OUT_DIR  = DATA_DIR / "collected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")  # set your project ID here or via env var
BQ_DATASET  = "longitudinal_mttr"
TEMP_TABLE  = f"{GCP_PROJECT}.{BQ_DATASET}.release_dates_tmp"

# Limit GH Archive scan to years where our data lives
GH_ARCHIVE_START_YEAR = "2011"
GH_ARCHIVE_END_YEAR   = "2025"


UPLOAD_SCHEMA = [
    bigquery.SchemaField("repo_name",    "STRING"),
    bigquery.SchemaField("published_at", "TIMESTAMP"),
]

SQL = f"""
-- ── 1. Events for our repos ──────────────────────────────────────────────────
WITH repo_events AS (
  SELECT
    repo.name                                          AS repo_name,
    type,
    created_at,
    actor.login                                        AS actor_login,
    JSON_EXTRACT_SCALAR(payload, '$.action')           AS action,
    SAFE_CAST(
      JSON_EXTRACT_SCALAR(payload, '$.distinct_size')
    AS INT64)                                          AS push_distinct_size
  FROM `githubarchive.year.*`
  WHERE _TABLE_SUFFIX BETWEEN '{GH_ARCHIVE_START_YEAR}' AND '{GH_ARCHIVE_END_YEAR}'
    AND repo.name IN (SELECT DISTINCT repo_name FROM `{TEMP_TABLE}`)
    AND type IN ('WatchEvent', 'ForkEvent', 'PushEvent')
),

-- ── 2a. Cumulative daily stars & forks ───────────────────────────────────────
daily_sf AS (
  SELECT
    repo_name,
    DATE(created_at) AS event_date,
    COUNTIF(type = 'WatchEvent' AND action = 'started') AS d_stars,
    COUNTIF(type = 'ForkEvent')                          AS d_forks
  FROM repo_events
  WHERE type IN ('WatchEvent', 'ForkEvent')
  GROUP BY repo_name, event_date
),
cumul_sf AS (
  SELECT
    repo_name,
    event_date,
    SUM(d_stars) OVER (PARTITION BY repo_name ORDER BY event_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumul_stars,
    SUM(d_forks) OVER (PARTITION BY repo_name ORDER BY event_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumul_forks
  FROM daily_sf
),

-- ── 2b. Cumulative daily commits (sum of distinct_size per push) ──────────────
daily_commits AS (
  SELECT
    repo_name,
    DATE(created_at) AS event_date,
    SUM(COALESCE(push_distinct_size, 0)) AS d_commits
  FROM repo_events
  WHERE type = 'PushEvent'
  GROUP BY repo_name, event_date
),
cumul_commits AS (
  SELECT
    repo_name,
    event_date,
    SUM(d_commits) OVER (PARTITION BY repo_name ORDER BY event_date
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumul_commits
  FROM daily_commits
),

-- ── 2c. First-seen date per contributor per repo (for distinct count up to date) ──
first_push AS (
  SELECT
    repo_name,
    actor_login AS contributor,
    MIN(DATE(created_at)) AS first_seen
  FROM repo_events
  WHERE type = 'PushEvent' AND actor_login IS NOT NULL AND actor_login != ''
  GROUP BY repo_name, actor_login
),

-- ── 3. Join release dates with cumulative series (asof join via MAX) ──────────
rd AS (
  SELECT repo_name, CAST(published_at AS DATE) AS release_date, published_at
  FROM `{TEMP_TABLE}`
),

sf_at_release AS (
  SELECT
    rd.repo_name,
    rd.published_at,
    MAX(csf.cumul_stars)   AS stars,
    MAX(csf.cumul_forks)   AS forks
  FROM rd
  LEFT JOIN cumul_sf csf
    ON csf.repo_name = rd.repo_name AND csf.event_date <= rd.release_date
  GROUP BY rd.repo_name, rd.published_at
),

commits_at_release AS (
  SELECT
    rd.repo_name,
    rd.published_at,
    MAX(cc.cumul_commits) AS commits_approx
  FROM rd
  LEFT JOIN cumul_commits cc
    ON cc.repo_name = rd.repo_name AND cc.event_date <= rd.release_date
  GROUP BY rd.repo_name, rd.published_at
),

contributors_at_release AS (
  SELECT
    rd.repo_name,
    rd.published_at,
    COUNT(DISTINCT fp.contributor) AS contributors_approx
  FROM rd
  LEFT JOIN first_push fp
    ON fp.repo_name = rd.repo_name AND fp.first_seen <= rd.release_date
  GROUP BY rd.repo_name, rd.published_at
)

-- ── 4. Final join ─────────────────────────────────────────────────────────────
SELECT
  rd.repo_name,
  rd.published_at,
  COALESCE(sf.stars,   0) AS stars,
  COALESCE(sf.forks,   0) AS forks,
  COALESCE(co.commits_approx,       0) AS commits_approx,
  COALESCE(ct.contributors_approx,  0) AS contributors_approx
FROM rd
LEFT JOIN sf_at_release         sf ON sf.repo_name = rd.repo_name AND sf.published_at = rd.published_at
LEFT JOIN commits_at_release    co ON co.repo_name = rd.repo_name AND co.published_at = rd.published_at
LEFT JOIN contributors_at_release ct ON ct.repo_name = rd.repo_name AND ct.published_at = rd.published_at
"""


def upload_release_dates(client: bigquery.Client, df: pd.DataFrame) -> None:
    dataset_ref = client.dataset(BQ_DATASET)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        client.create_dataset(bigquery.Dataset(dataset_ref))
        print(f"  Created dataset {BQ_DATASET}")

    # Convert published_at to proper timestamp
    df = df.copy()
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)

    job_cfg = bigquery.LoadJobConfig(
        schema=UPLOAD_SCHEMA,
        write_disposition="WRITE_TRUNCATE",
    )
    job = client.load_table_from_dataframe(df, TEMP_TABLE, job_config=job_cfg)
    job.result()
    print(f"  Uploaded {len(df):,} rows to {TEMP_TABLE}")


def run_query(client: bigquery.Client) -> pd.DataFrame:
    print("Running BigQuery query … (this may take several minutes and scan many TB)")
    job = client.query(SQL, project=GCP_PROJECT)
    result = job.result()
    df = result.to_dataframe()
    print(f"  Query complete: {len(df):,} rows returned")
    return df


def main():
    if not GCP_PROJECT:
        sys.exit("Set GCP_PROJECT environment variable to your Google Cloud project ID.")

    input_csv = OUT_DIR / "unique_repos_dates.csv"
    if not input_csv.exists():
        sys.exit(f"Run 00_preprocess.py first — {input_csv} not found.")

    output_csv = OUT_DIR / "github_metrics.csv"
    if output_csv.exists():
        print(f"{output_csv} already exists. Delete it to re-run.")
        return

    print("Loading unique_repos_dates.csv …")
    rd = pd.read_csv(input_csv)
    print(f"  {len(rd):,} (repo, date) pairs, {rd['repo_name'].nunique():,} unique repos")

    client = bigquery.Client(project=GCP_PROJECT)

    print(f"\nUploading release dates to BigQuery temp table {TEMP_TABLE} …")
    upload_release_dates(client, rd)

    result_df = run_query(client)

    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    # Clean up temp table
    client.delete_table(TEMP_TABLE, not_found_ok=True)
    print("Deleted temp table.")


if __name__ == '__main__':
    main()
