"""
Step 2: Collect release frequency metrics from deps.dev GetPackage API.

For each unique (ecosystem, package_name), fetches the full version list once,
caches the response, then computes per (ecosystem, package_name, package_version):
  - version_releases   : total versions published on or before this version's release date
  - release_frequency  : versions published in the 2-year window ending at this version's date

deps.dev has no rate limiting; uses 80 concurrent async requests.

Input:  data/collected/unique_package_versions.csv
Output: data/collected/depsdev_releases.csv
"""

import asyncio
import json
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import aiohttp
except ImportError:
    sys.exit("Install aiohttp: pip install aiohttp")

from utils_api import load_checkpoint, save_checkpoint, system_names

DATA_DIR    = Path("data")
OUT_DIR     = DATA_DIR / "collected"
CACHE_DIR   = OUT_DIR / "cache" / "depsdev_package"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT  = OUT_DIR / "depsdev_releases_checkpoint.json"
OUTPUT_CSV  = OUT_DIR / "depsdev_releases.csv"

BASE_URL    = "https://api.deps.dev/v3/systems/{system}/packages/{name}"
CONCURRENCY = 80
TWO_YEARS   = timedelta(days=365 * 2)


def cache_path(ecosystem: str, package_name: str) -> Path:
    safe = urllib.parse.quote(package_name, safe='')
    return CACHE_DIR / ecosystem / f"{safe}.json"


async def fetch_package(session: aiohttp.ClientSession,
                        ecosystem: str, package_name: str) -> list[dict]:
    """Fetch all versions for a package. Returns list of {version, publishedAt}."""
    path = cache_path(ecosystem, package_name)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get('versions', [])

    upper_sys, _ = system_names(ecosystem)
    encoded_name = urllib.parse.quote(package_name, safe='')
    url = BASE_URL.format(system=upper_sys, name=encoded_name)

    try:
        async with session.get(url) as resp:
            if resp.status == 404:
                versions = []
            else:
                resp.raise_for_status()
                body = await resp.json(content_type=None)
                versions = body.get('versions', [])
    except Exception as exc:
        print(f"  WARN fetch {ecosystem}/{package_name}: {exc}")
        versions = []

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'versions': versions}, f)

    return versions


def compute_metrics(versions: list[dict], release_ts: datetime) -> tuple[int, int]:
    """Return (version_releases, release_frequency) relative to release_ts."""
    two_yr_ago = release_ts - TWO_YEARS
    total = 0
    in_window = 0
    for v in versions:
        pub_str = v.get('publishedAt', '')
        if not pub_str:
            continue
        try:
            pub = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
        except ValueError:
            continue
        if pub <= release_ts:
            total += 1
            if pub >= two_yr_ago:
                in_window += 1
    return total, in_window


async def process_all(df: pd.DataFrame, completed: set) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []

    async def handle_one(row, session):
        key = f"{row.ecosystem}|{row.package_name}|{row.package_version}"
        if key in completed:
            return

        async with sem:
            versions = await fetch_package(session, row.ecosystem, row.package_name)

        # Find this version's published_at from the API response
        release_ts = None
        for v in versions:
            if v.get('versionKey', {}).get('version') == row.package_version:
                pub_str = v.get('publishedAt', '')
                if pub_str:
                    try:
                        release_ts = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
                    except ValueError:
                        pass
                break

        # Fall back to the published_at from our CSV if API didn't have it
        if release_ts is None:
            try:
                release_ts = pd.to_datetime(row.published_at, utc=True).to_pydatetime()
            except Exception:
                return

        version_releases, release_frequency = compute_metrics(versions, release_ts)

        results.append({
            'ecosystem':         row.ecosystem,
            'package_name':      row.package_name,
            'package_version':   row.package_version,
            'version_releases':  version_releases,
            'release_frequency': release_frequency,
        })
        completed.add(key)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [handle_one(row, session) for row in df.itertuples(index=False)]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            await coro
            if i % 1000 == 0:
                print(f"  {i:,}/{len(df):,} processed …")
                save_checkpoint(CHECKPOINT, completed)

    return results


def main():
    input_csv = OUT_DIR / "unique_package_versions.csv"
    if not input_csv.exists():
        sys.exit(f"Run 00_preprocess.py first — {input_csv} not found.")

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} unique (ecosystem, package, version) rows")

    completed = load_checkpoint(CHECKPOINT)
    print(f"Resuming: {len(completed):,} already done")

    # If output exists, load existing results to append
    existing_rows = []
    if OUTPUT_CSV.exists():
        existing_rows = pd.read_csv(OUTPUT_CSV).to_dict('records')
        print(f"  {len(existing_rows):,} rows already in output CSV")

    print(f"Fetching with {CONCURRENCY} concurrent requests …")
    new_results = asyncio.run(process_all(df, completed))

    all_results = existing_rows + new_results
    out_df = pd.DataFrame(all_results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    save_checkpoint(CHECKPOINT, completed)
    print(f"\nDone. Saved {len(out_df):,} rows to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
