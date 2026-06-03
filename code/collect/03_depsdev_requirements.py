"""
Step 3: Collect pin/float counts from deps.dev GetRequirements API.

For each (ecosystem, package_name, package_version), fetches the dependency
requirements and counts how many are pinned vs. floated.

Definition:
  pinned  — dependency locked to an exact version (no range operators)
  floated — dependency using a version range or wildcard (default)

deps.dev has no rate limiting; uses 80 concurrent async requests.
Results are checkpointed to disk every 500 rows.

Input:  data/collected/unique_package_versions.csv
Output: data/collected/depsdev_requirements.csv
"""

import asyncio
import csv
import json
import sys
import urllib.parse
from pathlib import Path

import pandas as pd

try:
    import aiohttp
except ImportError:
    sys.exit("Install aiohttp: pip install aiohttp")

from utils_api import count_pins_floats, load_checkpoint, save_checkpoint, system_names

DATA_DIR    = Path("data")
OUT_DIR     = DATA_DIR / "collected"
CACHE_DIR   = OUT_DIR / "cache" / "depsdev_requirements"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT  = OUT_DIR / "depsdev_requirements_checkpoint.json"
OUTPUT_CSV  = OUT_DIR / "depsdev_requirements.csv"

# Internal deps.dev requirements endpoint (confirmed working)
BASE_URL    = "https://deps.dev/_/s/{system}/p/{name}/v/{version}/requirements"
CONCURRENCY = 80
BATCH_SIZE  = 500


def cache_path(ecosystem: str, package_name: str, version: str) -> Path:
    safe_name = urllib.parse.quote(package_name, safe='')
    safe_ver  = urllib.parse.quote(version, safe='')
    return CACHE_DIR / ecosystem / safe_name / f"{safe_ver}.json"


async def fetch_requirements(session: aiohttp.ClientSession,
                              ecosystem: str,
                              package_name: str,
                              version: str) -> dict | None:
    """Fetch requirements JSON. Returns None on 404 or error."""
    path = cache_path(ecosystem, package_name, version)
    if path.exists():
        with open(path) as f:
            return json.load(f)

    _, lower_sys = system_names(ecosystem)
    encoded_name = urllib.parse.quote(package_name, safe='')
    encoded_ver  = urllib.parse.quote(version, safe='')
    url = BASE_URL.format(system=lower_sys, name=encoded_name, version=encoded_ver)

    try:
        async with session.get(url) as resp:
            if resp.status == 404:
                payload = {}
            else:
                resp.raise_for_status()
                payload = await resp.json(content_type=None)
    except Exception as exc:
        print(f"  WARN fetch {ecosystem}/{package_name}@{version}: {exc}")
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f)

    return payload


def open_output_csv(path: Path) -> tuple:
    """Open output CSV in append mode; return (file_handle, csv_writer)."""
    new_file = not path.exists()
    fh = open(path, 'a', newline='')
    writer = csv.DictWriter(fh, fieldnames=[
        'ecosystem', 'package_name', 'package_version', 'pins', 'floats'
    ])
    if new_file:
        writer.writeheader()
    return fh, writer


async def process_all(df: pd.DataFrame, completed: set) -> None:
    sem     = asyncio.Semaphore(CONCURRENCY)
    pending = []  # buffer for batch writes

    fh, writer = open_output_csv(OUTPUT_CSV)
    total = len(df)

    async def handle_one(row):
        key = f"{row.ecosystem}|{row.package_name}|{row.package_version}"
        if key in completed:
            return None

        async with sem:
            payload = await fetch_requirements(
                session, row.ecosystem, row.package_name, row.package_version
            )

        if payload is None:
            completed.add(key)
            return None

        pins, floats = count_pins_floats(row.ecosystem, payload)
        completed.add(key)
        return {
            'ecosystem':       row.ecosystem,
            'package_name':    row.package_name,
            'package_version': row.package_version,
            'pins':            pins,
            'floats':          floats,
        }

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [handle_one(row) for row in df.itertuples(index=False)]
        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            done += 1
            if result:
                pending.append(result)

            if len(pending) >= BATCH_SIZE:
                writer.writerows(pending)
                fh.flush()
                pending.clear()
                save_checkpoint(CHECKPOINT, completed)

            if done % 2000 == 0:
                print(f"  {done:,}/{total:,} processed …")

    # Write remaining
    if pending:
        writer.writerows(pending)
        fh.flush()
    fh.close()
    save_checkpoint(CHECKPOINT, completed)


def main():
    input_csv = OUT_DIR / "unique_package_versions.csv"
    if not input_csv.exists():
        sys.exit(f"Run 00_preprocess.py first — {input_csv} not found.")

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} unique (ecosystem, package, version) rows")

    completed = load_checkpoint(CHECKPOINT)
    print(f"Resuming: {len(completed):,} already done, "
          f"{len(df) - len(completed):,} remaining")

    print(f"Fetching with {CONCURRENCY} concurrent requests …")
    asyncio.run(process_all(df, completed))

    result_df = pd.read_csv(OUTPUT_CSV)
    print(f"\nDone. {len(result_df):,} rows saved to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
