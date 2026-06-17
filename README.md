# longitudinal-mttr

A longitudinal study of how fast open-source npm and PyPI packages remediate
vulnerable dependencies, and which repository characteristics predict faster
remediation. Uses OpenSSF Scorecard, deps.dev, and GitHub Archive data collected
over a two-year window (October 2023 – October 2025).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Getting the Code](#2-getting-the-code)
3. [Prerequisites](#3-prerequisites)
4. [Setting Up Python](#4-setting-up-python)
5. [Input Data](#5-input-data)
6. [Run the Pipeline](#6-run-the-pipeline)
7. [Script Reference](#7-script-reference)
8. [Output Files](#8-output-files)
9. [Key Concepts](#9-key-concepts)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Overview

**Research question:** Does adopting a Software Composition Analysis (SCA) tool
reduce the time to remediate known vulnerable dependencies? More broadly, which
package and repository characteristics predict faster remediation?

**Outcome variable:** MTTR (Mean Time to Remediate) — the number of days between a
package version's release and the first subsequent release that resolves all open
vulnerable dependencies. MTTR = 0.0 means the version had no vulnerable dependencies
in its observation window (not exposed, not "fast").

**Sample (after filtering to packages with any MTTR > 0):**
70,953 package-version rows · 1,867 packages · 1,839 GitHub repos · npm (66%) + PyPI (34%)

**Analyses:** Staggered Difference-in-Differences (Callaway & Sant'Anna 2021),
Panel Fixed-Effects OLS, and GPS-based continuous Dose-Response (Hirano & Imbens 2004).

---

## 2. Getting the Code

> **You do not need git.** Download the repository as a ZIP file:
>
> 1. Go to the repository page on GitHub.
> 2. Click the green **Code** button near the top right.
> 3. Select **Download ZIP**.
> 4. Unzip the downloaded file to a folder of your choice (e.g. `longitudinal-mttr/`).
> 5. Open a terminal and `cd` into that folder before running any commands below.

---

## 3. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 or newer | Download from https://www.python.org/downloads/ |
| pip | bundled with Python | Used to install packages |
| Google Cloud SDK | latest | **Only** needed for Step 1 (BigQuery). Install from https://cloud.google.com/sdk/docs/install |

**Check your Python version:**
```bash
python3 --version
```

---

## 4. Setting Up Python

Run these commands once from the project root folder:

```bash
# Create a virtual environment (keeps dependencies isolated)
python3 -m venv .venv

# Activate it — choose the line for your operating system:
source .venv/bin/activate          # Mac / Linux
.venv\Scripts\activate             # Windows (Command Prompt)
.venv\Scripts\Activate.ps1         # Windows (PowerShell)

# Install all required packages
pip install -r requirements.txt
```

> **Important:** You must activate the virtual environment (`source .venv/bin/activate`)
> every time you open a new terminal before running any of the project's scripts.
> Your prompt will show `(.venv)` when it is active.

---

## 5. Input Data

The pipeline runs in two phases. The raw input files are **not included** in the
repository download (they are large and contain external data).

### Phase A — Data Collection (Steps 0–4)

These scripts require raw input files that must be obtained separately:

| File | Purpose | Location in project |
|---|---|---|
| OpenSSF Scorecard data | Security scores for GitHub repos | `data/scorecard/scorecard_local_metrics_final.csv` |
| Dependency-Update-Tool scores | Pre-computed SCA detection scores | `data/dependency_update_tool/package_data_final_with_scores_dut_fixed.csv` |
| npm per-release MTTR data | MTTR metrics for npm packages | `data/npm/npm_data_depsdev_per_release_results.csv` |
| PyPI per-release MTTR data | MTTR metrics for PyPI packages | `data/pypi/pypi_data_depsdev_per_release_results.csv` |

Contact the authors to obtain these files, then place them in the locations above
before running Step 0.

### Phase B — Causal Analyses (Steps 5–6)

These scripts only need `data/scorecard_with_longitudinal_metrics.csv`, which is the
fully enriched dataset produced at the end of Phase A. **If you already have this
file** (shared by the authors), you can skip Steps 0–4 entirely and jump straight
to Step 5.

---

## 6. Run the Pipeline

Make sure your virtual environment is active (`source .venv/bin/activate`) before
running any script.

---

### Step 0 — Preprocess and Match (~30 seconds)

Matches npm/PyPI release records to OpenSSF Scorecard entries. Produces the core
dataset and extracts the list of unique repos/versions for downstream API queries.

```bash
python code/collect/00_preprocess.py
```

**Output:**
- `data/matched_scorecard_mttr.csv` — all package-version rows joined to Scorecard
- `data/collected/unique_repos_dates.csv` — input for Step 1
- `data/collected/unique_package_versions.csv` — input for Steps 2 & 3

---

### Step 1 — GitHub Metrics via BigQuery (~5–10 minutes, requires GCP)

Queries Google BigQuery's GH Archive dataset for stars, forks, commits, and
contributor counts per repo at each release date.

**One-time GCP setup (do this before running):**
```bash
# Install Google Cloud SDK from https://cloud.google.com/sdk/docs/install, then:
gcloud auth application-default login   # Opens a browser to sign in
export GCP_PROJECT=your-gcp-project-id # Replace with your actual project ID
```

```bash
# Always check the estimated cost before running (no charge for dry run):
python code/collect/01_gharchive_bigquery.py --dry-run

# When ready, run for real:
python code/collect/01_gharchive_bigquery.py
```

**Output:** `data/collected/github_metrics.csv`

> Note: If `data/collected/github_metrics.csv` already exists, the script skips
> the query. Delete it to force a re-run.

---

### Steps 2 & 3 — deps.dev Metrics (can run in parallel)

These two scripts query the deps.dev API for per-release statistics. They are
**resumable** — if interrupted, re-running picks up from the last checkpoint.

```bash
# Step 2: Release counts and release frequency (~5–15 minutes)
python code/collect/02_depsdev_releases.py

# Step 3: Pinned and floating dependency counts (~30–60 minutes)
python code/collect/03_depsdev_requirements.py
```

**Output:**
- `data/collected/depsdev_releases.csv` — `version_releases`, `release_frequency`
- `data/collected/depsdev_requirements.csv` — `pins`, `floats` per version

> Steps 2 and 3 have no dependency on each other. You can open two terminals and
> run them at the same time to save time. Both cache responses locally in
> `data/collected/cache/` so repeated runs are much faster.

---

### Step 4 — Merge All Metrics (~1 minute)

Joins the GitHub metrics and deps.dev metrics back onto the matched Scorecard data.
Produces the final enriched dataset used by all downstream analyses.

```bash
python code/collect/04_merge.py
```

**Output:** `data/scorecard_with_longitudinal_metrics.csv` (52 columns)

---

### Step 5 — Export Clean Analysis Dataset (~1 minute)

Produces a trimmed, human-readable version of the enriched dataset, keeping only
the 18 columns used in the causal analyses (outcome + 9 covariates + treatment +
identifiers). All internal metadata and unused Scorecard checks are dropped.

```bash
python code/collect/05_export_readable.py
```

**Output:**
- `data/analysis_dataset.csv` — 18 clearly-named columns, matched rows only
- `data/analysis_dataset_columns.csv` — column dictionary (name, original, description)

---

### Step 6 — Causal Analyses

Run in order within each subfolder.

#### Difference-in-Differences & Panel Fixed Effects

```bash
python code/did_analysis/00_prepare_panel.py       # ~30 sec — builds panel.csv
python code/did_analysis/01_sca_staggered_did.py   # ~2–3 min — staggered DiD
python code/did_analysis/02_panel_fe_regressions.py # ~5 min — panel OLS
```

**Output:** `data/did_analysis/` — CSV results + event-study plot

#### GPS Dose-Response

```bash
python code/dose_response/00_prepare_data.py        # ~30 sec
python code/dose_response/01_gps_dose_response.py   # ~5–8 min — 200 bootstrap reps × 8 vars
```

**Output:** `data/dose_response/` — 8 ADRF curves (CSV + PNG) + summary table

---

### Step 7 — Presentation Materials

```bash
python code/professor_summary/00_generate_summary.py   # ~30 sec
```

**Output:** `data/professor_summary/` — 6 slide-ready figures + `PROBLEM_STATEMENT.md`

---

### Step 8 — Descriptive Analysis Scripts (optional)

These were used for initial exploration and are independent of the causal analyses.
They read directly from the raw npm/PyPI per-release files. Run in any order.

```bash
python code/01_data_overview.py
python code/02_ecosystem_trends.py
python code/03_distribution_evolution.py
python code/04_mttu_vs_mttr.py
python code/05_cohort_analysis.py
python code/06_within_package_trends.py
python code/07_mttr_package_classification.py
python code/08_top10_trajectories.py
python code/09_top30_trajectories_by_class.py
```

**Output:** `figures/` — PNG plots

---

## 7. Script Reference

### Data Collection (`code/collect/`)

| Script | Purpose | Key Inputs | Key Outputs | Time | Resumable |
|---|---|---|---|---|---|
| `00_preprocess.py` | Match releases to Scorecard; filter to packages with any MTTR > 0 | npm/pypi CSVs, scorecard CSV, DUT CSV | `matched_scorecard_mttr.csv`, `unique_repos_dates.csv`, `unique_package_versions.csv` | ~30 s | No |
| `01_gharchive_bigquery.py` | GitHub stars/forks/commits/contributors via BigQuery | `unique_repos_dates.csv`, GCP credentials | `github_metrics.csv` | ~5–10 min | No (skips if output exists) |
| `02_depsdev_releases.py` | Release count & frequency per package via deps.dev | `unique_package_versions.csv` | `depsdev_releases.csv` | ~5–15 min | Yes |
| `03_depsdev_requirements.py` | Pinned & floating dependency counts per version | `unique_package_versions.csv` | `depsdev_requirements.csv` | ~30–60 min | Yes |
| `04_merge.py` | Join all metrics onto matched Scorecard data | Outputs of 00–03 | `scorecard_with_longitudinal_metrics.csv` | ~1 min | No |
| `05_export_readable.py` | Export clean 18-column analysis dataset | `scorecard_with_longitudinal_metrics.csv` | `analysis_dataset.csv`, `analysis_dataset_columns.csv` | ~1 min | No |

### Causal Analyses (`code/did_analysis/`, `code/dose_response/`)

| Script | Purpose | Key Inputs | Key Outputs | Time |
|---|---|---|---|---|
| `did_analysis/00_prepare_panel.py` | Build release-sequence panel with renamed columns | `scorecard_with_longitudinal_metrics.csv` | `did_analysis/panel.csv` | ~30 s |
| `did_analysis/01_sca_staggered_did.py` | Staggered DiD: effect of SCA adoption on MTTR | `panel.csv` | `sca_did_simple_att.csv`, `sca_did_event_study.csv`, `sca_did_event_study.png` | ~2–3 min |
| `did_analysis/02_panel_fe_regressions.py` | Panel FE OLS: mttr ~ each covariate | `panel.csv` | `panel_fe_results.csv` | ~5 min |
| `dose_response/00_prepare_data.py` | Filter & log-transform for GPS | `scorecard_with_longitudinal_metrics.csv` | `dose_response/data.csv` | ~30 s |
| `dose_response/01_gps_dose_response.py` | GPS dose-response curves (8 variables) | `dose_response/data.csv` | 8 × `{var}_adrf.{csv,png}`, `gps_dose_response_summary.csv` | ~5–8 min |

### Presentation (`code/professor_summary/`)

| Script | Purpose | Key Inputs | Key Outputs | Time |
|---|---|---|---|---|
| `00_generate_summary.py` | Slide-ready figures + problem statement | `scorecard_with_longitudinal_metrics.csv` | `fig1`–`fig6` PNGs, `PROBLEM_STATEMENT.md` | ~30 s |

### Descriptive Analysis (`code/*.py`)

| Script | Purpose | Input | Output |
|---|---|---|---|
| `01_data_overview.py` | Summary statistics, release trends | pypi per-release CSV | `figures/01_*.png` |
| `02_ecosystem_trends.py` | Yearly/quarterly MTTR trends + Mann-Kendall test | pypi per-release CSV | `figures/02_*.png` |
| `03_distribution_evolution.py` | MTTR distribution by year, effect sizes | pypi per-release CSV | `figures/03_*.png` |
| `04_mttu_vs_mttr.py` | MTTU vs MTTR correlation | pypi per-release CSV | `figures/04_*.png` |
| `05_cohort_analysis.py` | MTTR by package cohort (release year) | pypi per-release CSV | `figures/05_*.png` |
| `06_within_package_trends.py` | Per-package trajectory classification | pypi per-release CSV | `figures/06_*.png`, top-50 CSVs |
| `07_mttr_package_classification.py` | MTTR-only classification (5 classes) | pypi per-release CSV | `figures/07_*.png`, class summary CSV |
| `08_top10_trajectories.py` | Top-10 improving/worsening packages | pypi per-release CSV | `figures/08_*.png` |
| `09_top30_trajectories_by_class.py` | Top-30 per class across npm + PyPI | npm + pypi per-release CSVs | `figures/09_*.png` |

---

## 8. Output Files

After running the full pipeline, key outputs are:

```
data/
  analysis_dataset.csv              # Clean 18-column dataset (main analysis input)
  analysis_dataset_columns.csv      # Column dictionary
  scorecard_with_longitudinal_metrics.csv  # Full 52-column enriched dataset
  did_analysis/
    sca_did_simple_att.csv          # Overall ATT: 0.555 days (not significant)
    sca_did_event_study.csv         # ATT by release-sequence relative to SCA adoption
    sca_did_event_study.png         # Event-study plot
    panel_fe_results.csv            # Panel FE coefficients + p-values for 10 covariates
    RESULTS.md                      # Interpretation and statistical summary
  dose_response/
    gps_dose_response_summary.csv   # ADRF endpoint comparison for 8 variables
    {variable}_adrf.csv / .png      # Dose-response curve per variable (8 total)
    RESULTS.md                      # Interpretation
  professor_summary/
    PROBLEM_STATEMENT.md            # 1-page problem framing for methodological review
    fig1_mttr_distribution.png      # Zero-inflation and nonzero distribution
    fig2_panel_structure.png        # Releases-per-repo + spaghetti trajectories
    fig3_covariate_summary.png      # Box plots of 8 continuous covariates
    fig4_sca_cohorts.png            # SCA adoption cohort breakdown
    fig5_mttr_by_severity.png       # MTTR by vulnerability severity
    fig6_mttr_violin.png            # Horizontal violin: MTTR by ecosystem
figures/
  01_*.png … 09_*.png               # Descriptive analysis plots
```

---

## 9. Key Concepts

### MTTR (Mean Time to Remediate)
The number of **days** between the release of a package version and the first
subsequent release that resolves all known vulnerable dependencies. Measured
per package version.

- **MTTR = 0.0** — the package version had no vulnerable dependencies during its
  observation window. This does *not* mean remediation was fast; it means the version
  was never exposed to a known vulnerability.
- **MTTR > 0** — the version had at least one vulnerable dependency for some period;
  MTTR is how many days it took to ship a fix.

### OpenSSF Scorecard
An automated tool that scores a GitHub repository's security practices on a 0–10
scale across 10 checks (e.g., CI tests, code review, pinned dependencies, SCA tool
usage). Scores are collected once per repo-tag and joined to the matching package
version release.

### SCA Tool (Software Composition Analysis)
A tool that automatically opens pull requests to update vulnerable dependencies
(e.g., Dependabot, Renovate Bot, Snyk). Detected by the Scorecard
`Dependency-Update-Tool` check: score = 10 means a tool was found, 0 means none.

### Scorecard Match Method
Each package version is matched to a Scorecard entry using:
- `version_match` — Scorecard git tag normalized to match the package version string
- `date_fallback` — closest Scorecard run date within ±1 day of the release
- `no_match` — no Scorecard entry found (these rows have no GitHub-derived covariates)

---

## 10. Troubleshooting

**`ModuleNotFoundError: No module named 'X'`**
Your virtual environment is not active. Run `source .venv/bin/activate` (Mac/Linux)
or `.venv\Scripts\activate` (Windows) and try again.

**`google.auth.exceptions.DefaultCredentialsError`**
You haven't authenticated with Google Cloud. Run `gcloud auth application-default login`
and make sure `GCP_PROJECT` is set: `export GCP_PROJECT=your-project-id`.

**BigQuery query costs money**
Always run `python code/collect/01_gharchive_bigquery.py --dry-run` first. BigQuery
bills by bytes scanned, not by result rows. The dry run prints the estimated scan
size without executing the query.

**Script 02 or 03 is very slow / I want to stop and resume**
Both scripts save a checkpoint every 500 packages. Press `Ctrl+C` to stop safely,
then re-run the same script — it will skip already-processed entries and continue
from where it left off.

**deps.dev or BigQuery returns errors for some packages**
Partial failures are normal (some packages are deleted or private). The scripts log
errors and continue. Check the `data_status` / `data_error` columns in the output.

**`FileNotFoundError` when running analysis scripts**
You are likely missing `data/scorecard_with_longitudinal_metrics.csv`. Either run
the full pipeline (Steps 0–4) or obtain this file from the authors.
