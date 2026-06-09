# longitudinal-mttr

Run order
  
# From project root, with .venv activated
python code/collect/00_preprocess.py          # ~30 sec

# Set your GCP project, then:
# Check cost before committing (recommended first step)
export GCP_PROJECT=your-project-id
gcloud auth application-default login
python code/collect/01_gharchive_bigquery.py --dry-run

# Full run
python code/collect/01_gharchive_bigquery.py

# These two can run in parallel (no dependency on BQ):
python code/collect/02_depsdev_releases.py    # ~5–15 min (208K unique packages, cached)
python code/collect/03_depsdev_requirements.py # ~30–60 min (208K versions, checkpointed)

python code/collect/04_merge.py               # <1 min

Scripts 02 and 03 are resumable — if interrupted, re-running picks up from the checkpoint. Script 01 (BigQuery) outputs to
data/collected/github_metrics.csv and skips if that file already exists.
