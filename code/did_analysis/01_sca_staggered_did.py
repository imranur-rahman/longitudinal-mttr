"""
Step 1: Staggered difference-in-differences for SCA (Dependency-Update-Tool) adoption.

Treatment: a package's repo starts using a Scorecard-detected dependency-update
tool (sca_used: 0 -> 1) at some release. Different repos adopt at different
release-sequence "periods", so this is a staggered-adoption design, estimated
with the Callaway & Sant'Anna (2021) group-time ATT estimator via the
`differences` package.

Outcome: mttr
Covariates (time-varying, included as controls): num_pinned_deps,
  num_floating_deps, num_dependencies, version_releases, release_frequency,
  forks, stars, commits, maintainers

Input:  data/did_analysis/panel.csv
Outputs:
  data/did_analysis/sca_did_simple_att.csv     — overall ATT (simple aggregation)
  data/did_analysis/sca_did_event_study.csv    — ATT by relative period (event study)
  data/did_analysis/sca_did_event_study.png    — event-study plot
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from differences import ATTgt

DATA_DIR = Path("data") / "did_analysis"
IN_CSV = DATA_DIR / "panel.csv"

COVARIATES = [
    "num_pinned_deps", "num_floating_deps", "num_dependencies",
    "version_releases", "release_frequency",
    "forks", "stars", "commits", "maintainers",
]

# A handful of repos have extremely long release sequences (up to 1,624 vs. a
# median of ~10), which blow up the event-study's relative-period range and
# dominate it with high-variance estimates. Cap the time index to keep the
# analysis focused on the typical range of releases.
PERIOD_CAP = 50


def assign_cohort(group: pd.DataFrame):
    """Return (first_treated_period, status) for one unit's sca_used sequence.

    status is one of: 'never', 'always', 'switcher', 'unusable'
    """
    g = group.dropna(subset=["sca_used"]).sort_values("period")
    if g.empty:
        return None, "unusable"
    if (g["sca_used"] == 0).all():
        return None, "never"
    if g["sca_used"].iloc[0] == 1:
        return None, "always"
    # switcher: starts at 0, becomes 1 at some later period
    first_treated = g.loc[g["sca_used"] == 1, "period"].min()
    return first_treated, "switcher"


def main():
    print("Loading panel …")
    df = pd.read_csv(IN_CSV, low_memory=False)
    df["unit_id"] = df["ecosystem"] + "/" + df["github_repo"]
    print(f"  rows: {len(df):,}  units: {df['unit_id'].nunique():,}")

    # ── keep only units where ANY release has mttr > 0 ─────────────────────
    has_mttr = df.groupby("unit_id")["mttr"].max() > 0
    keep_units = has_mttr[has_mttr].index
    n_before = len(df)
    df = df[df["unit_id"].isin(keep_units)].copy()
    print(f"\nKept units with any mttr > 0: {len(keep_units):,} / {has_mttr.shape[0]:,} units "
          f"({n_before:,} -> {len(df):,} rows)")

    # ── cap the time index to avoid a handful of extreme-outlier repos (up
    # to 1,624 releases vs. a median of ~10) blowing up the event-study's
    # relative-period range and dominating it with high-variance estimates ──
    n_before = len(df)
    n_units_before = df["unit_id"].nunique()
    df = df[df["period"] <= PERIOD_CAP].copy()
    print(f"\nCapped period <= {PERIOD_CAP}: {n_before:,} -> {len(df):,} rows "
          f"({n_units_before:,} -> {df['unit_id'].nunique():,} units)")

    # ── assign cohort (first_treated_period) per unit ─────────────────────
    print("\nAssigning SCA-adoption cohorts …")
    cohorts = (
        df.groupby("unit_id")[["period", "sca_used"]]
          .apply(lambda g: pd.Series(assign_cohort(g), index=["first_treated_period", "status"]))
          .reset_index()
    )
    print(cohorts["status"].value_counts())

    df = df.merge(cohorts, on="unit_id", how="left")

    n_before = len(df)
    df = df[df["status"].isin(["never", "switcher"])].copy()
    print(f"\nDropped 'always' and 'unusable' units: {n_before:,} -> {len(df):,} rows")

    # ── drop rows with missing outcome/covariates ─────────────────────────
    needed = ["mttr"] + COVARIATES
    n_before = len(df)
    df = df.dropna(subset=needed).copy()
    print(f"Dropped rows with missing mttr/covariates: {n_before:,} -> {len(df):,} rows")

    n_units = df["unit_id"].nunique()
    n_never = cohorts.loc[cohorts["status"] == "never", "unit_id"].isin(df["unit_id"]).sum()
    n_switch = cohorts.loc[cohorts["status"] == "switcher", "unit_id"].isin(df["unit_id"]).sum()
    print(f"\nUnits remaining for estimation: {n_units:,}  "
          f"(never-treated: {n_never:,}, switchers: {n_switch:,})")

    # ── fit the staggered DiD ──────────────────────────────────────────────
    data = df.set_index(["unit_id", "period"])[["mttr", "first_treated_period"] + COVARIATES]

    print("\nFitting ATTgt …")
    att_gt = ATTgt(data, cohort_column="first_treated_period")
    formula = "mttr ~ " + " + ".join(COVARIATES)
    att_gt.fit(formula, control_group="never_treated")

    # ── aggregate results ──────────────────────────────────────────────────
    simple_att = att_gt.aggregate("simple", overall=True)
    print("\nOverall ATT (simple aggregation):")
    print(simple_att)
    simple_att.to_csv(DATA_DIR / "sca_did_simple_att.csv")

    event_study = att_gt.aggregate("event")
    event_overall = att_gt.aggregate("event", overall=True)
    print("\nOverall event-study ATT (across positive relative periods):")
    print(event_overall)

    event_study.to_csv(DATA_DIR / "sca_did_event_study.csv")
    print(f"\nSaved -> {DATA_DIR / 'sca_did_simple_att.csv'}")
    print(f"Saved -> {DATA_DIR / 'sca_did_event_study.csv'}")

    # ── plot event study ────────────────────────────────────────────────────
    es = event_study.copy()
    es.columns = ["_".join(c for c in col if c).strip() for col in es.columns.to_flat_index()]
    att_col = [c for c in es.columns if c.endswith("ATT")][0]
    se_col = [c for c in es.columns if c.endswith("std_error")][0]
    es = es.reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        es["relative_period"], es[att_col], yerr=1.96 * es[se_col],
        fmt="o-", capsize=3,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(-0.5, color="red", linestyle=":", linewidth=1, label="SCA adoption")
    ax.set_xlabel("Periods relative to SCA adoption")
    ax.set_ylabel("ATT on mttr_all_severities")
    ax.set_title("Event-study: effect of SCA adoption on MTTR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA_DIR / "sca_did_event_study.png", dpi=150)
    print(f"Saved -> {DATA_DIR / 'sca_did_event_study.png'}")


if __name__ == "__main__":
    main()
