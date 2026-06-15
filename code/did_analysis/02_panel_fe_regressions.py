"""
Step 2: Panel fixed-effects regressions of mttr on repo/package characteristics.

For each covariate below, run a univariate PanelOLS regression of mttr on that
covariate with entity (ecosystem/github_repo) and time (period) fixed effects,
clustering standard errors by entity. Also run one combined multivariable model
with all covariates together.

Covariates: num_pinned_deps, num_floating_deps, num_dependencies,
  version_releases, release_frequency, forks, stars, commits, maintainers,
  sca_used

Input:  data/did_analysis/panel.csv
Output: data/did_analysis/panel_fe_results.csv
"""

from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS

DATA_DIR = Path("data") / "did_analysis"
IN_CSV = DATA_DIR / "panel.csv"
OUT_CSV = DATA_DIR / "panel_fe_results.csv"

COVARIATES = [
    "num_pinned_deps", "num_floating_deps", "num_dependencies",
    "version_releases", "release_frequency",
    "forks", "stars", "commits", "maintainers", "sca_used",
]


def fit_panel_ols(data: pd.DataFrame, exog_cols: list[str]):
    exog = data[exog_cols].copy()
    exog.insert(0, "const", 1.0)
    model = PanelOLS(data["mttr"], exog, entity_effects=True, time_effects=True)
    return model.fit(cov_type="clustered", cluster_entity=True)


def main():
    print("Loading panel …")
    df = pd.read_csv(IN_CSV, low_memory=False)
    df["unit_id"] = df["ecosystem"] + "/" + df["github_repo"]
    df = df.set_index(["unit_id", "period"])
    print(f"  rows: {len(df):,}  units: {df.index.get_level_values('unit_id').nunique():,}")

    rows = []

    # ── univariate models ───────────────────────────────────────────────
    for var in COVARIATES:
        sub = df[["mttr", var]].dropna()
        print(f"\n[{var}] n={len(sub):,}")
        res = fit_panel_ols(sub, [var])
        rows.append({
            "model": f"univariate: {var}",
            "variable": var,
            "coef": res.params[var],
            "std_error": res.std_errors[var],
            "pvalue": res.pvalues[var],
            "n_obs": res.nobs,
            "n_entities": sub.index.get_level_values("unit_id").nunique(),
            "r2_within": res.rsquared_within,
        })
        print(f"  coef={res.params[var]:.5f}  se={res.std_errors[var]:.5f}  p={res.pvalues[var]:.4f}")

    # ── combined multivariable model ────────────────────────────────────
    sub = df[["mttr"] + COVARIATES].dropna()
    print(f"\n[combined] n={len(sub):,}")
    res = fit_panel_ols(sub, COVARIATES)
    for var in COVARIATES:
        rows.append({
            "model": "combined",
            "variable": var,
            "coef": res.params[var],
            "std_error": res.std_errors[var],
            "pvalue": res.pvalues[var],
            "n_obs": res.nobs,
            "n_entities": sub.index.get_level_values("unit_id").nunique(),
            "r2_within": res.rsquared_within,
        })
        print(f"  {var:<20} coef={res.params[var]:.5f}  se={res.std_errors[var]:.5f}  p={res.pvalues[var]:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
