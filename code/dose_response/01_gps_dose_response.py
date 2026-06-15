"""
Step 1: Generalized Propensity Score (GPS) dose-response analysis
(Hirano & Imbens, 2004).

For each continuous "dose" variable, estimates the average dose-response
function (ADRF) of mttr:
  1. GPS step: fit dose ~ confounders (OLS), compute GPS_i = N(dose_i; dose_hat_i, sigma^2).
  2. Outcome step: fit mttr ~ dose + dose^2 + GPS + GPS^2 + dose*GPS (OLS).
  3. ADRF: for a grid of dose values t, average the predicted outcome over all
     units using GPS_i(t) = N(t; dose_hat_i, sigma^2).
  95% CIs via bootstrap (resampling observations, repeating steps 1-3).

Dose variables: num_dependencies, num_pinned_deps, num_floating_deps,
  release_frequency, stars, forks, commits, maintainers
Confounders for each model: the other 7 dose variables + scorecard_aggregate_score
  + ecosystem (one-hot)

Input:  data/dose_response/data.csv
Outputs:
  data/dose_response/{var}_adrf.csv
  data/dose_response/{var}_adrf.png
  data/dose_response/gps_dose_response_summary.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_DIR = Path("data") / "dose_response"
IN_CSV = DATA_DIR / "data.csv"

DOSE_VARS = [
    "num_dependencies", "num_pinned_deps", "num_floating_deps",
    "release_frequency", "stars", "forks", "commits", "maintainers",
]

# These dose variables were log1p-transformed in 00_prepare_data.py.
LOG_VARS = {
    "num_dependencies", "num_pinned_deps", "num_floating_deps",
    "stars", "forks", "commits", "maintainers",
}

N_GRID = 19
N_BOOT = 200
RNG_SEED = 42


def model_col(var: str) -> str:
    return f"{var}_log" if var in LOG_VARS else var


def to_original(var: str, vals: np.ndarray) -> np.ndarray:
    return np.expm1(vals) if var in LOG_VARS else vals


def ols_fit(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def fit_adrf(df: pd.DataFrame, dose: str, confound_cols: list[str], grid: np.ndarray) -> np.ndarray:
    """Fit GPS + outcome model and return ADRF values for the given dose grid (model scale)."""
    D = df[model_col(dose)].to_numpy(dtype=float)
    Xc = df[confound_cols].to_numpy(dtype=float)
    Xc1 = np.column_stack([np.ones(len(df)), Xc])

    # ── Step 1: GPS ──────────────────────────────────────────────────────
    beta_d, resid_d = ols_fit(Xc1, D)
    dof = len(df) - Xc1.shape[1]
    sigma = np.sqrt(np.sum(resid_d ** 2) / dof)
    D_hat = Xc1 @ beta_d
    gps = norm.pdf((D - D_hat) / sigma) / sigma

    # ── Step 2: outcome model ───────────────────────────────────────────
    y = df["mttr"].to_numpy(dtype=float)
    X_out = np.column_stack([np.ones(len(df)), D, D ** 2, gps, gps ** 2, D * gps])
    beta_out, _ = ols_fit(X_out, y)

    # ── Step 3: ADRF over grid ──────────────────────────────────────────
    mu = np.empty(len(grid))
    n = len(df)
    for j, t in enumerate(grid):
        gps_t = norm.pdf((t - D_hat) / sigma) / sigma
        X_pred = np.column_stack([
            np.ones(n), np.full(n, t), np.full(n, t ** 2),
            gps_t, gps_t ** 2, np.full(n, t) * gps_t,
        ])
        mu[j] = (X_pred @ beta_out).mean()
    return mu


def main():
    print("Loading dose-response dataset …")
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"  rows: {len(df):,}")

    eco_dummies = pd.get_dummies(df["ecosystem"], prefix="eco", drop_first=True, dtype=float)
    df = pd.concat([df, eco_dummies], axis=1)
    eco_cols = list(eco_dummies.columns)

    all_model_cols = [model_col(v) for v in DOSE_VARS]
    rng = np.random.default_rng(RNG_SEED)
    summary_rows = []

    for dose in DOSE_VARS:
        print(f"\n=== {dose} ===")
        dcol = model_col(dose)
        confound_cols = [c for c in all_model_cols if c != dcol] + \
            ["scorecard_aggregate_score"] + eco_cols

        d_vals = df[dcol].to_numpy(dtype=float)
        grid_model = np.linspace(np.percentile(d_vals, 5), np.percentile(d_vals, 95), N_GRID)

        # point estimate
        mu = fit_adrf(df, dose, confound_cols, grid_model)

        # bootstrap CIs
        boot_mu = np.empty((N_BOOT, N_GRID))
        n = len(df)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            df_b = df.iloc[idx]
            boot_mu[b] = fit_adrf(df_b, dose, confound_cols, grid_model)
        ci_lower = np.percentile(boot_mu, 2.5, axis=0)
        ci_upper = np.percentile(boot_mu, 97.5, axis=0)

        grid_original = to_original(dose, grid_model)

        out = pd.DataFrame({
            "dose": grid_original,
            "mu": mu,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })
        out_csv = DATA_DIR / f"{dose}_adrf.csv"
        out.to_csv(out_csv, index=False)
        print(f"  saved -> {out_csv}")

        # plot
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(out["dose"], out["mu"], "o-", color="C0")
        ax.fill_between(out["dose"], out["ci_lower"], out["ci_upper"], alpha=0.25, color="C0")
        ax.set_xlabel(dose)
        ax.set_ylabel("Predicted mttr_all_severities (ADRF)")
        ax.set_title(f"GPS dose-response: {dose} -> mttr")
        fig.tight_layout()
        out_png = DATA_DIR / f"{dose}_adrf.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  saved -> {out_png}")

        # summary: slope between the 5th and 95th percentile dose, and a
        # significance heuristic based on whether the bootstrap CIs at the
        # two endpoints are non-overlapping (a distinguishable overall change).
        lo_idx, hi_idx = 0, N_GRID - 1  # grid spans 5th-95th percentile
        slope = (mu[hi_idx] - mu[lo_idx]) / (grid_original[hi_idx] - grid_original[lo_idx])
        endpoints_nonoverlapping = bool(
            (ci_upper[lo_idx] < ci_lower[hi_idx]) or (ci_upper[hi_idx] < ci_lower[lo_idx])
        )
        summary_rows.append({
            "dose_variable": dose,
            "mu_at_p5": mu[0],
            "mu_at_p95": mu[-1],
            "slope_p5_to_p95": slope,
            "endpoints_ci_nonoverlapping": endpoints_nonoverlapping,
        })
        print(f"  mu(p5)={mu[0]:.4f}  mu(p95)={mu[-1]:.4f}  slope={slope:.6f}  "
              f"endpoints_ci_nonoverlapping={endpoints_nonoverlapping}")

    summary = pd.DataFrame(summary_rows)
    summary_csv = DATA_DIR / "gps_dose_response_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved -> {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
