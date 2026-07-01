"""
Propensity score weighting (IPW / ATE) analysis for MTTR.

Follows the 6-step protocol from Olmos & Govindasamy (2015):
  1. Baseline outcome analysis (unweighted OLS)
  2. Balance analysis before weighting (SMDs)
  3. Propensity score estimation (logistic regression + GBM)
  4. ATE weight computation
  5. Balance analysis after weighting (SMDs)
  6. Weighted outcome regression (WLS)

Part B: repeat the same 6-step framework treating each of the 8 continuous
covariates as a binary treatment (median split), reporting ATE on mttr_days.

Input:  data/analysis_dataset.csv
Output: data/propensity_weighting/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_IN = ROOT / "data" / "analysis_dataset.csv"
OUT_DIR = ROOT / "data" / "propensity_weighting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTCOME = "mttr_days"
TREATMENT_COL = "sca_tool_score"  # 10=treated, 0=control, -1=drop

# These covariates appear in both the selection model and outcome model.
COVARIATES = [
    "num_direct_dependencies",
    "num_pinned_dependencies",
    "num_floating_dependencies",
    "total_version_releases",
    "release_frequency_days",
    "github_stars",
    "github_forks",
    "github_commits",
    "github_maintainers",
    "scorecard_overall_score",
]

# Covariates that are heavily right-skewed; log1p-transform for the selection model.
LOG_VARS = {
    "num_direct_dependencies",
    "num_pinned_dependencies",
    "num_floating_dependencies",
    "total_version_releases",
    "github_stars",
    "github_forks",
    "github_commits",
    "github_maintainers",
}

WEIGHT_CLIP_PERCENTILE = 99  # clip extreme weights above this percentile
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smd(x, treat, weights=None):
    """Standardized mean difference for variable x by binary treat indicator."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(treat, dtype=float)
    if weights is None:
        weights = np.ones(len(x))
    w = np.asarray(weights, dtype=float)

    def wmean(vals, wts):
        return np.sum(vals * wts) / np.sum(wts)

    def wvar(vals, wts):
        m = wmean(vals, wts)
        return np.sum(wts * (vals - m) ** 2) / np.sum(wts)

    t_mask, c_mask = t == 1, t == 0
    mu_t = wmean(x[t_mask], w[t_mask])
    mu_c = wmean(x[c_mask], w[c_mask])
    # pooled SD uses unweighted variances (standard definition)
    var_t = np.var(x[t_mask])
    var_c = np.var(x[c_mask])
    pooled_sd = np.sqrt((var_t + var_c) / 2)
    if pooled_sd == 0:
        return 0.0
    return (mu_t - mu_c) / pooled_sd


def balance_table(df, covariates, treat_col, weights=None):
    rows = []
    for cov in covariates + ["ecosystem_pypi"]:
        if cov not in df.columns:
            continue
        s = smd(df[cov], df[treat_col], weights)
        rows.append({"covariate": cov, "smd": s, "abs_smd": abs(s)})
    return pd.DataFrame(rows)


def compute_ate_weights(ps, treat, clip_pct=WEIGHT_CLIP_PERCENTILE):
    """ATE weights: 1/ps for treated, 1/(1-ps) for controls. Clip at clip_pct."""
    w = np.where(treat == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    cap = np.percentile(w, clip_pct)
    w = np.clip(w, None, cap)
    return w


def fit_propensity_logit(X_train, y_train):
    """Logistic regression propensity scores (sklearn, L2 regularization, scaled)."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    clf.fit(X_sc, y_train)
    ps = clf.predict_proba(X_sc)[:, 1]
    return ps


def fit_propensity_gbm(X_train, y_train):
    """GBM propensity scores with paper-recommended parameters."""
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,              # interaction_depth=4 in gbm
        learning_rate=0.0005,     # shrinkage=0.0005
        subsample=0.8,            # train.fraction=0.8
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    ps = clf.predict_proba(X_train)[:, 1]
    return ps


def run_wls(df, outcome, treat_col, covariate_cols, weights=None):
    """Weighted (or unweighted) OLS via statsmodels formula API."""
    rhs_terms = [treat_col] + covariate_cols
    # sanitize col names to statsmodels-safe format
    formula_rhs = " + ".join(f"Q('{c}')" for c in rhs_terms)
    formula = f"Q('{outcome}') ~ {formula_rhs}"
    if weights is None:
        result = smf.ols(formula, data=df).fit(cov_type="HC3")
    else:
        result = smf.wls(formula, data=df, weights=weights).fit(cov_type="HC3")
    return result


def extract_treatment_ate(result, treat_col):
    """Pull ATE coefficient + stats for the treatment variable from a WLS result."""
    key = f"Q('{treat_col}')"
    if key not in result.params:
        # statsmodels may name it differently
        key = treat_col
    if key not in result.params:
        return None
    return {
        "ate": result.params[key],
        "se": result.bse[key],
        "pvalue": result.pvalues[key],
        "ci_low": result.conf_int().loc[key, 0],
        "ci_high": result.conf_int().loc[key, 1],
    }


# ---------------------------------------------------------------------------
# Data loading + filtering
# ---------------------------------------------------------------------------
print("Loading data ...")
df_raw = pd.read_csv(DATA_IN)
print(f"  Raw rows: {len(df_raw):,}")

# Drop unknown SCA scores
df = df_raw[df_raw[TREATMENT_COL] != -1].copy()
print(f"  After dropping sca=-1: {len(df):,}")

# Binarize treatment
df["treat"] = (df[TREATMENT_COL] == 10).astype(int)

# Filter to packages with at least one MTTR > 0 (consistent with DiD sample)
pkg_has_nonzero = df.groupby("package_name")[OUTCOME].transform("max") > 0
df = df[pkg_has_nonzero].copy()
print(f"  After MTTR>0 filter (at least one non-zero per package): {len(df):,}")
print(f"  Unique packages: {df['package_name'].nunique():,}")
print(f"  Treated (SCA=1): {df['treat'].sum():,}  Control (SCA=0): {(df['treat']==0).sum():,}")

# Drop rows with missing covariates
df = df.dropna(subset=COVARIATES + [OUTCOME]).copy()
print(f"  After dropping NA in covariates/outcome: {len(df):,}")

# One-hot ecosystem (baseline = npm)
df["ecosystem_pypi"] = (df["ecosystem"] == "pypi").astype(int)

# Log1p-transform for selection model
for col in LOG_VARS:
    df[f"log_{col}"] = np.log1p(df[col])

# Selection model feature columns (log-transformed where appropriate)
SEL_FEATURES = (
    [f"log_{c}" if c in LOG_VARS else c for c in COVARIATES]
    + ["ecosystem_pypi"]
)
# Outcome model uses raw covariates + ecosystem
OUT_COVARIATES = COVARIATES + ["ecosystem_pypi"]

X_sel = df[SEL_FEATURES].values
y_treat = df["treat"].values

print(f"\nSelection model features: {SEL_FEATURES}")
print(f"Outcome model covariates: {OUT_COVARIATES}")

# ---------------------------------------------------------------------------
# STEP 1 — Baseline outcome regression (unweighted)
# ---------------------------------------------------------------------------
print("\n=== STEP 1: Baseline OLS (unweighted) ===")
step1 = run_wls(df, OUTCOME, "treat", OUT_COVARIATES, weights=None)
print(step1.summary())
with open(OUT_DIR / "step1_baseline_regression.txt", "w") as f:
    f.write(str(step1.summary()))

# ---------------------------------------------------------------------------
# STEP 2 — Pre-weighting balance
# ---------------------------------------------------------------------------
print("\n=== STEP 2: Balance BEFORE weighting ===")
bal_before = balance_table(df, COVARIATES, "treat", weights=None)
print(bal_before.to_string(index=False))
bal_before.to_csv(OUT_DIR / "step2_balance_before.csv", index=False)

# ---------------------------------------------------------------------------
# STEP 3 — Propensity score estimation
# ---------------------------------------------------------------------------
print("\n=== STEP 3a: Logistic regression propensity scores ===")
ps_logit = fit_propensity_logit(X_sel, y_treat)
df["ps_logit"] = ps_logit
print(f"  PS range: [{ps_logit.min():.4f}, {ps_logit.max():.4f}]  mean={ps_logit.mean():.4f}")

print("=== STEP 3b: GBM propensity scores ===")
ps_gbm = fit_propensity_gbm(X_sel, y_treat)
df["ps_gbm"] = ps_gbm
print(f"  PS range: [{ps_gbm.min():.4f}, {ps_gbm.max():.4f}]  mean={ps_gbm.mean():.4f}")

# ---------------------------------------------------------------------------
# STEP 4 — ATE weight computation
# ---------------------------------------------------------------------------
print("\n=== STEP 4: ATE weights ===")
w_logit = compute_ate_weights(ps_logit, y_treat)
w_gbm = compute_ate_weights(ps_gbm, y_treat)
df["w_ate_logit"] = w_logit
df["w_ate_gbm"] = w_gbm

for name, w in [("logit", w_logit), ("gbm", w_gbm)]:
    print(f"  {name}: mean={w.mean():.3f}  max={w.max():.3f}  "
          f"moment_check(treated/control)={w[y_treat==1].mean():.3f}/{w[y_treat==0].mean():.3f}")

# ---------------------------------------------------------------------------
# STEP 5 — Post-weighting balance
# ---------------------------------------------------------------------------
print("\n=== STEP 5: Balance AFTER weighting ===")
bal_after_logit = balance_table(df, COVARIATES, "treat", weights=w_logit)
bal_after_gbm = balance_table(df, COVARIATES, "treat", weights=w_gbm)
print("  Logit weights:")
print(bal_after_logit.to_string(index=False))
print("  GBM weights:")
print(bal_after_gbm.to_string(index=False))
bal_after_logit.to_csv(OUT_DIR / "step5_balance_after_logit.csv", index=False)
bal_after_gbm.to_csv(OUT_DIR / "step5_balance_after_gbm.csv", index=False)

# ---------------------------------------------------------------------------
# STEP 6 — Weighted outcome regression
# ---------------------------------------------------------------------------
print("\n=== STEP 6: Weighted outcome regression ===")
step6_logit = run_wls(df, OUTCOME, "treat", OUT_COVARIATES, weights=w_logit)
step6_gbm = run_wls(df, OUTCOME, "treat", OUT_COVARIATES, weights=w_gbm)
print("  --- Logit weights ---")
print(step6_logit.summary())
print("  --- GBM weights ---")
print(step6_gbm.summary())

with open(OUT_DIR / "step6_outcome_logit.txt", "w") as f:
    f.write(str(step6_logit.summary()))
with open(OUT_DIR / "step6_outcome_gbm.txt", "w") as f:
    f.write(str(step6_gbm.summary()))

# Also run with log1p(mttr_days) as outcome (robustness)
df["log_mttr"] = np.log1p(df[OUTCOME])
step6_logit_log = run_wls(df, "log_mttr", "treat", OUT_COVARIATES, weights=w_logit)
step6_gbm_log = run_wls(df, "log_mttr", "treat", OUT_COVARIATES, weights=w_gbm)
with open(OUT_DIR / "step6_outcome_logit_logmttr.txt", "w") as f:
    f.write(str(step6_logit_log.summary()))
with open(OUT_DIR / "step6_outcome_gbm_logmttr.txt", "w") as f:
    f.write(str(step6_gbm_log.summary()))

# ---------------------------------------------------------------------------
# Love plot: SMD before vs. after
# ---------------------------------------------------------------------------
cov_labels = COVARIATES + ["ecosystem_pypi"]
smd_before = bal_before.set_index("covariate")["abs_smd"]
smd_logit = bal_after_logit.set_index("covariate")["abs_smd"]
smd_gbm = bal_after_gbm.set_index("covariate")["abs_smd"]

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(cov_labels))
ax.scatter(smd_before.reindex(cov_labels), y_pos, marker="o", color="black",
           label="Before weighting", zorder=3)
ax.scatter(smd_logit.reindex(cov_labels), y_pos, marker="^", color="steelblue",
           label="After (logit)", zorder=3)
ax.scatter(smd_gbm.reindex(cov_labels), y_pos, marker="s", color="darkorange",
           label="After (GBM)", zorder=3)
ax.axvline(0.1, color="red", linestyle="--", linewidth=0.8, label="SMD=0.1 threshold")
ax.set_yticks(y_pos)
ax.set_yticklabels(cov_labels)
ax.set_xlabel("Absolute Standardized Mean Difference")
ax.set_title("Love Plot: Covariate Balance Before and After IPW")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "balance_plot.png", dpi=150)
plt.close()

# Propensity score overlap plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, (name, ps) in zip(axes, [("Logistic Regression", ps_logit), ("GBM", ps_gbm)]):
    ax.hist(ps[y_treat == 1], bins=50, alpha=0.5, density=True,
            color="steelblue", label="Treated (SCA=1)")
    ax.hist(ps[y_treat == 0], bins=50, alpha=0.5, density=True,
            color="darkorange", label="Control (SCA=0)")
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Density")
    ax.set_title(f"PS overlap: {name}")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "propensity_overlap.png", dpi=150)
plt.close()

print("\nBalance plot and overlap plot saved.")

# ---------------------------------------------------------------------------
# PART B — ATE of each covariate (binary treatment via median split)
# ---------------------------------------------------------------------------
print("\n=== PART B: ATE of each covariate (median-split binary treatment) ===")

ATE_VARS = COVARIATES  # 8 continuous covariates

ate_rows = []

# First add the main SCA treatment result for comparison
sca_ate_logit = extract_treatment_ate(step6_logit, "treat")
sca_ate_gbm = extract_treatment_ate(step6_gbm, "treat")
ate_rows.append({
    "variable": "sca_tool",
    "median_threshold": None,
    "n_treated": int((df["treat"] == 1).sum()),
    "n_control": int((df["treat"] == 0).sum()),
    "ate_logit": sca_ate_logit["ate"],
    "se_logit": sca_ate_logit["se"],
    "p_logit": sca_ate_logit["pvalue"],
    "ci_low_logit": sca_ate_logit["ci_low"],
    "ci_high_logit": sca_ate_logit["ci_high"],
    "ate_gbm": sca_ate_gbm["ate"],
    "se_gbm": sca_ate_gbm["se"],
    "p_gbm": sca_ate_gbm["pvalue"],
    "ci_low_gbm": sca_ate_gbm["ci_low"],
    "ci_high_gbm": sca_ate_gbm["ci_high"],
    "max_smd_before": float(bal_before["abs_smd"].max()),
    "max_smd_after_logit": float(bal_after_logit["abs_smd"].max()),
    "max_smd_after_gbm": float(bal_after_gbm["abs_smd"].max()),
})

for var in ATE_VARS:
    print(f"\n  --- {var} ---")
    med = df[var].median()

    df_b = df.copy()
    df_b["treat_x"] = (df_b[var] > med).astype(int)

    n_t = int((df_b["treat_x"] == 1).sum())
    n_c = int((df_b["treat_x"] == 0).sum())
    print(f"  median={med:.4f}  treated={n_t:,}  control={n_c:,}")

    # Selection model features: log1p of other log-vars + non-log vars,
    # plus sca treatment and ecosystem, minus the target var itself
    sel_features_b = []
    for c in COVARIATES:
        if c == var:
            continue  # exclude target
        sel_features_b.append(f"log_{c}" if c in LOG_VARS else c)
    sel_features_b += ["ecosystem_pypi", "treat"]

    # Ensure all cols exist
    sel_features_b = [c for c in sel_features_b if c in df_b.columns]
    X_b = df_b[sel_features_b].values
    y_b = df_b["treat_x"].values

    # Skip if degenerate
    if y_b.sum() < 10 or (len(y_b) - y_b.sum()) < 10:
        print(f"  Skipping {var}: degenerate split")
        continue

    ps_b_logit = fit_propensity_logit(X_b, y_b)
    ps_b_gbm = fit_propensity_gbm(X_b, y_b)
    w_b_logit = compute_ate_weights(ps_b_logit, y_b)
    w_b_gbm = compute_ate_weights(ps_b_gbm, y_b)

    # Balance before/after
    conf_vars_b = [c for c in COVARIATES if c != var] + ["ecosystem_pypi", "treat"]
    conf_vars_b = [c for c in conf_vars_b if c in df_b.columns]
    bal_b_before = balance_table(df_b, conf_vars_b, "treat_x", weights=None)
    bal_b_after_logit = balance_table(df_b, conf_vars_b, "treat_x", weights=w_b_logit)
    bal_b_after_gbm = balance_table(df_b, conf_vars_b, "treat_x", weights=w_b_gbm)

    # Outcome model: outcome ~ treat_x + confounders (raw)
    out_cov_b = [c for c in COVARIATES if c != var] + ["ecosystem_pypi", "treat"]
    out_cov_b = [c for c in out_cov_b if c in df_b.columns]

    res_b_logit = run_wls(df_b, OUTCOME, "treat_x", out_cov_b, weights=w_b_logit)
    res_b_gbm = run_wls(df_b, OUTCOME, "treat_x", out_cov_b, weights=w_b_gbm)

    ate_l = extract_treatment_ate(res_b_logit, "treat_x")
    ate_g = extract_treatment_ate(res_b_gbm, "treat_x")

    if ate_l is None or ate_g is None:
        print(f"  Could not extract ATE for {var}")
        continue

    print(f"  ATE (logit): {ate_l['ate']:.4f}  SE={ate_l['se']:.4f}  p={ate_l['pvalue']:.4f}")
    print(f"  ATE (GBM):   {ate_g['ate']:.4f}  SE={ate_g['se']:.4f}  p={ate_g['pvalue']:.4f}")

    ate_rows.append({
        "variable": var,
        "median_threshold": float(med),
        "n_treated": n_t,
        "n_control": n_c,
        "ate_logit": ate_l["ate"],
        "se_logit": ate_l["se"],
        "p_logit": ate_l["pvalue"],
        "ci_low_logit": ate_l["ci_low"],
        "ci_high_logit": ate_l["ci_high"],
        "ate_gbm": ate_g["ate"],
        "se_gbm": ate_g["se"],
        "p_gbm": ate_g["pvalue"],
        "ci_low_gbm": ate_g["ci_low"],
        "ci_high_gbm": ate_g["ci_high"],
        "max_smd_before": float(bal_b_before["abs_smd"].max()),
        "max_smd_after_logit": float(bal_b_after_logit["abs_smd"].max()),
        "max_smd_after_gbm": float(bal_b_after_gbm["abs_smd"].max()),
    })

ate_df = pd.DataFrame(ate_rows)
ate_df.to_csv(OUT_DIR / "ate_all_variables.csv", index=False)
print("\nATE summary table:")
display_cols = ["variable", "ate_logit", "se_logit", "p_logit",
                "ate_gbm", "p_gbm", "max_smd_before", "max_smd_after_logit"]
print(ate_df[display_cols].to_string(index=False, float_format="{:.4f}".format))

# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------
def fmt_result(res, treat_col="treat"):
    ate = extract_treatment_ate(res, treat_col)
    if ate is None:
        return "N/A"
    sig = "**" if ate["pvalue"] < 0.05 else ""
    return (f"{sig}ATE={ate['ate']:.3f}{sig}  SE={ate['se']:.3f}  "
            f"p={ate['pvalue']:.4f}  95%CI=[{ate['ci_low']:.3f}, {ate['ci_high']:.3f}]")

step1_ate = extract_treatment_ate(step1, "treat")
results_md = f"""# Propensity Score Weighting (IPW) Analysis: ATE on MTTR

**Method:** Inverse-probability weighting (ATE), following Olmos & Govindasamy (2015).
Propensity scores estimated via (a) logistic regression and (b) GBM
(GradientBoostingClassifier, n_estimators=100, depth=4, lr=0.0005, subsample=0.8).
ATE weights clipped at the {WEIGHT_CLIP_PERCENTILE}th percentile to limit extreme leverage.
Weighted regressions use HC3 robust standard errors.

**Sample:** packages with at least one MTTR > 0 in `analysis_dataset.csv`.
n = {len(df):,} package-version rows, {df['package_name'].nunique():,} packages.
Treated (SCA detected): {int((df['treat']==1).sum()):,}  Control: {int((df['treat']==0).sum()):,}.

---

## Step 1 — Baseline OLS (no weighting)

Treatment (SCA tool): {fmt_result(step1)}

---

## Step 2 — Pre-weighting balance

Max |SMD| = {float(bal_before['abs_smd'].max()):.3f}

| Covariate | |SMD| before |
|---|---|
{chr(10).join(f"| {r.covariate} | {r.abs_smd:.3f} |" for _, r in bal_before.iterrows())}

---

## Step 3 — Propensity score estimation

- **Logistic regression**: PS range [{ps_logit.min():.4f}, {ps_logit.max():.4f}], mean={ps_logit.mean():.4f}
- **GBM**: PS range [{ps_gbm.min():.4f}, {ps_gbm.max():.4f}], mean={ps_gbm.mean():.4f}

See `propensity_overlap.png` for distribution by treatment group.

---

## Step 5 — Post-weighting balance

| Covariate | |SMD| before | |SMD| after (logit) | |SMD| after (GBM) |
|---|---|---|---|
{chr(10).join(
    f"| {r.covariate} | {bal_before.set_index('covariate')['abs_smd'].get(r.covariate, float('nan')):.3f} | {r.abs_smd:.3f} | {bal_after_gbm.set_index('covariate')['abs_smd'].get(r.covariate, float('nan')):.3f} |"
    for _, r in bal_after_logit.iterrows()
)}

See `balance_plot.png` (Love plot) for visual comparison.

---

## Step 6 — Weighted outcome regression

**Logit weights:**  {fmt_result(step6_logit)}
**GBM weights:**    {fmt_result(step6_gbm)}
**Logit weights (log1p MTTR):**  {fmt_result(step6_logit_log)}
**GBM weights (log1p MTTR):**    {fmt_result(step6_gbm_log)}

---

## Part B — ATE of each variable (median-split binary treatment)

| Variable | Median threshold | ATE (logit) | p (logit) | ATE (GBM) | p (GBM) | Max SMD before | Max SMD after (logit) |
|---|---|---|---|---|---|---|---|
{chr(10).join(
    f"| {r.variable} | {r.median_threshold if r.median_threshold is not None else 'binary'} | {r.ate_logit:.3f} | {r.p_logit:.4f} | {r.ate_gbm:.3f} | {r.p_gbm:.4f} | {r.max_smd_before:.3f} | {r.max_smd_after_logit:.3f} |"
    for _, r in ate_df.iterrows()
)}

*Significant at p<0.05 if p < 0.05.*

---

## Comparison with prior analyses

- **DiD (Callaway-Sant'Anna):** ATT = 0.555 days (p > 0.05) — null result
- **Panel FE:** SCA coefficient ≈ 0.034 (p ≈ 0.06–0.09) — marginally positive, not significant
- **IPW/ATE (this analysis):** see Step 6 above

---

## Caveats

- Weighted OLS treats `mttr_days` as continuous; the outcome is zero-inflated
  (~74% zeros even in the MTTR>0 filtered sample). Log1p results address right-skew
  but not zero-inflation. A PPML / hurdle model would be the next step.
- The median split for Part B binary treatments is arbitrary; results may differ at
  other thresholds (e.g., top/bottom quartile).
- Cross-sectional IPW (no entity fixed effects) — unlike the Panel FE analysis,
  between-package confounding is addressed only via observed covariates.
"""

with open(OUT_DIR / "RESULTS.md", "w") as f:
    f.write(results_md)

print(f"\nAll outputs written to {OUT_DIR}")
print("Done.")
