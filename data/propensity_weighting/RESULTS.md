# Propensity Score Weighting (IPW) Analysis: ATE on MTTR

**Method:** Inverse-probability weighting (ATE), following Olmos & Govindasamy (2015).
Propensity scores estimated via (a) logistic regression and (b) GBM
(GradientBoostingClassifier, n_estimators=100, depth=4, lr=0.0005, subsample=0.8).
ATE weights clipped at the 99th percentile to limit extreme leverage.
Weighted regressions use HC3 robust standard errors.

**Sample:** packages with at least one MTTR > 0 in `analysis_dataset.csv`.
n = 42,975 package-version rows, 1,825 packages.
Treated (SCA detected): 29,795  Control: 13,180.

---

## Step 1 — Baseline OLS (no weighting)

Treatment (SCA tool): ATE=-0.100  SE=0.068  p=0.1402  95%CI=[-0.233, 0.033]

---

## Step 2 — Pre-weighting balance

Max |SMD| = 0.392

| Covariate | |SMD| before |
|---|---|
| num_direct_dependencies | 0.012 |
| num_pinned_dependencies | 0.121 |
| num_floating_dependencies | 0.069 |
| total_version_releases | 0.286 |
| release_frequency_days | 0.085 |
| github_stars | 0.210 |
| github_forks | 0.186 |
| github_commits | 0.392 |
| github_maintainers | 0.355 |
| scorecard_overall_score | 0.086 |
| ecosystem_pypi | 0.231 |

---

## Step 3 — Propensity score estimation

- **Logistic regression**: PS range [0.1431, 0.9681], mean=0.6932
- **GBM**: PS range [0.6601, 0.7079], mean=0.6933

See `propensity_overlap.png` for distribution by treatment group.

---

## Step 5 — Post-weighting balance

| Covariate | |SMD| before | |SMD| after (logit) | |SMD| after (GBM) |
|---|---|---|---|
| num_direct_dependencies | 0.012 | 0.056 | 0.015 |
| num_pinned_dependencies | 0.121 | 0.017 | 0.127 |
| num_floating_dependencies | 0.069 | 0.062 | 0.066 |
| total_version_releases | 0.286 | 0.094 | 0.287 |
| release_frequency_days | 0.085 | 0.052 | 0.089 |
| github_stars | 0.210 | 0.117 | 0.207 |
| github_forks | 0.186 | 0.106 | 0.184 |
| github_commits | 0.392 | 0.243 | 0.384 |
| github_maintainers | 0.355 | 0.154 | 0.341 |
| scorecard_overall_score | 0.086 | 0.050 | 0.080 |
| ecosystem_pypi | 0.231 | 0.007 | 0.229 |

See `balance_plot.png` (Love plot) for visual comparison.

---

## Step 6 — Weighted outcome regression

**Logit weights:**  ATE=0.089  SE=0.060  p=0.1381  95%CI=[-0.028, 0.206]
**GBM weights:**    ATE=-0.104  SE=0.068  p=0.1271  95%CI=[-0.239, 0.030]
**Logit weights (log1p MTTR):**  **ATE=0.042**  SE=0.008  p=0.0000  95%CI=[0.027, 0.057]
**GBM weights (log1p MTTR):**    **ATE=0.038**  SE=0.008  p=0.0000  95%CI=[0.023, 0.054]

---

## Part B — ATE of each variable (median-split binary treatment)

| Variable | Median threshold | ATE (logit) | p (logit) | ATE (GBM) | p (GBM) | Max SMD before | Max SMD after (logit) |
|---|---|---|---|---|---|---|---|
| sca_tool | nan | 0.089 | 0.1381 | -0.104 | 0.1271 | 0.392 | 0.243 |
| num_direct_dependencies | 13.0 | -1.009 | 0.0000 | -1.945 | 0.0000 | 0.509 | 0.244 |
| num_pinned_dependencies | 2.0 | -0.390 | 0.0000 | -0.374 | 0.0000 | 0.484 | 0.282 |
| num_floating_dependencies | 22.0 | -0.361 | 0.0013 | -1.286 | 0.0000 | 0.718 | 0.430 |
| total_version_releases | 168.0 | -0.569 | 0.0000 | -1.617 | 0.0000 | 0.959 | 0.468 |
| release_frequency_days | 74.0 | -1.476 | 0.0000 | -2.173 | 0.0000 | 1.097 | 0.473 |
| github_stars | 28.0 | -0.516 | 0.0000 | -0.156 | 0.0114 | 0.834 | 0.623 |
| github_forks | 14.0 | 1.289 | 0.0000 | 0.262 | 0.0000 | 0.927 | 0.602 |
| github_commits | 809.0 | -0.929 | 0.0000 | -1.550 | 0.0000 | 1.195 | 0.585 |
| github_maintainers | 6.0 | 0.269 | 0.0051 | -0.182 | 0.0020 | 0.692 | 0.390 |
| scorecard_overall_score | 5.0 | -0.088 | 0.1370 | -0.048 | 0.4040 | 0.522 | 0.104 |

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
