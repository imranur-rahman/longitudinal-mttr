# GPS Continuous Dose-Response Analysis: Effect of Repo/Package Characteristics on MTTR

**Data:** `data/dose_response/data.csv` (209,924 package-version rows, matched to a
GitHub repo). Outcome: `mttr` (`mttr_all_severities`).

**Method:** Generalized Propensity Score (GPS) average dose-response function (ADRF),
Hirano & Imbens (2004). For each of 8 continuous "dose" variables:
1. Fit `dose ~ confounders` (OLS) to get the GPS — a normal density evaluating how
   "typical" a unit's observed dose is given its confounders.
2. Fit `mttr ~ dose + dose^2 + GPS + GPS^2 + dose*GPS` (OLS).
3. For a grid of dose values spanning the 5th-95th percentile, average the predicted
   outcome over all units (using each unit's confounder-implied GPS at that dose) to
   get the ADRF `mu(t)`.
4. 95% CIs via 200 bootstrap resamples.

Confounders for each model: the other 7 dose variables (log1p-transformed where
heavily skewed) + `scorecard_aggregate_score` + `ecosystem` (one-hot).

Heavily right-skewed count variables (`num_dependencies`, `num_pinned_deps`,
`num_floating_deps`, `stars`, `forks`, `commits`, `maintainers`) were log1p-transformed
for modeling; dose grids/plots are reported back on the original scale via `expm1`.
`release_frequency` was left untransformed.

## Results summary

| Dose variable | mu(p5) | mu(p95) | slope (p5->p95) | Shape | Endpoints CI non-overlapping |
|---|---|---|---|---|---|
| num_dependencies | 0.032 | 0.478 | +0.0139 | Sharp rise 0->~5 deps, then plateau/slight increase to 32 | Yes |
| num_pinned_deps | 0.302 | 0.338 | +0.0013 | Rises 0->~5 (peak ~0.45), falls to ~23, flat after | No |
| num_floating_deps | 0.796 | 0.205 | -0.0083 | Steep monotonic decline 0->70 | Yes |
| release_frequency | 0.701 | 0.276 | -0.0008 | U-shaped: high at low freq, minimum near 250-300, rises again to 550 | Yes |
| stars | 0.700 | 0.331 | -0.0001 | Monotonic decline, fast decay then long tail | Yes |
| forks | 0.381 | 0.921 | +0.0007 | Monotonic increase, saturates quickly (<100 forks) | Yes |
| commits | 0.092 | 0.111 | +0.0000 | Inverted-U: rises sharply to a peak (~0.43) near 100-300 commits, then declines | No |
| maintainers | 0.114 | 0.420 | +0.0099 | Sharp rise 0->~7 maintainers, then plateau to 31 | Yes |

(`endpoints_ci_nonoverlapping` = whether the bootstrap CIs at the 5th and 95th
percentile dose levels are non-overlapping — a rough indicator that the dose-response
function has a distinguishable overall change across its observed range. It does
*not* by itself indicate the curve is monotonic or that any single point differs from
zero.)

## Interpretation

- **num_dependencies, maintainers, forks**: all show a similar pattern — a steep
  increase in predicted MTTR over the low range of the dose (0 to ~5-7), then a
  plateau. Packages/repos with essentially zero dependencies, zero forks, or a
  single maintainer have the lowest predicted MTTR; beyond a small handful, more
  doesn't predict materially higher MTTR. This is consistent with "having any
  meaningful amount of activity/complexity" being associated with non-trivial MTTR,
  but further increases not mattering much.
- **num_floating_deps and stars**: both show smooth monotonic *declines* — more
  floating (unpinned) dependencies and more popular repos (by stars) are associated
  with *lower* predicted MTTR across their observed ranges. This is the opposite
  direction one might naively expect for floating dependencies (often considered a
  riskier practice), and may reflect that more actively-maintained/popular projects
  use floating deps as a deliberate fast-update strategy and also remediate faster
  for unrelated reasons (resourcing, visibility).
- **release_frequency**: U-shaped — both very infrequent and very frequent releasers
  have higher predicted MTTR than repos releasing every ~250-300 days. This is the
  only clearly non-monotonic, "Goldilocks"-style relationship among the 8 variables.
- **num_pinned_deps and commits**: both non-monotonic with similar endpoint values
  (CIs overlap at the extremes), but with a pronounced hump in the middle of the
  dose range (peaking around 5 pinned deps / 100-300 commits). The overall p5->p95
  change is not statistically distinguishable, but the within-range hump is
  pronounced and its CI band excludes the endpoint values — i.e., there may be a
  real "mid-range" effect that a single linear slope (as in the panel-FE analysis)
  would completely miss.

## Comparison to the panel-FE associational results

The earlier panel fixed-effects regressions (`data/did_analysis/RESULTS.md`) found
essentially flat, near-zero linear coefficients for all of these variables (with
`forks`/`stars` only marginally significant and tiny in magnitude). The GPS dose-
response analysis tells a richer story for the *same* variables:

- The panel-FE null result for `num_dependencies`, `maintainers`, and `forks` masks a
  real but **concave/saturating** relationship — most of the "effect" happens at the
  low end of the dose range and a linear term averages it away.
- `release_frequency`'s near-zero linear panel-FE coefficient is consistent with its
  **U-shaped** dose-response curve here — a linear fit through a U-shape is
  approximately flat.
- `num_floating_deps` and `stars` showing a clear *negative* dose-response here,
  versus near-zero panel-FE coefficients, may reflect that the panel-FE model's
  entity fixed effects absorb most of the cross-sectional variation that the
  (cross-sectional) GPS model is picking up — i.e., this may be more of a
  **between-repo** than a **within-repo** relationship.

## Caveats

- This is a **cross-sectional** analysis (no repo/time fixed effects) — unlike the
  panel-FE results, between-repo confounding is addressed only via the GPS/confounder
  adjustment, not via entity fixed effects. Apparent effects (especially for
  `num_floating_deps` and `stars`) may partly reflect unobserved between-repo
  differences not captured by the included confounders.
- GPS estimation assumes the dose, conditional on confounders, is approximately
  normally distributed (after log1p for skewed counts) — this is an approximation,
  particularly in the tails of the dose distributions (e.g., `commits` ranges from 0
  to 2.5M).
- The Hirano-Imbens flexible outcome specification (`dose + dose^2 + GPS + GPS^2 +
  dose*GPS`) constrains the ADRF shape to what this polynomial can express; the
  observed non-monotonic shapes (U-shape, inverted-U, hump) are real features of this
  specification but more flexible (e.g., spline-based) outcome models could reveal
  additional structure.
- `mttr` is zero for the majority of observations (median = 0, mean = 0.38); the OLS
  outcome model treats this as a continuous outcome rather than using a
  zero-inflated/hurdle specification.

## Reproduction

```
python code/dose_response/00_prepare_data.py        # -> data/dose_response/data.csv
python code/dose_response/01_gps_dose_response.py   # -> {var}_adrf.{csv,png}, gps_dose_response_summary.csv
```
