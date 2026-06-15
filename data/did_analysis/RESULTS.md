# DiD / Panel FE Analysis: Effect of Repo Characteristics on MTTR

**Data:** `data/scorecard_with_longitudinal_metrics.csv` (321,083 rows), reshaped into a
per-(ecosystem, github_repo) release-sequence panel (`data/did_analysis/panel.csv`,
247,342 rows, 13,938 units). `period` = 1-indexed rank of each package version's
release date within its repo.

## 1. Staggered DiD: Effect of SCA adoption on MTTR

**Design:** Callaway & Sant'Anna (2021) group-time ATT estimator (`differences` package).
Treatment = a repo's Scorecard `Dependency-Update-Tool` check turning from 0 to 10
(SCA/dependency-update-tool adoption) at some release.

**Sample construction:**
- Restricted to units with at least one release where `mttr_all_severities > 0`
  (1,826 / 13,938 units; 53,754 rows) — needed both for computational tractability
  and to focus on repos where MTTR is actually observed.
- Capped the time index at `period <= 50` (median release count is ~10; only 106/1,826
  units exceed 100 releases). Without this cap, a handful of extreme outliers
  (e.g. `npm/cryptorubic/rubic-sdk` with 1,624 releases, `npm/wagmi-dev/viem` with 1,450)
  dominated the event-study aggregation, producing a spurious relative-period range of
  -509 to +74 and a meaningless overall estimate.
- Dropped "always-treated" (SCA on from period 1) and "unusable" (no valid SCA score)
  units, keeping "never-treated" and "switcher" units.
- Final sample: 807 units (711 never-treated, 96 switchers), 10,299 observations.

**Covariates (controls):** num_pinned_deps, num_floating_deps, num_dependencies,
version_releases, release_frequency, forks, stars, commits, maintainers.

**Results:**

| Estimate | ATT | Std. Error | 95% CI |
|---|---|---|---|
| Overall (simple aggregation) | 0.555 | 0.848 | [-1.11, 2.22] |
| Overall (event-study aggregation) | 0.861 | 0.663 | [-0.44, 2.16] |

Both estimates are **positive but not statistically significant** (CIs include 0).
The event-study plot (`sca_did_event_study.png`) shows ATT estimates fluctuating
around zero in both the pre- and post-adoption periods, with no clear discontinuity
at the adoption point — i.e., no evidence of an anticipation effect pre-adoption, and
no evidence of a treatment effect post-adoption.

**Conclusion:** No statistically significant evidence that adopting a Scorecard-detected
dependency-update tool changes mean time to remediate (MTTR) vulnerabilities.

## 2. Panel fixed-effects regressions

**Design:** `PanelOLS` with entity `(ecosystem, github_repo)` and time (`period`)
fixed effects, standard errors clustered by entity. Full panel (247,342 rows,
13,938 units; no mttr-positivity filter).

**Univariate models** (one covariate at a time):

| Variable | Coef. | Std. Error | p-value | n |
|---|---|---|---|---|
| num_pinned_deps | 0.00016 | 0.00046 | 0.733 | 209,924 |
| num_floating_deps | -0.00007 | 0.00032 | 0.828 | 209,924 |
| num_dependencies | -0.00078 | 0.00158 | 0.620 | 247,342 |
| version_releases | -0.00002 | 0.00004 | 0.659 | 209,924 |
| release_frequency | -0.00003 | 0.00002 | **0.033** | 209,924 |
| forks | -0.00002 | 0.00001 | 0.089 | 247,342 |
| stars | -0.00000 | 0.00000 | 0.982 | 247,342 |
| commits | -0.00000 | 0.00000 | 0.803 | 247,342 |
| maintainers | 0.00167 | 0.00143 | 0.245 | 247,342 |
| sca_used | 0.03314 | 0.01975 | 0.093 | 246,230 |

**Combined model** (all covariates together, n = 208,859):

| Variable | Coef. | Std. Error | p-value |
|---|---|---|---|
| num_pinned_deps | 0.00026 | – | 0.564 |
| num_floating_deps | -0.00005 | – | 0.869 |
| num_dependencies | -0.00067 | – | 0.731 |
| version_releases | -0.00002 | – | 0.698 |
| release_frequency | -0.00004 | – | 0.087 |
| forks | -0.00006 | – | **0.0008** |
| stars | 0.00001 | – | **0.0011** |
| commits | 0.00000 | – | 0.555 |
| maintainers | 0.00167 | – | 0.436 |
| sca_used | 0.03637 | – | 0.062 |

**Interpretation:**
- In isolation, `release_frequency` has a small but statistically significant
  negative association with MTTR (more frequent releases -> slightly faster
  remediation), but this weakens to p=0.087 once other covariates are controlled for.
- `forks` and `stars` are the only covariates significant in the combined model —
  `forks` negatively associated, `stars` positively associated with MTTR — but the
  coefficients are tiny (~0.00006 days per fork/star), so the practical effect size
  is negligible despite statistical significance (likely an artifact of the large n).
- `sca_used` is marginally positive (p~0.06-0.09) in both univariate and combined
  models, consistent with (but not confirming) the staggered DiD's null result.
- Dependency pinning/floating, dependency count, version-release count, commits, and
  maintainer count show no meaningful association with MTTR after fixed effects.

## Overall takeaway

Across both the causal (staggered DiD) and associational (panel FE) analyses, there is
**no robust evidence that any of the examined repo/package characteristics — including
SCA/dependency-update-tool adoption — meaningfully affect MTTR** once repo- and
release-sequence fixed effects are accounted for. The few nominally significant
coefficients (`release_frequency`, `forks`, `stars`) have effect sizes too small to be
practically meaningful.

## Caveats / robustness notes

- The `PERIOD_CAP = 50` choice for the staggered DiD is somewhat ad hoc; results should
  be checked for sensitivity to this cutoff (e.g., 30 or 100).
- The mttr-positivity filter for the DiD sample (1,826 units) is a large reduction from
  the full panel (13,938 units) and may introduce selection effects — the panel FE
  results (run on the full panel) are a useful cross-check but use a different sample.
- `sca_used` is derived from a single Scorecard check (`Dependency-Update-Tool_score`)
  evaluated at a single point in time per release; it is a coarse proxy for actual SCA
  practice and may not capture changes in tooling that occurred between scorecard runs.

## Reproduction

```
python code/did_analysis/00_prepare_panel.py        # -> data/did_analysis/panel.csv
python code/did_analysis/01_sca_staggered_did.py     # -> sca_did_simple_att.csv, sca_did_event_study.{csv,png}
python code/did_analysis/02_panel_fe_regressions.py  # -> panel_fe_results.csv
```
