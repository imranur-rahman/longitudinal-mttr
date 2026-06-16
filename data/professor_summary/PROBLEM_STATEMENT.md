# Problem Statement: Statistical Methods for Longitudinal Vulnerability Remediation Analysis

## Research Question

Does adopting a Software Composition Analysis (SCA) / dependency-update tool reduce
the **Mean Time to Remediate (MTTR)** known vulnerabilities in open-source packages?
More broadly: which repository and package characteristics — dependency structure,
release cadence, popularity, team size — are causally associated with faster
vulnerability remediation?

## Data

**Sources:** OpenSSF Scorecard (security practice scores for GitHub repos),
deps.dev (per-release dependency graphs and vulnerability exposure windows),
GH Archive / BigQuery (repository activity metrics: stars, forks, commits, contributors).

**Collection window:** October 6, 2023 – October 6, 2025.

**Unit of observation:** one row = one package version release. Each row records
the MTTR for that version (the time from the version's release until the first
subsequent release that resolved any open vulnerabilities), along with Scorecard
checks and repo activity metrics at that point in time.

**Sample (after filtering to packages with any version MTTR > 0):**
- Rows: 70,953
- Unique packages: 1,867
- Unique GitHub repos: 1,839
- Ecosystems: npm (66%), PyPI (34%)

## Outcome: MTTR

MTTR is measured in **days** for each package-version row:
- **18,297 rows (26%) have MTTR = 0** (the package-version had no
  open vulnerabilities in its window, or a fix appeared in the same release).
- Among the 52,656 non-zero rows: median = 0.43 days,
  p75 = 1.58 days, max = 253.8 days.
- The distribution is **heavily zero-inflated and right-skewed** (see Fig. 1).
- Separate MTTR values are available by severity (critical, high, medium, low);
  medians range from 0.27 days (low severity) to 0.51 days (medium severity)
  among non-zero observations.

## Exposures / Treatments

**Binary (staggered over time):**
- `sca_used`: whether the repo's Scorecard `Dependency-Update-Tool` check passes
  (score = 10), indicating use of an automated dependency-update tool (e.g.,
  Dependabot, Renovate). Derived from the Scorecard score: 1 if score = 10,
  0 if score = 0, missing otherwise.
- Cohort breakdown (matched repos): 713 never-treated, 1,021 always-treated,
  102 switchers (adopt during observation window), unusable (score always missing).

**Continuous (8 variables):**
num_dependencies, num_pinned_deps, num_floating_deps, release_frequency (days/release),
GitHub stars, forks, commits (approx.), and maintainers (approx. contributor count).
All are available at the package-version level (time-varying). Pins/floats/release
frequency are ~80% complete; the rest are 99–100% complete.

## Panel Structure

Observations are **repeated per (ecosystem, GitHub repo)**: each repo contributes
a median of 10 rows (p75 = 26, max = 1,624), one per version release.

**Key challenge:** release timing is irregular (some repos release weekly, others
annually), so the natural time index is a **release-sequence number** (1, 2, 3, …)
rather than calendar time. This affects the validity of time-fixed-effect and
staggered DiD estimators.

## Methods Tried So Far

### 1. Staggered Difference-in-Differences (Callaway & Sant'Anna 2021)
*For the binary SCA adoption treatment.*
Units filtered to repos with any MTTR > 0 and period ≤ 50 (807 units: 713
never-treated, 96 switchers; 10,299 observations after dropping always-treated and
missing covariates). Outcome: MTTR. Covariates: all 8 continuous exposures.

- Overall ATT = **0.555 days** (SE = 0.848; 95% CI [−1.11, 2.22]; **not significant**)
- Event-study ATT = **0.861 days** (SE = 0.663; 95% CI [−0.44, 2.16]; **not significant**)
- No evidence of pre-trend violation in the event-study plot.

### 2. Panel Fixed-Effects OLS (linearmodels.PanelOLS)
*For continuous exposures; entity + time FE, clustered SEs by repo.*
Full panel: 247,342 rows, 13,938 repos.

| Variable | β (univariate) | p | β (combined) | p |
|---|---|---|---|---|
| release_frequency | −0.000030 | **0.033** | −0.000040 | 0.087 |
| forks | −0.000020 | 0.089 | −0.000060 | **0.001** |
| stars | −0.000001 | 0.982 | +0.000010 | **0.001** |
| sca_used | +0.033 | 0.093 | +0.036 | 0.062 |
| all others | ≈ 0 | > 0.2 | ≈ 0 | > 0.2 |

Effect sizes are negligibly small despite some statistical significance at large n.

### 3. GPS Continuous Dose-Response (Hirano & Imbens 2004 ADRF)
*For continuous exposures; cross-sectional, 209,924 rows.*
Reveals non-linear shapes invisible to a single linear coefficient:
- **Saturating increases**: num_dependencies, maintainers, forks (sharp rise at
  low dose, then plateau — linear OLS averages this to ~0).
- **Monotonic declines**: num_floating_deps (↓ MTTR as more floating deps), stars (↓ MTTR).
- **U-shaped**: release_frequency (MTTR lowest near median releasers).
- **Non-monotonic humps**: num_pinned_deps, commits (peak mid-range; endpoint CIs overlap).

## Candidate Next Method: Poisson Pseudo Maximum Likelihood (PPML)

Santos Silva & Tenreyro (2006) showed that PPML (a Poisson log-linear regression)
is consistent for non-negative, zero-inflated outcomes without requiring
log-transformation (which would exclude the zero rows). It is also heteroskedasticity-
robust and extends to panel settings with high-dimensional fixed effects. We are
considering applying PPML in place of OLS in both the panel FE and GPS outcome steps.

## Open Questions for the Professor

1. **Zero-inflated outcome**: OLS is mis-specified for a 74%-zero outcome. Is PPML
   the right fix, or would a two-part (hurdle) model or Tobit be preferred?
2. **Irregular timing / release-sequence index**: Is the staggered DiD estimator
   (Callaway & Sant'Anna) valid when "time" = release-sequence number rather than
   calendar time? What's the right time unit here?
3. **Low switcher count**: With only 96 repos in the "switcher" cohort, is the
   staggered DiD likely underpowered? What's the right power-analysis approach?
4. **Cross-sectional GPS vs. panel GPS**: The GPS model currently ignores the panel
   structure (no entity FE). Is there a feasible panel-based extension, or a better
   continuous-treatment causal method for this design?
5. **Effect size interpretability**: ATT ≈ 0.55 days and panel-FE β < 0.001 per
   unit — are these effect sizes practically negligible, or could they be masked by
   OLS mis-specification on a zero-inflated, right-skewed outcome?
