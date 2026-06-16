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

**MTTR (Mean Time to Remediate)** measures how fast a package updates its vulnerable
dependencies over the package's lifetime. For each package version, MTTR is the
number of **days** between the release of that version and the first subsequent
release in which all known vulnerable dependencies were resolved (updated or removed).

- **MTTR = 0.0** means the package version never had any vulnerable dependencies in
  its observation window — either no known vulnerabilities affected its dependencies,
  or the fix was already included in the same release. These are *not* "fast
  remediators"; they are packages that were not exposed.
- **MTTR > 0** means the package had at least one vulnerable dependency at release
  time and took MTTR days to ship a fix. Among the 52,656 such rows:
  median = 0.43 days, p75 = 1.58 days, max = 253.8 days.
- Even within the sample filtered to *packages with any version MTTR > 0*, **18,297
  individual version rows (26%) still have MTTR = 0** — because not every
  version of an affected package coincided with an open vulnerability window.
- The distribution is **heavily zero-inflated and right-skewed** (see Figs. 1 & 6).
- Separate MTTR values are available by vulnerability severity (critical, high, medium,
  low); medians among non-zero rows range from 0.27 days (low) to 0.51 days (medium).

## Independent Variables

### Binary treatment: SCA tool adoption (`sca_used`)
Derived from the OpenSSF Scorecard `Dependency-Update-Tool` check (score 0–10):
`sca_used = 1` if score = 10 (tool detected, e.g. Dependabot or Renovate),
`sca_used = 0` if score = 0 (no tool detected), missing if score = −1 or NaN.
Cohort breakdown (matched repos, mttr>0 packages): 713 never-treated,
1,021 always-treated, 102 switchers (adopt during the window),
3 unusable (always missing score).

### Continuous variables (8)

| Variable | Description | Scale | Completeness |
|---|---|---|---|
| `num_dependencies` | Total number of direct dependencies declared by the package version | Count | 100% |
| `num_pinned_deps` | Number of dependencies pinned to an exact version (reduces supply-chain drift) | Count | ~80% |
| `num_floating_deps` | Number of dependencies with a version range / unpinned (e.g. `^1.2.0`) | Count | ~80% |
| `release_frequency` | Average days between consecutive releases for this repo at time of this version | Days/release | ~80% |
| `stars` | GitHub repository star count at time of Scorecard run | Count | 100% |
| `forks` | GitHub repository fork count | Count | 100% |
| `commits` | Approximate total commit count for the repository | Count | 100% |
| `maintainers` | Approximate number of unique contributors to the repository | Count | 100% |

All 8 are **time-varying** (measured at each package version's release), making them
covariates in a longitudinal/panel setting rather than static repo attributes.
Highly right-skewed variables (all except `release_frequency`) were log₁₀(1+x)
transformed for the GPS dose-response analysis.

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
