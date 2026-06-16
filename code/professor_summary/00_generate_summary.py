"""
Generate slide-ready figures and a problem-statement Markdown for presentation
to a statistics professor.

Dataset: data/scorecard_with_longitudinal_metrics.csv filtered to packages where
at least one version has mttr_all_severities > 0.0 (70,953 rows, 1,867 packages,
1,839 matched GitHub repos).

Outputs (all in data/professor_summary/):
  fig1_mttr_distribution.png  — zero-inflation + nonzero distribution
  fig2_panel_structure.png    — releases-per-repo + spaghetti trajectories
  fig3_covariate_summary.png  — box plots of 8 continuous dose variables
  fig4_sca_cohorts.png        — SCA adoption cohort breakdown
  fig5_mttr_by_severity.png   — MTTR by vulnerability severity
  PROBLEM_STATEMENT.md        — concise written summary for the professor
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})

DATA_DIR = Path("data")
IN_CSV = DATA_DIR / "scorecard_with_longitudinal_metrics.csv"
OUT_DIR = DATA_DIR / "professor_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def strip_gh_prefix(url: str) -> str:
    return re.sub(r"^https?://github\.com/", "", str(url).strip().rstrip("/"))


# ── load & filter ─────────────────────────────────────────────────────────────
print("Loading data …")
raw = pd.read_csv(IN_CSV, low_memory=False)
print(f"  total rows: {len(raw):,}")

pkg_max = raw.groupby(["ecosystem", "package_name"])["mttr_all_severities"].max()
qualifying = pkg_max[pkg_max > 0].reset_index()[["ecosystem", "package_name"]]
df = raw.merge(qualifying, on=["ecosystem", "package_name"])
print(f"  after mttr>0 package filter: {len(df):,} rows, "
      f"{df[['ecosystem','package_name']].drop_duplicates().shape[0]:,} packages")

matched = df[df["match_method"] != "no_match"].copy()
matched["github_repo"] = matched["github_repo"].apply(strip_gh_prefix)
print(f"  matched rows: {len(matched):,}  unique repos: {matched['github_repo'].nunique():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — MTTR distribution (zero-inflation + nonzero)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFig 1: MTTR distribution …")
mttr = df["mttr_all_severities"]
nonzero = mttr[mttr > 0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: zero vs. >0 bar chart
ax = axes[0]
counts = [int((mttr == 0).sum()), int((mttr > 0).sum())]
bars = ax.bar(["= 0 (no vuln in window\nor fixed immediately)", "> 0 (remediation\ntook > 0 days)"],
               counts, color=["#aec6e8", "#1f77b4"], edgecolor="white", width=0.5)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{c:,}\n({c/len(mttr)*100:.0f}%)", ha="center", va="bottom", fontsize=12)
ax.set_ylabel("Number of package-version rows")
ax.set_title("MTTR = 0 vs. > 0\n(packages with any mttr > 0)")
ax.set_ylim(0, max(counts) * 1.18)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Right: nonzero MTTR distribution (log scale)
ax = axes[1]
log_vals = np.log10(nonzero + 1e-6)
ax.hist(log_vals, bins=60, color="#1f77b4", edgecolor="none")
med = np.log10(nonzero.median())
p75 = np.log10(nonzero.quantile(0.75))
ax.axvline(med, color="red", linestyle="--", linewidth=1.8,
           label=f"Median = {nonzero.median():.2f} days")
ax.axvline(p75, color="orange", linestyle="--", linewidth=1.8,
           label=f"p75 = {nonzero.quantile(0.75):.2f} days")
ax.set_xlabel("log₁₀(MTTR in days)")
ax.set_ylabel("Count")
ax.set_title(f"Non-zero MTTR distribution\n(n = {len(nonzero):,}; max = {nonzero.max():.1f} days)")
ax.legend()

fig.suptitle("Outcome variable: Mean Time to Remediate (MTTR) — all severities", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_mttr_distribution.png", bbox_inches="tight")
plt.close(fig)
print("  saved fig1_mttr_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Panel structure
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Panel structure …")
releases_per_repo = matched.groupby(["ecosystem", "github_repo"]).size()
matched_sorted = matched.sort_values(["ecosystem", "github_repo", "published_at"])
matched_sorted["period"] = matched_sorted.groupby(["ecosystem", "github_repo"]).cumcount() + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of releases per repo (truncated at 100)
ax = axes[0]
trunc = releases_per_repo.clip(upper=100)
ax.hist(trunc, bins=40, color="#1f77b4", edgecolor="none")
ax.axvline(releases_per_repo.median(), color="red", linestyle="--", linewidth=1.8,
           label=f"Median = {releases_per_repo.median():.0f}")
ax.axvline(releases_per_repo.quantile(0.75), color="orange", linestyle="--", linewidth=1.8,
           label=f"p75 = {releases_per_repo.quantile(0.75):.0f}")
ax.set_xlabel("Releases per repo (capped at 100)")
ax.set_ylabel("Number of repos")
ax.set_title(f"Releases per repo (n = {len(releases_per_repo):,} repos)\nmax = {releases_per_repo.max():,} (right tail not shown)")
ax.legend()

# Right: spaghetti plot — sample 30 repos, median trajectory
ax = axes[1]
rng = np.random.default_rng(42)
repo_ids = matched_sorted.groupby(["ecosystem", "github_repo"]).filter(
    lambda g: 5 <= len(g) <= 50
)
unique_repos = repo_ids[["ecosystem", "github_repo"]].drop_duplicates()
sample_repos = unique_repos.sample(min(30, len(unique_repos)), random_state=42)
sample_data = matched_sorted.merge(sample_repos, on=["ecosystem", "github_repo"])
for (eco, repo), grp in sample_data.groupby(["ecosystem", "github_repo"]):
    ax.plot(grp["period"], grp["mttr_all_severities"], alpha=0.25, linewidth=0.8,
            color="#1f77b4")
med_traj = matched_sorted.groupby("period")["mttr_all_severities"].median()
ax.plot(med_traj.index, med_traj.values, color="red", linewidth=2.5,
        label="Median across repos")
ax.set_xlim(0, 51)
ax.set_ylim(-0.5, 15)
ax.set_xlabel("Release sequence number (period)")
ax.set_ylabel("MTTR (days)")
ax.set_title("MTTR trajectories: 30 sampled repos\n(5–50 releases; median in red)")
ax.legend()

fig.suptitle("Panel structure: longitudinal package-version data", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_panel_structure.png", bbox_inches="tight")
plt.close(fig)
print("  saved fig2_panel_structure.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Covariate summary (box plots, log1p scale)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Covariate summary …")
covar_cols = {
    "num_dependencies": "# dependencies",
    "pins": "# pinned deps",
    "floats": "# floating deps",
    "release_frequency": "release frequency\n(days/release)",
    "stars": "GitHub stars",
    "forks": "GitHub forks",
    "commits_approx": "commits (approx.)",
    "contributors_approx": "maintainers (approx.)",
}
cov_data = {}
for col, label in covar_cols.items():
    vals = matched[col].dropna()
    cov_data[label] = np.log1p(vals.values)

fig, ax = plt.subplots(figsize=(10, 6))
labels = list(cov_data.keys())
data = [cov_data[l] for l in labels]
bp = ax.boxplot(data, vert=False, patch_artist=True,
                boxprops=dict(facecolor="#aec6e8", color="#1f77b4"),
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
                whiskerprops=dict(color="#1f77b4"),
                capprops=dict(color="#1f77b4"))
ax.set_yticks(range(1, len(labels) + 1))
ax.set_yticklabels(labels)
ax.set_xlabel("log₁₀(1 + value)  [natural log scale]")
ax.set_title("Continuous exposure variables — distribution\n(matched repos, mttr>0 packages; red line = median)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_covariate_summary.png", bbox_inches="tight")
plt.close(fig)
print("  saved fig3_covariate_summary.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — SCA cohort breakdown
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4: SCA cohorts …")

def sca_used_val(score):
    if score == 10.0:
        return 1
    elif score == 0.0:
        return 0
    return np.nan

matched["sca_used"] = matched["Dependency-Update-Tool_score"].map({10.0: 1, 0.0: 0})

def assign_status(g):
    s = g.dropna()
    if s.empty:
        return "unusable"
    if (s == 0).all():
        return "never-treated"
    if (s == 1).all():
        return "always-treated"
    if s.iloc[0] == 0:
        return "switcher"
    return "always-treated"

sca_status = (
    matched.sort_values("published_at")
    .groupby(["ecosystem", "github_repo"])["sca_used"]
    .apply(assign_status)
    .value_counts()
)

fig, ax = plt.subplots(figsize=(8, 5))
colors = {"never-treated": "#aec6e8", "always-treated": "#1f77b4",
          "switcher": "#ff7f0e", "unusable": "#d3d3d3"}
bars = ax.barh(sca_status.index, sca_status.values,
               color=[colors.get(s, "#999") for s in sca_status.index],
               edgecolor="white")
for bar, v in zip(bars, sca_status.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"{v:,}", va="center", fontsize=12)
ax.set_xlabel("Number of repos")
ax.set_title("SCA adoption cohorts\n(matched repos, mttr>0 packages)\nOnly 'switchers' are usable for staggered DiD")
ax.set_xlim(0, sca_status.max() * 1.15)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_sca_cohorts.png", bbox_inches="tight")
plt.close(fig)
print("  saved fig4_sca_cohorts.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — MTTR by severity
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5: MTTR by severity …")
severity_cols = {
    "mttr_critical": "Critical",
    "mttr_high": "High",
    "mttr_medium": "Medium",
    "mttr_low": "Low",
    "mttr_all_severities": "All severities",
}

fig, ax = plt.subplots(figsize=(11, 5))
sev_data, sev_labels = [], []
for col, label in severity_cols.items():
    vals = df[df[col] > 0][col]
    sev_data.append(np.log10(vals.values))
    sev_labels.append(f"{label}\n(n={len(vals):,})")

bp = ax.boxplot(sev_data, patch_artist=True,
                boxprops=dict(facecolor="#aec6e8", color="#1f77b4"),
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
                whiskerprops=dict(color="#1f77b4"),
                capprops=dict(color="#1f77b4"))
ax.set_xticklabels(sev_labels)
ax.set_ylabel("log₁₀(MTTR in days)\n[nonzero values only]")
ax.set_title("MTTR distribution by vulnerability severity\n(red line = median)")
yticks = [-1, 0, 1, 2]
ax.set_yticks(yticks)
ax.set_yticklabels([f"10^{t} = {10**t:.1g}d" for t in yticks])
fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_mttr_by_severity.png", bbox_inches="tight")
plt.close(fig)
print("  saved fig5_mttr_by_severity.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Horizontal violin plot of mttr_all_severities (nonzero, raw values)
#          split by ecosystem
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 6: MTTR violin plot …")

import seaborn as sns

CLIP_DAYS = 20   # p99 ≈ 19.7 days; clip display here, annotate outliers

vdf = df[df["mttr_all_severities"] > 0][["ecosystem", "mttr_all_severities"]].copy()
n_clipped = int((vdf["mttr_all_severities"] > CLIP_DAYS).sum())
pct_clipped = n_clipped / len(vdf) * 100

fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(
    data=vdf, x="mttr_all_severities", y="ecosystem",
    order=["npm", "pypi"],
    palette={"npm": "#1f77b4", "pypi": "#ff7f0e"},
    inner="quartile", cut=0, linewidth=0.9,
    ax=ax,
)
ax.set_xlim(0, CLIP_DAYS)
ax.set_xlabel("MTTR — all severities (days, raw values)\n[non-zero values only]")
ax.set_ylabel("Ecosystem")
ax.set_title(
    "Distribution of MTTR (mttr_all_severities) by ecosystem\n"
    f"(non-zero values; display clipped at {CLIP_DAYS} days — "
    f"{n_clipped:,} rows ({pct_clipped:.1f}%) beyond clip not shown)"
)
ax.axvline(vdf["mttr_all_severities"].median(), color="red", linestyle="--",
           linewidth=1.5, label=f"Overall median = {vdf['mttr_all_severities'].median():.2f} d")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig6_mttr_violin.png", bbox_inches="tight")
plt.close(fig)
print("  saved fig6_mttr_violin.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Problem statement Markdown
# ═══════════════════════════════════════════════════════════════════════════════
print("\nWriting PROBLEM_STATEMENT.md …")

n_rows = len(df)
n_pkgs = df[["ecosystem", "package_name"]].drop_duplicates().shape[0]
n_repos = matched["github_repo"].nunique()
n_zero = int((df["mttr_all_severities"] == 0).sum())
n_nonzero = int((df["mttr_all_severities"] > 0).sum())
pct_zero = n_zero / n_rows * 100
nonzero_med = df.loc[df["mttr_all_severities"] > 0, "mttr_all_severities"].median()
nonzero_p75 = df.loc[df["mttr_all_severities"] > 0, "mttr_all_severities"].quantile(0.75)
nonzero_max = df["mttr_all_severities"].max()
n_switchers = int((sca_status.get("switcher", 0)))
n_never = int(sca_status.get("never-treated", 0))
n_always = int(sca_status.get("always-treated", 0))

md = f"""# Problem Statement: Statistical Methods for Longitudinal Vulnerability Remediation Analysis

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
- Rows: {n_rows:,}
- Unique packages: {n_pkgs:,}
- Unique GitHub repos: {n_repos:,}
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
  time and took MTTR days to ship a fix. Among the {n_nonzero:,} such rows:
  median = {nonzero_med:.2f} days, p75 = {nonzero_p75:.2f} days, max = {nonzero_max:.1f} days.
- Even within the sample filtered to *packages with any version MTTR > 0*, **{n_zero:,}
  individual version rows ({pct_zero:.0f}%) still have MTTR = 0** — because not every
  version of an affected package coincided with an open vulnerability window.
- The distribution is **heavily zero-inflated and right-skewed** (see Figs. 1 & 6).
- Separate MTTR values are available by vulnerability severity (critical, high, medium,
  low); medians among non-zero rows range from 0.27 days (low) to 0.51 days (medium).

## Independent Variables

### Binary treatment: SCA tool adoption (`sca_used`)
Derived from the OpenSSF Scorecard `Dependency-Update-Tool` check (score 0–10):
`sca_used = 1` if score = 10 (tool detected, e.g. Dependabot or Renovate),
`sca_used = 0` if score = 0 (no tool detected), missing if score = −1 or NaN.
Cohort breakdown (matched repos, mttr>0 packages): {n_never:,} never-treated,
{n_always:,} always-treated, {n_switchers:,} switchers (adopt during the window),
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
Units filtered to repos with any MTTR > 0 and period ≤ 50 (807 units: {n_never:,}
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
"""

(OUT_DIR / "PROBLEM_STATEMENT.md").write_text(md)
print("  saved PROBLEM_STATEMENT.md")
print("\nAll outputs saved to", OUT_DIR)
