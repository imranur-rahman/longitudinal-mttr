"""Script 5: Cohort analysis — MTTU and MTTR by package cohort and age."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy import stats

from utils import load_results, setup_style, assign_cohort, mann_kendall_summary, COHORT_COLORS

setup_style()
os.makedirs("figures", exist_ok=True)

df = load_results()
df = assign_cohort(df)

# Compute package age in years (integer)
first_release = df.groupby("package_name")["package_release_date"].min().rename("first_release")
df = df.join(first_release, on="package_name")
df["age_years"] = ((df["package_release_date"] - df["first_release"]).dt.days / 365.25).astype(int)

cohorts = ["pre-2015", "2015–2018", "2019–2021", "2022+"]
MAX_AGE = 10  # plot up to 10 years of age

print("=" * 60)
print("COHORT ANALYSIS")
print("=" * 60)


def plot_cohort_trend(df, col, label, outpath, title, nonzero_only=False):
    if nonzero_only:
        plot_df = df[df[col] > 0].copy()
    else:
        plot_df = df.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    mk_results = {}

    for i, cohort in enumerate(cohorts):
        sub = plot_df[plot_df["cohort"] == cohort]
        if sub.empty:
            continue
        age_median = sub.groupby("age_years")[col].median()
        age_n = sub.groupby("age_years")[col].count()
        # Keep age bins with ≥5 packages
        valid_ages = age_n[age_n >= 5].index
        age_median = age_median.reindex(range(0, MAX_AGE + 1)).dropna()
        age_median = age_median[age_median.index.isin(valid_ages)]

        if len(age_median) < 2:
            continue

        color = COHORT_COLORS[i % len(COHORT_COLORS)]
        ax.plot(age_median.index, age_median.values, color=color, linewidth=2,
                marker="o", markersize=4, label=cohort)

        mk = mann_kendall_summary(age_median)
        mk_results[cohort] = mk
        print(f"  {cohort} [{label}]: trend={mk['trend']}, τ={mk['tau']}, p={mk['p_value']:.3e}")

    ax.set_xlabel("Package age (years since first release)")
    ax.set_ylabel(f"Median {label} (days)")
    ax.set_title(title)
    ax.set_xlim(0, MAX_AGE)
    ax.legend(title="Cohort")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

    return mk_results


def kruskal_at_age(df, col, ages=[1, 2, 3]):
    print(f"\n  Kruskal-Wallis across cohorts at fixed ages [{col}]:")
    for age in ages:
        groups = [df[(df["cohort"] == c) & (df["age_years"] == age)][col].dropna().values
                  for c in cohorts]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) < 2:
            continue
        stat, p = stats.kruskal(*groups)
        print(f"    Age {age}: H={stat:.2f}, p={p:.3e}")


print("\n--- MTTU cohort trends ---")
plot_cohort_trend(
    df, "mttu", "MTTU",
    "figures/05_cohort_mttu_by_age.pdf",
    "MTTU by package cohort and age",
    nonzero_only=False,
)
kruskal_at_age(df, "mttu")

print("\n--- MTTR cohort trends ---")
plot_cohort_trend(
    df, "mttr_all_severities", "MTTR",
    "figures/05_cohort_mttr_by_age.pdf",
    "MTTR by package cohort and age (non-zero MTTR releases)",
    nonzero_only=True,
)
kruskal_at_age(df[df["mttr_all_severities"] > 0], "mttr_all_severities")
