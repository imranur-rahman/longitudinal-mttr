"""Script 3: MTTR (primary) and MTTU distribution evolution by year."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from scipy import stats
from itertools import combinations

from utils import load_results, setup_style, PALETTE

setup_style()
os.makedirs("figures", exist_ok=True)

df = load_results()

EPOCH_PAIRS = [
    (2010, 2015),
    (2015, 2020),
    (2020, 2024),
]


def rank_biserial(x, y):
    """Rank-biserial correlation as effect size for Wilcoxon rank-sum."""
    n1, n2 = len(x), len(y)
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 1 - (2 * u_stat) / (n1 * n2)


def plot_violin_by_year(df_metric, col, label, color, outpath, title):
    years = sorted(df_metric["release_year"].unique())
    cap = df_metric[col].quantile(0.95)
    plot_df = df_metric[["release_year", col]].copy()
    plot_df[col] = plot_df[col].clip(upper=cap)

    # Keep only years with ≥ 20 data points for readable violins
    year_counts = plot_df.groupby("release_year")[col].count()
    valid_years = year_counts[year_counts >= 20].index.tolist()
    plot_df = plot_df[plot_df["release_year"].isin(valid_years)]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(
        data=plot_df, x="release_year", y=col, ax=ax,
        color=color, linewidth=0.8, inner="quartile", cut=0,
    )
    # Overlay median dots
    medians = plot_df.groupby("release_year")[col].median()
    x_positions = {yr: i for i, yr in enumerate(valid_years)}
    for yr, med in medians.items():
        ax.scatter(x_positions[yr], med, color="white", s=20, zorder=5)

    ax.set_xlabel("Release year")
    ax.set_ylabel(f"{label} (days, capped at 95th pct: {cap:.0f}d)")
    ax.set_title(title)
    ax.set_xticks(range(len(valid_years)))
    ax.set_xticklabels(valid_years, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")


def statistical_tests(df_metric, col, label):
    print(f"\n--- {label} statistical tests ---")
    year_groups = {yr: grp[col].values for yr, grp in df_metric.groupby("release_year") if len(grp) >= 5}
    if len(year_groups) < 2:
        print("  Insufficient data for tests.")
        return

    # Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(*year_groups.values())
    print(f"  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.3e}")

    # Pairwise Wilcoxon (Bonferroni)
    print(f"\n  Pairwise Wilcoxon rank-sum (Bonferroni corrected, selected pairs):")
    n_comparisons = len(EPOCH_PAIRS)
    for y1, y2 in EPOCH_PAIRS:
        if y1 not in year_groups or y2 not in year_groups:
            continue
        x1, x2 = year_groups[y1], year_groups[y2]
        u_stat, p_raw = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        p_adj = min(p_raw * n_comparisons, 1.0)
        rb = rank_biserial(x1, x2)
        sig = "*" if p_adj < 0.05 else "ns"
        print(f"    {y1} vs {y2}: p_adj={p_adj:.3e} [{sig}], effect size r={rb:.3f}, "
              f"median {y1}={np.median(x1):.1f}d, {y2}={np.median(x2):.1f}d")


# --- MTTR (primary) ---
mttr_df = df[df["mttr_all_severities"] > 0].copy()
print(f"MTTR non-zero releases: {len(mttr_df):,}")

plot_violin_by_year(
    mttr_df, "mttr_all_severities", "MTTR", PALETTE["mttr"],
    "figures/03_mttr_violin_by_year.pdf",
    "Distribution of MTTR by release year (non-zero releases only)",
)
statistical_tests(mttr_df, "mttr_all_severities", "MTTR")

# --- MTTU (companion) — restrict to non-zero to avoid degenerate violins pre-2017 ---
mttu_df = df[df["mttu"] > 0].copy()
print(f"\nMTTU non-zero releases: {len(mttu_df):,}")
plot_violin_by_year(
    mttu_df, "mttu", "MTTU", PALETTE["mttu"],
    "figures/03_mttu_violin_by_year.pdf",
    "Distribution of MTTU by release year (non-zero releases only)",
)
statistical_tests(mttu_df, "mttu", "MTTU")
