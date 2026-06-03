"""Script 4: MTTU vs MTTR comparison — scatter and time-series."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy import stats

from utils import load_results, setup_style, PALETTE

setup_style()
os.makedirs("figures", exist_ok=True)

df = load_results()

# Restrict to releases with non-zero MTTR
sub = df[df["mttr_all_severities"] > 0].copy()
print(f"Releases with non-zero MTTR: {len(sub):,}")

# --- Spearman correlation ---
rho, p_val = stats.spearmanr(sub["mttu"], sub["mttr_all_severities"])
print(f"Spearman correlation (MTTU vs MTTR): ρ={rho:.4f}, p={p_val:.3e}")

# --- Figure 1: Scatter with KDE ---
cap_mttu = sub["mttu"].quantile(0.95)
cap_mttr = sub["mttr_all_severities"].quantile(0.95)
plot_sub = sub[(sub["mttu"] <= cap_mttu) & (sub["mttr_all_severities"] <= cap_mttr)]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(plot_sub["mttu"], plot_sub["mttr_all_severities"],
           alpha=0.15, s=8, color="#607D8B", rasterized=True, label="Release")

# KDE contours
try:
    from scipy.stats import gaussian_kde
    xy = np.vstack([plot_sub["mttu"], plot_sub["mttr_all_severities"]])
    if xy.shape[1] > 100:
        kde = gaussian_kde(xy)
        xi = np.linspace(0, cap_mttu, 100)
        yi = np.linspace(0, cap_mttr, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        ax.contour(Xi, Yi, Zi, levels=5, colors=["#263238"], linewidths=0.7, alpha=0.6)
except Exception:
    pass

# y=x line
max_val = min(cap_mttu, cap_mttr)
ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y = x")
ax.text(0.97, 0.03,
        f"Spearman ρ={rho:.3f}\np={p_val:.2e}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))

ax.set_xlabel(f"MTTU (days, capped at 95th pct: {cap_mttu:.0f}d)")
ax.set_ylabel(f"MTTR (days, capped at 95th pct: {cap_mttr:.0f}d)")
ax.set_title("MTTU vs MTTR per release\n(non-zero MTTR releases only)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("figures/04_mttu_vs_mttr_scatter.pdf")
plt.close()
print("Saved figures/04_mttu_vs_mttr_scatter.pdf")

# --- Figure 2: Dual time-series of yearly medians ---
yearly_mttu = df.groupby("release_year")["mttu"].median()
yearly_mttr = df[df["mttr_all_severities"] > 0].groupby("release_year")["mttr_all_severities"].median()

common_years = sorted(set(yearly_mttu.index) & set(yearly_mttr.index))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(common_years, yearly_mttu.reindex(common_years), color=PALETTE["mttu"],
        linewidth=2, marker="o", markersize=4, label="MTTU (all releases)")
ax.plot(common_years, yearly_mttr.reindex(common_years), color=PALETTE["mttr"],
        linewidth=2, marker="s", markersize=4, label="MTTR (non-zero releases)")

ax.set_xlabel("Release year")
ax.set_ylabel("Median (days)")
ax.set_title("MTTU vs MTTR: yearly median comparison")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
plt.tight_layout()
plt.savefig("figures/04_mttu_vs_mttr_timeseries.pdf")
plt.close()
print("Saved figures/04_mttu_vs_mttr_timeseries.pdf")
