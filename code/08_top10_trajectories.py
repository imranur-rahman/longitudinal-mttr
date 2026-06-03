"""Script 8: Individual MTTR trajectories for top-10 improving and worsening packages."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy import stats

from utils import load_results, setup_style, PALETTE

setup_style()
os.makedirs("figures", exist_ok=True)

COL = "mttr_all_severities"

# ── Load & classify ───────────────────────────────────────────────────────────
df = load_results()
df = df.sort_values(["package_name", "package_release_date"])

records = []
for pkg, grp in df[df[COL] > 0].groupby("package_name"):
    grp = grp.sort_values("package_release_date")
    vals = grp[COL].values
    dates = grp["package_release_date"].values
    n = len(vals)
    if n < 2:
        continue
    x = np.arange(n)
    slope, intercept, _, p, _ = stats.linregress(x, vals)
    records.append({
        "package_name": pkg,
        "slope": slope,
        "intercept": intercept,
        "p_value": p,
        "n_releases": n,
        "dates": dates,
        "vals": vals,
    })

res = pd.DataFrame([{k: v for k, v in r.items() if k not in ("dates", "vals")}
                    for r in records])
# Keep raw dates/vals in a lookup dict
pkg_data = {r["package_name"]: (r["dates"], r["vals"]) for r in records}

improving = res[res["slope"] < 0].nsmallest(10, "slope").reset_index(drop=True)
worsening = res[res["slope"] > 0].nlargest(10, "slope").reset_index(drop=True)

print("TOP 10 IMPROVING (by slope):")
for _, row in improving.iterrows():
    print(f"  {row['package_name']:40s}  slope={row['slope']:+.2f}  p={row['p_value']:.2e}  n={row['n_releases']}")

print("\nTOP 10 WORSENING (by slope):")
for _, row in worsening.iterrows():
    print(f"  {row['package_name']:40s}  slope={row['slope']:+.2f}  p={row['p_value']:.2e}  n={row['n_releases']}")


# ── Plotting helper ───────────────────────────────────────────────────────────
def plot_trajectories(top10_df, color, title, outpath):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes_flat = axes.flatten()

    for i, (_, row) in enumerate(top10_df.iterrows()):
        ax = axes_flat[i]
        pkg = row["package_name"]
        dates, vals = pkg_data[pkg]
        n = len(vals)

        # Actual MTTR values — solid line + markers
        ax.plot(dates, vals, color=color, linewidth=1.8, alpha=0.9, zorder=3,
                marker="o", markersize=5, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.8, label="MTTR")

        # Regression line: y = intercept + slope * index, mapped to date axis
        y_start = row["intercept"]
        y_end   = row["intercept"] + row["slope"] * (n - 1)
        ax.plot([dates[0], dates[-1]], [y_start, y_end],
                color="black", linewidth=1.6, linestyle="--", zorder=4,
                alpha=0.85, label="OLS fit")

        # Annotation box
        ann = (f"slope={row['slope']:+.1f} d/rel\n"
               f"p={row['p_value']:.1e}  n={n}")
        ax.text(0.03, 0.97, ann, transform=ax.transAxes,
                va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#AAAAAA", alpha=0.9))

        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.set_title(pkg, fontsize=8, pad=3)
        ax.set_ylabel("MTTR (days)", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


plot_trajectories(
    improving, PALETTE["improving"],
    "Top 10 Improving packages — individual MTTR values",
    "figures/08_top10_improving.pdf",
)

plot_trajectories(
    worsening, PALETTE["worsening"],
    "Top 10 Worsening packages — individual MTTR values",
    "figures/08_top10_worsening.pdf",
)
