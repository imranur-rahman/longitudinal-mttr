"""Script 1: Data characterization and coverage overview."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import load_results, setup_style, PALETTE

setup_style()
os.makedirs("figures", exist_ok=True)

df = load_results()

# --- Summary stats ---
print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"Total releases      : {len(df):,}")
print(f"Unique packages     : {df['package_name'].nunique():,}")
print(f"Date range          : {df['package_release_date'].min().date()} → {df['package_release_date'].max().date()}")
print()

for metric, col in [("MTTU", "mttu"), ("MTTR (all sev.)", "mttr_all_severities")]:
    nonzero = (df[col] > 0).sum()
    print(f"{metric}:")
    print(f"  Non-zero rows     : {nonzero:,} ({100*nonzero/len(df):.1f}%)")
    s = df[col]
    print(f"  Mean              : {s.mean():.2f} days")
    print(f"  Median            : {s.median():.2f} days")
    print(f"  25th pct          : {s.quantile(0.25):.2f} days")
    print(f"  75th pct          : {s.quantile(0.75):.2f} days")
    print(f"  95th pct          : {s.quantile(0.95):.2f} days")
    print(f"  Max               : {s.max():.2f} days")
    print()

# --- Figure 1: Releases per year ---
year_counts = df.groupby("release_year").size()
nonzero_mttu = df[df["mttu"] > 0].groupby("release_year").size()
nonzero_mttr = df[df["mttr_all_severities"] > 0].groupby("release_year").size()

fig, ax = plt.subplots(figsize=(10, 4))
x = year_counts.index
width = 0.6

ax.bar(x, year_counts.values, width=width, color="#BDBDBD", label="All releases", zorder=2)
ax.bar(x, nonzero_mttu.reindex(x, fill_value=0).values, width=width * 0.65,
       color=PALETTE["mttu"], alpha=0.85, label="Non-zero MTTU", zorder=3)
ax.bar(x, nonzero_mttr.reindex(x, fill_value=0).values, width=width * 0.35,
       color=PALETTE["mttr"], alpha=0.85, label="Non-zero MTTR", zorder=4)

ax.set_xlabel("Release year")
ax.set_ylabel("Number of releases")
ax.set_title("PyPI releases per year and metric coverage")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_xticks(x[::2])
plt.tight_layout()
plt.savefig("figures/01_releases_per_year.pdf")
plt.close()
print("Saved figures/01_releases_per_year.pdf")

# --- Figure 2: Dependency count distribution ---
fig, ax = plt.subplots(figsize=(7, 4))
deps = df["num_dependencies"].clip(upper=50)
ax.hist(deps, bins=range(0, 52), color="#607D8B", edgecolor="white", linewidth=0.4, zorder=2)
ax.set_xlabel("Number of dependencies (clipped at 50)")
ax.set_ylabel("Number of releases")
ax.set_title("Distribution of dependency count per release")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
plt.tight_layout()
plt.savefig("figures/01_dependency_dist.pdf")
plt.close()
print("Saved figures/01_dependency_dist.pdf")
