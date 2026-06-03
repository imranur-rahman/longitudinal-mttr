"""Script 6: Within-package longitudinal trends for MTTU and MTTR."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import stats

from utils import load_results, setup_style, PALETTE

setup_style()
os.makedirs("figures", exist_ok=True)

df = load_results()
df = df.sort_values(["package_name", "package_release_date"])

# Ecosystem-level quartiles (for Stable-Low / Stable-High thresholds)
Q25_MTTU = df["mttu"].quantile(0.25)
Q75_MTTU = df["mttu"].quantile(0.75)
Q25_MTTR = df[df["mttr_all_severities"] > 0]["mttr_all_severities"].quantile(0.25)
Q75_MTTR = df[df["mttr_all_severities"] > 0]["mttr_all_severities"].quantile(0.75)
CV_THRESHOLD = 1.0  # coefficient of variation threshold for Stable-Variable

CLASS_ORDER = ["improving", "worsening", "stable_low", "stable_high", "stable_variable"]
CLASS_LABELS = {
    "improving": "Improving",
    "worsening": "Worsening",
    "stable_low": "Stable-Low",
    "stable_high": "Stable-High",
    "stable_variable": "Stable-Variable",
}
CLASS_COLORS = {k: PALETTE[k] for k in CLASS_ORDER}


def classify_package(values, q25, q75):
    values = np.array(values, dtype=float)
    n = len(values)
    if n < 2:
        med = np.median(values)
        if med <= q25:
            return "stable_low", 0.0, 1.0
        elif med >= q75:
            return "stable_high", 0.0, 1.0
        else:
            return "stable_variable", 0.0, 1.0

    x = np.arange(n)
    slope, intercept, r, p, se = stats.linregress(x, values)

    if p < 0.05:
        return ("improving" if slope < 0 else "worsening"), slope, p

    med = np.median(values)
    cv = (np.std(values) / med) if med > 0 else 0.0
    if med <= q25:
        return "stable_low", slope, p
    elif med >= q75:
        return "stable_high", slope, p
    elif cv > CV_THRESHOLD:
        return "stable_variable", slope, p
    else:
        return "stable_low", slope, p  # consistently near-zero


def analyze_metric(df, col, label, q25, q75, nonzero_only=False):
    if nonzero_only:
        work_df = df[df[col] > 0].copy()
    else:
        work_df = df.copy()

    records = []
    for pkg, grp in work_df.groupby("package_name"):
        vals = grp.sort_values("package_release_date")[col].values
        cls, slope, p = classify_package(vals, q25, q75)
        records.append({
            "package_name": pkg,
            "cls": cls,
            "slope": slope,
            "p_value": p,
            "n_releases": len(vals),
            "median": np.median(vals),
        })
    return pd.DataFrame(records)


print("=" * 60)
print("WITHIN-PACKAGE TRENDS")
print("=" * 60)

results = {}
for col, label, key, q25, q75, nz in [
    ("mttu", "MTTU", "mttu", Q25_MTTU, Q75_MTTU, False),
    ("mttr_all_severities", "MTTR", "mttr", Q25_MTTR, Q75_MTTR, True),
]:
    print(f"\n--- {label} ---")
    res = analyze_metric(df, col, label, q25, q75, nonzero_only=nz)
    results[key] = res

    counts = res["cls"].value_counts()
    total = len(res)
    print(f"  Total packages analyzed: {total:,}")
    for cls in CLASS_ORDER:
        n = counts.get(cls, 0)
        print(f"    {CLASS_LABELS[cls]:20s}: {n:5,}  ({100*n/total:.1f}%)")

    # Top 50 improving / worsening
    improving = res[res["cls"] == "improving"].nsmallest(50, "slope")
    worsening = res[res["cls"] == "worsening"].nlargest(50, "slope")
    improving.to_csv(f"figures/06_top50_improving_{key}.csv", index=False)
    worsening.to_csv(f"figures/06_top50_worsening_{key}.csv", index=False)
    print(f"  Saved top-50 CSVs for {label}.")


# --- Figure 1: Classification bar chart (MTTU vs MTTR side by side) ---
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(CLASS_ORDER))
width = 0.35

for i, (key, label) in enumerate([("mttu", "MTTU"), ("mttr", "MTTR")]):
    res = results[key]
    counts = res["cls"].value_counts()
    total = len(res)
    vals = [100 * counts.get(c, 0) / total for c in CLASS_ORDER]
    bars = ax.bar(x + (i - 0.5) * width, vals, width,
                  color=[CLASS_COLORS[c] for c in CLASS_ORDER],
                  alpha=0.85 if i == 0 else 0.55,
                  edgecolor="white", linewidth=0.5,
                  label=label)

ax.set_xticks(x)
ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER])
ax.set_ylabel("Percentage of packages (%)")
ax.set_title("Within-package trend classification")

import matplotlib.patches as mpatches
legend_patches = [mpatches.Patch(facecolor="#555555", alpha=0.85, label="MTTU"),
                  mpatches.Patch(facecolor="#555555", alpha=0.55, label="MTTR")]
ax.legend(handles=legend_patches)
plt.tight_layout()
plt.savefig("figures/06_package_trend_classification.pdf")
plt.close()
print("\nSaved figures/06_package_trend_classification.pdf")


# --- Figure 2: Top-50 slope horizontal bar chart (MTTU + MTTR panels) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

for row_i, (key, label) in enumerate([("mttu", "MTTU"), ("mttr", "MTTR")]):
    res = results[key]
    for col_i, (direction, top_n, col_color) in enumerate([
        ("improving", res[res["cls"] == "improving"].nsmallest(50, "slope"), PALETTE["improving"]),
        ("worsening", res[res["cls"] == "worsening"].nlargest(50, "slope"), PALETTE["worsening"]),
    ]):
        ax = axes[row_i][col_i]
        if top_n.empty:
            ax.text(0.5, 0.5, "No packages", ha="center", va="center")
            continue
        top_n = top_n.head(50)
        y_pos = np.arange(len(top_n))
        ax.barh(y_pos, top_n["slope"].abs(), color=col_color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_n["package_name"], fontsize=6)
        ax.set_xlabel("Absolute slope (days/release)")
        ax.set_title(f"Top 50 {direction.capitalize()} packages — {label}")
        ax.invert_yaxis()

plt.tight_layout()
plt.savefig("figures/06_top50_slope_chart.pdf")
plt.close()
print("Saved figures/06_top50_slope_chart.pdf")


# --- Figure 3: Example package trajectories ---
def pick_examples(res, df_all, col, direction, n=2):
    candidates = res[res["cls"] == direction]
    if candidates.empty:
        return []
    if direction == "improving":
        top = candidates.nsmallest(n * 5, "slope")
    elif direction == "worsening":
        top = candidates.nlargest(n * 5, "slope")
    else:
        top = candidates.nlargest(n * 5, "n_releases")

    picked = []
    for pkg in top["package_name"]:
        vals = df_all[df_all["package_name"] == pkg].sort_values("package_release_date")[col].values
        if len(vals) >= 2:
            picked.append((pkg, vals))
        if len(picked) == n:
            break
    return picked


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
example_specs = [
    ("improving", PALETTE["improving"]),
    ("worsening", PALETTE["worsening"]),
    ("stable_variable", PALETTE["stable_variable"]),
]

for row_i, (key, label, col) in enumerate([
    ("mttu", "MTTU", "mttu"),
    ("mttr", "MTTR", "mttr_all_severities"),
]):
    res = results[key]
    for col_i, (direction, color) in enumerate(example_specs):
        ax = axes[row_i][col_i]
        examples = pick_examples(res, df, col, direction, n=2)
        for pkg, vals in examples:
            ax.plot(np.arange(len(vals)), vals, marker="o", markersize=3,
                    linewidth=1.5, alpha=0.85, label=pkg)
        ax.set_title(f"{CLASS_LABELS[direction]} — {label}", fontsize=10)
        ax.set_xlabel("Release index")
        ax.set_ylabel(f"{label} (days)")
        ax.legend(fontsize=7, loc="best")

plt.tight_layout()
plt.savefig("figures/06_example_packages.pdf")
plt.close()
print("Saved figures/06_example_packages.pdf")
