"""Script 7: MTTR-only package trend classification — statistics and figures."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pandas as pd
from scipy import stats

from utils import load_results, setup_style, assign_cohort, PALETTE

setup_style()
os.makedirs("figures", exist_ok=True)

COL = "mttr_all_severities"
CV_THRESHOLD = 1.0

CLASS_ORDER = ["improving", "worsening", "stable_low", "stable_high", "stable_variable"]
CLASS_LABELS = {
    "improving":       "Improving",
    "worsening":       "Worsening",
    "stable_low":      "Stable-Low",
    "stable_high":     "Stable-High",
    "stable_variable": "Stable-Variable",
}
CLASS_COLORS = {k: PALETTE[k] for k in CLASS_ORDER}

# ── Load & prep ───────────────────────────────────────────────────────────────
df = load_results()
df = assign_cohort(df)
df = df.sort_values(["package_name", "package_release_date"])

# Ecosystem quartiles on non-zero MTTR releases
mttr_nonzero = df[df[COL] > 0][COL]
Q25 = mttr_nonzero.quantile(0.25)
Q75 = mttr_nonzero.quantile(0.75)

# ── Classification ────────────────────────────────────────────────────────────
def classify_package(values):
    values = np.array(values, dtype=float)
    n = len(values)
    if n < 2:
        med = np.median(values)
        if med <= Q25:   return "stable_low",      0.0, 1.0
        elif med >= Q75: return "stable_high",     0.0, 1.0
        else:            return "stable_variable", 0.0, 1.0
    x = np.arange(n)
    slope, _, _, p, _ = stats.linregress(x, values)
    if p < 0.05:
        return ("improving" if slope < 0 else "worsening"), slope, p
    med = np.median(values)
    cv  = (np.std(values) / med) if med > 0 else 0.0
    if   med <= Q25:          return "stable_low",      slope, p
    elif med >= Q75:          return "stable_high",     slope, p
    elif cv  > CV_THRESHOLD:  return "stable_variable", slope, p
    else:                     return "stable_low",      slope, p

records = []
for pkg, grp in df[df[COL] > 0].groupby("package_name"):
    vals = grp[COL].values
    cls, slope, p = classify_package(vals)
    cohort = grp["cohort"].iloc[0]
    records.append({
        "package_name": pkg,
        "cls":          cls,
        "slope":        slope,
        "p_value":      p,
        "n_releases":   len(vals),
        "median_mttr":  np.median(vals),
        "cohort":       str(cohort),
    })

res = pd.DataFrame(records)

# ── Summary statistics ────────────────────────────────────────────────────────
print("=" * 65)
print("MTTR PACKAGE CLASSIFICATION SUMMARY")
print("=" * 65)
print(f"Total packages with ≥1 non-zero MTTR release: {len(res):,}")
print(f"Ecosystem MTTR Q25 = {Q25:.2f} d  |  Q75 = {Q75:.2f} d")
print()

summary_rows = []
for cls in CLASS_ORDER:
    sub = res[res["cls"] == cls]
    n   = len(sub)
    pct = 100 * n / len(res)
    med_n   = sub["n_releases"].median()
    med_m   = sub["median_mttr"].median()
    q25_m   = sub["median_mttr"].quantile(0.25)
    q75_m   = sub["median_mttr"].quantile(0.75)
    med_sl  = sub["slope"].median()
    print(f"  {CLASS_LABELS[cls]:20s}  n={n:5,} ({pct:5.1f}%)  "
          f"med_releases={med_n:.0f}  med_MTTR={med_m:.1f}d  "
          f"[{q25_m:.1f}, {q75_m:.1f}]  med_slope={med_sl:.3f} d/rel")
    summary_rows.append(dict(
        cls=CLASS_LABELS[cls], n=n, pct=round(pct, 2),
        median_n_releases=round(med_n, 1),
        median_mttr=round(med_m, 2),
        q25_mttr=round(q25_m, 2),
        q75_mttr=round(q75_m, 2),
        median_slope=round(med_sl, 4),
    ))

pd.DataFrame(summary_rows).to_csv("figures/07_class_summary.csv", index=False)
print("\nSaved figures/07_class_summary.csv")

# Cross-class statistical tests
print("\n── Cross-class tests ──────────────────────────────────────────")
kw_groups = [res[res["cls"] == c]["median_mttr"].values for c in CLASS_ORDER]
kw_stat, kw_p = stats.kruskal(*kw_groups)
print(f"  Kruskal-Wallis (median_mttr across 5 classes): H={kw_stat:.2f}, p={kw_p:.3e}")

imp = res[res["cls"] == "improving"]["median_mttr"].values
wor = res[res["cls"] == "worsening"]["median_mttr"].values
u_stat, u_p = stats.mannwhitneyu(imp, wor, alternative="less")
print(f"  Mann-Whitney U (Improving < Worsening in MTTR): U={u_stat:.0f}, p={u_p:.3e}")

# Chi-square: class × cohort
ct = pd.crosstab(res["cls"], res["cohort"])
chi2, chi_p, _, _ = stats.chi2_contingency(ct)
print(f"  Chi-square (class × cohort): χ²={chi2:.2f}, p={chi_p:.3e}")

# ── Figure 1: Classification overview horizontal bar chart ────────────────────
counts  = res["cls"].value_counts().reindex(CLASS_ORDER)
total   = len(res)
colors  = [CLASS_COLORS[c] for c in CLASS_ORDER]
labels  = [CLASS_LABELS[c] for c in CLASS_ORDER]

fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(CLASS_ORDER))
bars = ax.barh(y, counts.values, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Number of packages")
ax.set_title("MTTR trend classification of PyPI packages")
for bar, cnt in zip(bars, counts.values):
    pct = 100 * cnt / total
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
            f"{cnt:,}  ({pct:.1f}%)", va="center", fontsize=9)
ax.set_xlim(0, counts.max() * 1.28)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
plt.tight_layout()
plt.savefig("figures/07_mttr_class_overview.pdf")
plt.close()
print("\nSaved figures/07_mttr_class_overview.pdf")

# ── Figure 2: Per-class statistics — 3 box-plot panels ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))

panels = [
    ("n_releases",   "Releases per package",       True,  None),
    ("median_mttr",  "Per-package median MTTR (d)", False, None),
    ("slope",        "Regression slope (d/release)",False, ["improving", "worsening"]),
]

for ax, (col_name, ylabel, logy, only_cls) in zip(axes, panels):
    data_per_cls = []
    tick_labels  = []
    tick_colors  = []
    cls_list = only_cls if only_cls else CLASS_ORDER
    for cls in cls_list:
        sub = res[res["cls"] == cls][col_name].dropna()
        if only_cls and cls == "improving":
            sub = sub.abs()
        elif only_cls and cls == "worsening":
            sub = sub.abs()
        data_per_cls.append(sub.values)
        tick_labels.append(CLASS_LABELS[cls])
        tick_colors.append(CLASS_COLORS[cls])

    bp = ax.boxplot(data_per_cls, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], tick_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")

    # Kruskal-Wallis annotation
    if len(data_per_cls) >= 2 and all(len(d) >= 2 for d in data_per_cls):
        kw_s, kw_p2 = stats.kruskal(*data_per_cls)
        ax.text(0.98, 0.98, f"KW p={kw_p2:.2e}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CCCCCC"))

plt.suptitle("Per-class MTTR statistics", y=1.01)
plt.tight_layout()
plt.savefig("figures/07_mttr_class_stats.pdf")
plt.close()
print("Saved figures/07_mttr_class_stats.pdf")

# ── Figure 3: Empirical CDF per class ────────────────────────────────────────
cap = mttr_nonzero.quantile(0.95)

fig, ax = plt.subplots(figsize=(7, 5))
for cls in CLASS_ORDER:
    pkgs = res[res["cls"] == cls]["package_name"].values
    vals = df[(df["package_name"].isin(pkgs)) & (df[COL] > 0)][COL].values
    vals = vals[vals <= cap]
    if len(vals) == 0:
        continue
    xs = np.sort(vals)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color=CLASS_COLORS[cls], linewidth=1.8, label=CLASS_LABELS[cls])

ax.set_xlabel(f"MTTR (days, capped at 95th pct: {cap:.0f}d)")
ax.set_ylabel("Empirical CDF")
ax.set_title("MTTR distribution per trend class")
ax.legend()
plt.tight_layout()
plt.savefig("figures/07_mttr_class_cdf.pdf")
plt.close()
print("Saved figures/07_mttr_class_cdf.pdf")

# ── Figure 4: Class composition by cohort (stacked 100% bar) ─────────────────
cohort_order = ["pre-2015", "2015–2018", "2019–2021", "2022+"]
ct_pct = (pd.crosstab(res["cohort"], res["cls"])
            .reindex(cohort_order, fill_value=0)
            .reindex(CLASS_ORDER, axis=1, fill_value=0))
ct_pct_norm = ct_pct.div(ct_pct.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(8, 5))
bottoms = np.zeros(len(cohort_order))
x = np.arange(len(cohort_order))
for cls in CLASS_ORDER:
    vals = ct_pct_norm[cls].values
    ax.bar(x, vals, bottom=bottoms, color=CLASS_COLORS[cls],
           label=CLASS_LABELS[cls], edgecolor="white", linewidth=0.4)
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels(cohort_order)
ax.set_ylabel("Percentage of packages (%)")
ax.set_title("MTTR trend class composition by release cohort")
ax.set_ylim(0, 100)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1), fontsize=9)
ax.text(0.01, 0.02, f"χ²={chi2:.1f}, p={chi_p:.2e}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))
plt.tight_layout()
plt.savefig("figures/07_mttr_class_by_cohort.pdf", bbox_inches="tight")
plt.close()
print("Saved figures/07_mttr_class_by_cohort.pdf")

# ── Figure 5: Trajectory gallery (5 classes × 3 packages) ────────────────────
def pick_representative(cls, n=3):
    sub = res[res["cls"] == cls]
    # Require ≥ 3 releases; pick those closest to class median MTTR
    sub = sub[sub["n_releases"] >= 3].copy()
    if sub.empty:
        sub = res[res["cls"] == cls].copy()
    class_med = sub["median_mttr"].median()
    sub["dist"] = (sub["median_mttr"] - class_med).abs()
    return sub.nsmallest(n, "dist")["package_name"].tolist()


fig, axes = plt.subplots(len(CLASS_ORDER), 3, figsize=(13, 3.5 * len(CLASS_ORDER)))

for row_i, cls in enumerate(CLASS_ORDER):
    pkgs = pick_representative(cls, n=3)
    for col_i in range(3):
        ax = axes[row_i][col_i]
        if col_i >= len(pkgs):
            ax.axis("off")
            continue
        pkg = pkgs[col_i]
        pkg_df = (df[(df["package_name"] == pkg) & (df[COL] > 0)]
                  .sort_values("package_release_date"))
        vals = pkg_df[COL].values
        x    = np.arange(len(vals))
        ax.plot(x, vals, marker="o", markersize=4, linewidth=1.5,
                color=CLASS_COLORS[cls], alpha=0.9)
        # Regression line
        if len(vals) >= 2:
            slope, intercept, _, _, _ = stats.linregress(x, vals)
            ax.plot(x, intercept + slope * x, "k--", linewidth=1, alpha=0.6)
            slope_str = f"slope={slope:+.1f} d/rel"
        else:
            slope_str = ""
        ax.set_title(f"{pkg}\n{slope_str}", fontsize=8)
        ax.set_xlabel("Release index", fontsize=8)
        if col_i == 0:
            ax.set_ylabel(f"{CLASS_LABELS[cls]}\nMTTR (days)", fontsize=8)
        else:
            ax.set_ylabel("MTTR (days)", fontsize=8)

plt.suptitle("Representative MTTR trajectories by trend class", y=1.005)
plt.tight_layout()
plt.savefig("figures/07_mttr_trajectory_gallery.pdf", bbox_inches="tight")
plt.close()
print("Saved figures/07_mttr_trajectory_gallery.pdf")
