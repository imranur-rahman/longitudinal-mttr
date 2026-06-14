"""Script 9: Top-30 MTTR trajectories per trend class, split by ecosystem."""
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
CV_THRESHOLD = 1.0
TOP_N = 30
NCOLS = 5
NROWS = 6  # 6 × 5 = 30

NPM_PATH  = "data/npm/npm_data_depsdev_per_release_results.csv"
PYPI_PATH = "data/pypi/pypi_data_depsdev_per_release_results.csv"

CLASS_ORDER = ["improving", "worsening", "stable_low", "stable_high", "stable_variable"]
CLASS_LABELS = {
    "improving":       "Improving",
    "worsening":       "Worsening",
    "stable_low":      "Stable-Low",
    "stable_high":     "Stable-High",
    "stable_variable": "Stable-Mid",
}

# ── Load both ecosystems ──────────────────────────────────────────────────────
print("Loading PyPI data...")
pypi = load_results(PYPI_PATH)
print("Loading npm data...")
npm  = load_results(NPM_PATH)
df = pd.concat([pypi, npm], ignore_index=True)
df = df.sort_values(["package_name", "package_release_date"])
print(f"Total releases loaded: {len(df):,}  "
      f"(pypi={len(pypi):,}, npm={len(npm):,})\n")

# ── Classification (per-ecosystem quartiles) ──────────────────────────────────
def classify_package(values, q25, q75):
    values = np.array(values, dtype=float)
    n = len(values)
    if n < 2:
        med = np.median(values)
        if med <= q25:   return "stable_low",      0.0, 1.0
        elif med >= q75: return "stable_high",     0.0, 1.0
        else:            return "stable_variable", 0.0, 1.0
    x = np.arange(n)
    slope, _, _, p, _ = stats.linregress(x, values)
    if p < 0.05:
        return ("improving" if slope < 0 else "worsening"), slope, p
    med = np.median(values)
    cv  = (np.std(values) / med) if med > 0 else 0.0
    if   med <= q25:          return "stable_low",      slope, p
    elif med >= q75:          return "stable_high",     slope, p
    elif cv  > CV_THRESHOLD:  return "stable_variable", slope, p
    else:                     return "stable_low",      slope, p


def classify_ecosystem(eco_df):
    """Classify all packages in one ecosystem; return (res_df, pkg_data)."""
    nonzero = eco_df[eco_df[COL] > 0]
    q25 = nonzero[COL].quantile(0.25)
    q75 = nonzero[COL].quantile(0.75)

    records  = []
    pkg_data = {}

    for pkg, grp in nonzero.groupby("package_name"):
        grp   = grp.sort_values("package_release_date")
        vals  = grp[COL].values
        dates = grp["package_release_date"].values
        vers  = grp["package_version"].values
        cls, slope, p = classify_package(vals, q25, q75)
        cv = (np.std(vals) / np.median(vals)) if np.median(vals) > 0 else 0.0
        records.append({
            "package_name": pkg,
            "cls":          cls,
            "slope":        slope,
            "p_value":      p,
            "n_releases":   len(vals),
            "median_mttr":  np.median(vals),
            "cv":           cv,
        })
        pkg_data[pkg] = (dates, vals, vers)

    res = pd.DataFrame(records)
    res = res[res["n_releases"] >= 2].copy()
    return res, pkg_data, q25, q75


# ── Top-N selection ───────────────────────────────────────────────────────────
def top_n(res, cls):
    sub = res[res["cls"] == cls].copy()
    if cls == "improving":
        return sub.nsmallest(TOP_N, "slope")
    elif cls == "worsening":
        return sub.nlargest(TOP_N, "slope")
    else:
        return sub.nlargest(TOP_N, "cv")


# ── Plot one (ecosystem, category) pair ───────────────────────────────────────
def plot_category(res, pkg_data, cls, ecosystem, q25, q75):
    top = top_n(res, cls)
    if top.empty:
        print(f"  [{ecosystem}] No packages for {cls}, skipping.")
        return

    actual_n = min(len(top), TOP_N)
    color    = PALETTE[cls]
    label    = CLASS_LABELS[cls]

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 28))
    axes_flat = axes.flatten()

    for i in range(NROWS * NCOLS):
        ax = axes_flat[i]
        if i >= actual_n:
            ax.axis("off")
            continue

        row  = top.iloc[i]
        pkg  = row["package_name"]
        dates, vals, versions = pkg_data[pkg]

        ax.plot(dates, vals,
                color=color, linewidth=1.5, alpha=0.9, zorder=3,
                marker="o", markersize=4,
                markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.6)

        for date, val, ver in zip(dates, vals, versions):
            ax.annotate(str(ver), (date, val),
                        xytext=(2, 3), textcoords="offset points",
                        fontsize=5, rotation=45, ha="left", va="bottom", alpha=0.75)

        if cls in ("improving", "worsening"):
            ann = f"slope={row['slope']:+.1f} d/rel\np={row['p_value']:.1e}  n={row['n_releases']}"
        else:
            ann = f"CV={row['cv']:.2f}  med={row['median_mttr']:.1f}d\nn={row['n_releases']}"

        ax.text(0.03, 0.97, ann, transform=ax.transAxes,
                va="top", ha="left", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#AAAAAA", alpha=0.9))

        ax.set_title(pkg, fontsize=7, pad=2)
        ax.set_ylabel("MTTR (days)", fontsize=6)
        ax.tick_params(axis="both", labelsize=6)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    eco_upper = ecosystem.upper()
    fig.suptitle(
        f"[{eco_upper}] Top-{actual_n} {label} packages — MTTR by release date  "
        f"(Q25={q25:.1f}d, Q75={q75:.1f}d)",
        fontsize=13, y=1.002,
    )
    plt.tight_layout()
    outpath = f"figures/09_top30_{ecosystem}_{cls}.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")


# ── Run for each ecosystem ────────────────────────────────────────────────────
for ecosystem, eco_label in [("pypi", "PyPI"), ("npm", "npm")]:
    eco_df = df[df["ecosystem"] == ecosystem]
    print(f"── {eco_label} ({'─' * 50})")
    res, pkg_data, q25, q75 = classify_ecosystem(eco_df)
    print(f"  Packages classified (≥2 releases): {len(res):,}  "
          f"Q25={q25:.1f}d  Q75={q75:.1f}d")
    for cls in CLASS_ORDER:
        n = (res["cls"] == cls).sum()
        print(f"    {CLASS_LABELS[cls]:15s}: {n:,}")
    for cls in CLASS_ORDER:
        plot_category(res, pkg_data, cls, ecosystem, q25, q75)
    print()
