"""Script 2: Ecosystem-level MTTU and MTTR trends (yearly + quarterly)."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy import stats

from utils import load_results, setup_style, mann_kendall_summary, PALETTE

try:
    import pyhomogeneity as hg
    HAS_PHG = True
except ImportError:
    HAS_PHG = False
    print("pyhomogeneity not available; Pettitt test will be skipped.")

setup_style()
os.makedirs("figures", exist_ok=True)

df = load_results()


def bootstrap_median_ci(values, n_boot=1000, ci=0.95):
    rng = np.random.default_rng(42)
    medians = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(medians, (1 - ci) / 2 * 100)
    hi = np.percentile(medians, (1 + ci) / 2 * 100)
    return lo, hi


def aggregate_metric(df, col, freq="year"):
    if freq == "year":
        grouped = df.groupby("release_year")[col]
        index_vals = sorted(df["release_year"].unique())
    else:
        grouped = df.groupby("release_quarter")[col]
        index_vals = sorted(df["release_quarter"].unique())

    rows = []
    for k in index_vals:
        vals = grouped.get_group(k).dropna().values
        if len(vals) == 0:
            continue
        lo, hi = bootstrap_median_ci(vals)
        rows.append({
            "period": k,
            "median": np.median(vals),
            "q25": np.percentile(vals, 25),
            "q75": np.percentile(vals, 75),
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(vals),
        })
    return pd.DataFrame(rows)


def plot_trend(agg, col_label, metric_key, freq_label, outpath, title):
    mk_res = mann_kendall_summary(agg["median"])

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(agg))
    color = PALETTE[metric_key]

    ax.fill_between(x, agg["q25"], agg["q75"], alpha=0.15, color=color, label="IQR")
    ax.fill_between(x, agg["ci_lo"], agg["ci_hi"], alpha=0.3, color=color, label="95% bootstrap CI")
    ax.plot(x, agg["median"], color=color, linewidth=2, marker="o", markersize=4, label="Median")

    # Pettitt change-point
    if HAS_PHG and len(agg) >= 5:
        try:
            pt = hg.pettitt_test(agg["median"].values)
            if pt.p < 0.05:
                cp_x = pt.change_point
                ax.axvline(cp_x, color="black", linestyle="--", linewidth=1,
                           label=f"Pettitt CP (p={pt.p:.3f})")
        except Exception:
            pass

    # Annotate MK result
    tau_str = f"τ={mk_res['tau']:.3f}" if mk_res['tau'] is not None else "τ=N/A"
    p_str = f"p={mk_res['p_value']:.3e}" if mk_res['p_value'] is not None else "p=N/A"
    slope_str = f"slope={mk_res['slope']:.3f} days/period" if mk_res['slope'] is not None else ""
    trend_str = mk_res['trend']
    ax.text(0.02, 0.97, f"Mann-Kendall: {trend_str}\n{tau_str}, {p_str}\n{slope_str}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))

    tick_step = max(1, len(agg) // 10)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(agg["period"].iloc[::tick_step], rotation=45, ha="right")
    ax.set_xlabel(f"Release {freq_label}")
    ax.set_ylabel(f"Median {col_label} (days)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")
    print(f"  MK: {trend_str}, {tau_str}, {p_str}, {slope_str}")


metrics = [
    ("mttu", "MTTU", "mttu"),
    ("mttr_all_severities", "MTTR", "mttr"),
]

print("=" * 60)
print("ECOSYSTEM TRENDS")
print("=" * 60)

for col, label, key in metrics:
    sub = df if col == "mttu" else df[df[col] > 0]
    print(f"\n--- {label} (n={len(sub):,} releases) ---")

    for freq, freq_label, freq_key in [("year", "year", "yearly"), ("quarter", "quarter", "quarterly")]:
        agg = aggregate_metric(sub, col, freq=freq)
        outpath = f"figures/02_{key}_trend_{freq_key}.pdf"
        title = f"PyPI {label} trend by release {freq_label}"
        plot_trend(agg, label, key, freq_label, outpath, title)
