import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pymannkendall as mk

DATA_PATH = "data/pypi/pypi_data_depsdev_per_release_results.csv"

PALETTE = {
    "mttu": "#2196F3",
    "mttr": "#E91E63",
    "improving": "#4CAF50",
    "worsening": "#F44336",
    "stable_low": "#8BC34A",
    "stable_high": "#FF9800",
    "stable_variable": "#9E9E9E",
}

COHORT_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]


def setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_results(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["package_release_date", "window_end"])
    df = df[df["status"] == "ok"].copy()
    df["release_year"] = df["package_release_date"].dt.year
    df["release_quarter"] = df["package_release_date"].dt.to_period("Q").astype(str)
    return df


def assign_cohort(df):
    first_year = df.groupby("package_name")["package_release_date"].min().dt.year
    first_year.name = "cohort_year"
    df = df.join(first_year, on="package_name")

    bins = [0, 2014, 2018, 2021, 9999]
    labels = ["pre-2015", "2015–2018", "2019–2021", "2022+"]
    df["cohort"] = pd.cut(df["cohort_year"], bins=bins, labels=labels)
    return df


def mann_kendall_summary(series):
    series = series.dropna()
    if len(series) < 3:
        return {"trend": "insufficient data", "tau": None, "p_value": None, "slope": None}
    result = mk.original_test(series)
    return {
        "trend": result.trend,
        "tau": result.Tau,
        "p_value": result.p,
        "slope": result.slope,
    }
