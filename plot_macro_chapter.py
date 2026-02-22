"""
plot_macro_chapter.py — Chapter 6 figures.

Produces 6 figures (saved to data_plots/):

  A.  macro_dashboard.png      — 4-panel macro indicator overview
  B.  cn_regime_ts.png         — log10(kappa_H) with regime background bands
  C.  regime_boxplot.png       — box plots of log10(kappa_H) by regime
  D.  scatter_indicators.png   — 2x5 scatter grid: indicators vs log10(kappa_H)
  E.  regression_forest.png    — forest plot of standardised HAC OLS coefficients
  F.  event_study.png          — 4-panel event zooms (GFC, Glut, COVID, Ukraine)

Run with:
    MPLBACKEND=Agg python plot_macro_chapter.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from macro_analysis import (
    load_macro_indicators,
    merge_with_rolling,
    classify_regimes,
    run_ols_regression,
    REGIME_ORDER,
    REGIME_COLORS,
    THRESH_VIX,
    THRESH_OVX,
    THRESH_GPR,
    REGRESSORS_RAW,
    REGRESSOR_LABELS,
)

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("data_plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
STYLE = {
    "CO1": {"color": "seagreen",  "label": "Brent (CO1)", "ls": "-"},
    "CL1": {"color": "steelblue", "label": "WTI (CL1)",   "ls": "--"},
}

# Crisis event lines (date, label, color)
EVENTS = [
    (pd.Timestamp("2008-09-15"), "GFC\n2008",          "firebrick"),
    (pd.Timestamp("2016-01-20"), "Supply\nGlut\n2016", "darkorange"),
    (pd.Timestamp("2020-04-21"), "COVID-19\n2020",     "purple"),
    (pd.Timestamp("2022-03-08"), "Ukraine\n2022",      "navy"),
]


def _add_event_lines(ax, y_frac=0.97, fontsize=7.5):
    ylims = ax.get_ylim()
    yspan = ylims[1] - ylims[0]
    for dt, label, color in EVENTS:
        ax.axvline(dt, color=color, lw=0.9, ls="--", alpha=0.7)
        ax.text(dt, ylims[0] + y_frac * yspan, label,
                ha="center", va="top", fontsize=fontsize, color=color,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))


def _fmt_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=9)


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
print("Loading data...")
macro  = load_macro_indicators()
merged = merge_with_rolling()
merged = classify_regimes(merged)
print("Running OLS regressions...")
ols_results = run_ols_regression(merged)

# ---------------------------------------------------------------------------
# Figure A — macro_dashboard.png
# ---------------------------------------------------------------------------
print("\n[A] macro_dashboard.png")

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Macro-Financial Indicators — Monthly Averages (2006–2026)",
             fontsize=12, fontweight="bold")

macro_plt = macro.set_index("Date")

# Panel 1: OVX
ax = axes[0]
ax.fill_between(macro_plt.index, macro_plt["OVX"], alpha=0.3, color="tomato")
ax.plot(macro_plt.index, macro_plt["OVX"], color="tomato", lw=0.9)
ax.axhline(THRESH_OVX, color="black", ls=":", lw=0.8, label=f"Threshold ({THRESH_OVX:.0f})")
ax.set_ylabel("OVX", fontsize=9)
ax.set_title("Crude Oil Volatility Index (OVX)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25)
ax.tick_params(axis="y", labelsize=9)

# Panel 2: VIX
ax = axes[1]
ax.fill_between(macro_plt.index, macro_plt["VIX"], alpha=0.3, color="steelblue")
ax.plot(macro_plt.index, macro_plt["VIX"], color="steelblue", lw=0.9)
ax.axhline(THRESH_VIX, color="black", ls=":", lw=0.8, label=f"Threshold ({THRESH_VIX:.0f})")
ax.set_ylabel("VIX", fontsize=9)
ax.set_title("CBOE Equity Volatility Index (VIX)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25)
ax.tick_params(axis="y", labelsize=9)

# Panel 3: GPR
ax = axes[2]
ax.fill_between(macro_plt.index, macro_plt["GPR"], alpha=0.3, color="goldenrod")
ax.plot(macro_plt.index, macro_plt["GPR"], color="goldenrod", lw=0.9)
ax.axhline(THRESH_GPR, color="black", ls=":", lw=0.8, label=f"Threshold ({THRESH_GPR:.0f})")
ax.set_ylabel("GPR Index", fontsize=9)
ax.set_title("Geopolitical Risk Index (Caldara \\& Iacoviello)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25)
ax.tick_params(axis="y", labelsize=9)

# Panel 4: Inventory change (z-score)
ax = axes[3]
pos = macro_plt["dInventory_z"].clip(lower=0)
neg = macro_plt["dInventory_z"].clip(upper=0)
ax.bar(macro_plt.index, pos, width=20, color="mediumseagreen", alpha=0.7, label="Build")
ax.bar(macro_plt.index, neg, width=20, color="salmon",        alpha=0.7, label="Draw")
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("z-score", fontsize=9)
ax.set_title(r"EIA Crude Inventory Change ($\Delta$Inventory, z-scored)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25)
ax.tick_params(axis="y", labelsize=9)

for ax in axes:
    _add_event_lines(ax)
    _fmt_xaxis(ax)

axes[-1].set_xlabel("Year", fontsize=9)
plt.tight_layout()
plt.savefig("data_plots/macro_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/macro_dashboard.png")


# ---------------------------------------------------------------------------
# Figure B — cn_regime_ts.png
# ---------------------------------------------------------------------------
print("[B] cn_regime_ts.png")

REGIME_SHADE = {
    "Calm":             (None, 0.0),
    "Geopolitical":     ("#4e79a7", 0.20),
    "Financial Stress": ("#f28e2b", 0.22),
    "Oil Stress":       ("#e15759", 0.22),
    "Compound":         ("#b07aa1", 0.30),
}


def _shade_regimes(ax, sub):
    """Draw background regime bands for a single underlying's time series."""
    dates  = sub["Date"].values
    regime = sub["regime"].values
    # Walk through consecutive same-regime spans
    i = 0
    while i < len(regime):
        j = i + 1
        while j < len(regime) and regime[j] == regime[i]:
            j += 1
        color, alpha = REGIME_SHADE[regime[i]]
        if color is not None and alpha > 0:
            t0 = pd.Timestamp(dates[i]) - pd.Timedelta("15d")
            t1 = pd.Timestamp(dates[j - 1]) + pd.Timedelta("15d")
            ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=0)
        i = j


fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig.suptitle(r"Hessian Condition Number $\kappa_H$ by Market Regime (2006–2026)",
             fontsize=12, fontweight="bold")

for ax, und in zip(axes, ["CO1", "CL1"]):
    sub = merged[merged["Underlying"] == und].sort_values("Date")
    _shade_regimes(ax, sub)
    ax.plot(sub["Date"], sub["log_cn"],
            color=STYLE[und]["color"], lw=1.0, label=STYLE[und]["label"])
    ax.set_ylabel(r"$\log_{10}(\kappa_H)$", fontsize=9)
    ax.set_title(STYLE[und]["label"], fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="y", labelsize=9)
    _add_event_lines(ax)
    _fmt_xaxis(ax)

# Legend for regimes
legend_patches = [
    mpatches.Patch(color=REGIME_SHADE["Financial Stress"][0],
                   alpha=REGIME_SHADE["Financial Stress"][1] * 3, label="Financial Stress"),
    mpatches.Patch(color=REGIME_SHADE["Oil Stress"][0],
                   alpha=REGIME_SHADE["Oil Stress"][1] * 3,       label="Oil Stress"),
    mpatches.Patch(color=REGIME_SHADE["Compound"][0],
                   alpha=REGIME_SHADE["Compound"][1] * 3,         label="Compound"),
    mpatches.Patch(color=REGIME_SHADE["Geopolitical"][0],
                   alpha=REGIME_SHADE["Geopolitical"][1] * 3,     label="Geopolitical"),
]
axes[0].legend(handles=legend_patches, fontsize=8, loc="upper left",
               ncol=4, framealpha=0.8)

axes[-1].set_xlabel("Year", fontsize=9)
plt.tight_layout()
plt.savefig("data_plots/cn_regime_ts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/cn_regime_ts.png")


# ---------------------------------------------------------------------------
# Figure C — regime_boxplot.png
# ---------------------------------------------------------------------------
print("[C] regime_boxplot.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle(r"$\log_{10}(\kappa_H)$ by Market Regime",
             fontsize=12, fontweight="bold")

palette = {r: REGIME_SHADE[r][0] if REGIME_SHADE[r][0] else "#cccccc"
           for r in REGIME_ORDER}

for ax, und in zip(axes, ["CO1", "CL1"]):
    sub = merged[(merged["Underlying"] == und) & merged["OVX"].notna()].copy()
    sns.boxplot(
        data=sub, x="regime", y="log_cn",
        order=REGIME_ORDER,
        palette=palette,
        width=0.5, linewidth=0.8,
        ax=ax,
        showfliers=True, flierprops=dict(marker=".", ms=3, alpha=0.5),
    )
    # Annotate with median and count
    for i, reg in enumerate(REGIME_ORDER):
        grp = sub[sub["regime"] == reg]["log_cn"]
        if len(grp) == 0:
            continue
        ax.text(i, grp.median() + 0.05, f"n={len(grp)}",
                ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax.set_title(STYLE[und]["label"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Regime", fontsize=9)
    ax.set_ylabel(r"$\log_{10}(\kappa_H)$", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("data_plots/regime_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/regime_boxplot.png")


# ---------------------------------------------------------------------------
# Figure D — scatter_indicators.png
# ---------------------------------------------------------------------------
print("[D] scatter_indicators.png")

SCATTER_COLS = ["OVX", "VIX", "logGPR", "dInventory_z", "DXY"]
SCATTER_LABELS = [
    "OVX", "VIX", r"$\log$(GPR)", r"$\Delta$Inventory$_z$", "DXY"
]

fig, axes = plt.subplots(2, 5, figsize=(16, 6))
fig.suptitle(r"$\log_{10}(\kappa_H)$ vs Macro Indicators",
             fontsize=12, fontweight="bold")

for row_i, und in enumerate(["CO1", "CL1"]):
    sub = merged[(merged["Underlying"] == und)].dropna(subset=SCATTER_COLS + ["log_cn"])
    for col_j, (col, xlabel) in enumerate(zip(SCATTER_COLS, SCATTER_LABELS)):
        ax = axes[row_i, col_j]
        x = sub[col].values
        y = sub["log_cn"].values
        ax.scatter(x, y, s=8, alpha=0.4, color=STYLE[und]["color"])

        # OLS fit line
        if len(x) > 2:
            m, b, r, p, _ = stats.linregress(x, y)
            xfit = np.linspace(x.min(), x.max(), 100)
            ax.plot(xfit, m * xfit + b, color="black", lw=1.2)
            r2 = r ** 2
            sig = "*" if p < 0.05 else ("†" if p < 0.10 else "")
            ax.text(0.97, 0.96, f"$R^2$={r2:.2f}{sig}",
                    ha="right", va="top", fontsize=7.5,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

        if col_j == 0:
            ax.set_ylabel(f"{STYLE[und]['label']}\n" + r"$\log_{10}(\kappa_H)$", fontsize=8)
        if row_i == 0:
            ax.set_title(xlabel, fontsize=9, fontweight="bold")
        if row_i == 1:
            ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

fig.text(0.5, -0.01, "† p<0.10,  * p<0.05", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("data_plots/scatter_indicators.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/scatter_indicators.png")


# ---------------------------------------------------------------------------
# Figure E — regression_forest.png
# ---------------------------------------------------------------------------
print("[E] regression_forest.png")

PRED_LABELS = ["OVX", "VIX", "log(GPR)", r"$\Delta$Inv$_z$", "DXY"]
PRED_VARS   = ["OVX_std", "VIX_std", "logGPR_std", "dInventory_z_std", "DXY_std"]

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Standardised OLS Coefficients with 95\\% CI\n"
             r"Dependent variable: $\log_{10}(\kappa_H)$",
             fontsize=11, fontweight="bold")

n_pred  = len(PRED_VARS)
y_pos   = np.arange(n_pred)
bar_w   = 0.32

for i_und, und in enumerate(["CO1", "CL1"]):
    res    = ols_results[und]
    coefs  = [res.params.get(v, np.nan)    for v in PRED_VARS]
    ci_lo  = [res.conf_int().loc[v, 0] if v in res.conf_int().index else np.nan for v in PRED_VARS]
    ci_hi  = [res.conf_int().loc[v, 1] if v in res.conf_int().index else np.nan for v in PRED_VARS]
    err_lo = np.array(coefs) - np.array(ci_lo)
    err_hi = np.array(ci_hi) - np.array(coefs)

    offset = (i_und - 0.5) * bar_w
    ax.barh(y_pos + offset, coefs,
            height=bar_w, xerr=[err_lo, err_hi],
            color=STYLE[und]["color"], alpha=0.8,
            error_kw=dict(elinewidth=1.0, capsize=3),
            label=STYLE[und]["label"])

ax.axvline(0, color="black", lw=0.9, ls="--")
ax.set_yticks(y_pos)
ax.set_yticklabels(PRED_LABELS, fontsize=10)
ax.set_xlabel("Standardised coefficient $\\hat{\\beta}$", fontsize=10)
ax.set_title("", fontsize=10)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, axis="x", alpha=0.3)
ax.tick_params(axis="x", labelsize=9)

# R² annotation
co1_r2 = ols_results["CO1"].rsquared
cl1_r2 = ols_results["CL1"].rsquared
ax.text(0.02, 0.04,
        f"CO1: $R^2$={co1_r2:.3f}   CL1: $R^2$={cl1_r2:.3f}",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey", alpha=0.8))

plt.tight_layout()
plt.savefig("data_plots/regression_forest.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/regression_forest.png")


# ---------------------------------------------------------------------------
# Figure F — event_study.png
# ---------------------------------------------------------------------------
print("[F] event_study.png")

EVENT_WINDOWS = [
    ("GFC 2008",          "2007-10-01", "2009-04-01"),
    ("Oil Supply Glut 2016", "2015-07-01", "2017-01-01"),
    ("COVID-19 2020",     "2019-10-01", "2021-04-01"),
    ("Ukraine War 2022",  "2021-09-01", "2023-03-01"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Event Studies: OVX and Condition Number Around Crisis Episodes",
             fontsize=12, fontweight="bold")

macro_plt2 = macro.set_index("Date")

for idx, (title, t0, t1) in enumerate(EVENT_WINDOWS):
    ax_main = axes[idx // 2, idx % 2]
    t0, t1  = pd.Timestamp(t0), pd.Timestamp(t1)

    # Macro slice
    m_slice = macro_plt2.loc[t0:t1]

    # CN slice
    ax_cn = ax_main.twinx()

    # OVX bars (left)
    ax_main.bar(m_slice.index, m_slice["OVX"], width=20,
                color="tomato", alpha=0.5, label="OVX (left)")
    ax_main.set_ylabel("OVX", fontsize=8, color="tomato")
    ax_main.tick_params(axis="y", labelcolor="tomato", labelsize=8)

    # CN lines (right)
    for und in ["CO1", "CL1"]:
        sub = merged[(merged["Underlying"] == und) &
                     (merged["Date"] >= t0) & (merged["Date"] <= t1)]
        ax_cn.plot(sub["Date"], sub["log_cn"],
                   color=STYLE[und]["color"], lw=1.3, ls=STYLE[und]["ls"],
                   label=STYLE[und]["label"])

    ax_cn.set_ylabel(r"$\log_{10}(\kappa_H)$", fontsize=8, color="#333333")
    ax_cn.tick_params(axis="y", labelcolor="#333333", labelsize=8)

    # Crisis vertical line
    crisis_dates = {
        "GFC 2008":             pd.Timestamp("2008-09-15"),
        "Oil Supply Glut 2016": pd.Timestamp("2016-01-20"),
        "COVID-19 2020":        pd.Timestamp("2020-04-21"),
        "Ukraine War 2022":     pd.Timestamp("2022-03-08"),
    }
    ax_main.axvline(crisis_dates[title], color="black", ls="--", lw=1.0, alpha=0.7)

    ax_main.set_title(title, fontsize=10, fontweight="bold")
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_main.tick_params(axis="x", labelsize=7.5, rotation=30)
    ax_main.grid(True, alpha=0.2)

    # Combine legends from both axes
    h1, l1 = ax_main.get_legend_handles_labels()
    h2, l2 = ax_cn.get_legend_handles_labels()
    ax_main.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")

plt.tight_layout()
plt.savefig("data_plots/event_study.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/event_study.png")

print("\nAll 6 Chapter 6 figures saved to data_plots/.")
