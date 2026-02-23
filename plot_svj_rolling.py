"""
plot_svj_rolling.py — SVJ rolling analysis figures for Chapter 7.

Requires:
  • data_plots/rolling_svj_hessian.csv   (from rolling_svj_hessian.py)
  • data_plots/rolling_hessian.csv       (Heston baseline)
  • results/CO1_svj_calibration.csv      (static 2021-12-01 SVJ params)
  • data/Data-Hardcodded.xlsx            (macro indicators)

Produces (saved to data_plots/ and copied to Dissertation_LaTeX/Latex_Paper/):

  A.  svj_vol_surface_co1.png  — 3-D vol surface: SVJ model + market scatter
  B.  svj_condition_number_ts.png  — SVJ vs Heston κ_H time series
  C.  svj_cn_regime_ts.png         — SVJ κ_H with regime background shading
  D.  svj_macro_scatter.png        — scatter: macro vars vs log10(SVJ κ_H)
  E.  svj_regression_forest.png    — HAC OLS forest plot for SVJ κ_H ~ macro

Run with:
    MPLBACKEND=Agg PYTHONUNBUFFERED=1 conda run --no-capture-output python plot_svj_rolling.py
"""

import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats

import surface_loader
import svj_model
import macro_analysis

warnings.filterwarnings("ignore")
os.makedirs("data_plots", exist_ok=True)

LATEX_DIR         = "Dissertation_LaTeX/Latex_Paper"
CAL_DATE          = pd.Timestamp("2021-12-01")
ROLLING_SVJ_CSV   = "data_plots/rolling_svj_hessian.csv"
ROLLING_HESTON_CSV = "data_plots/rolling_hessian.csv"
RESULTS_DIR       = "results"

RMSE_MAX  = 8.0
CN_MAX    = 1e8

STYLE = {
    "CO1": {"color": "seagreen",    "label": "Brent (CO1)"},
    "CL1": {"color": "steelblue",   "label": "WTI (CL1)"},
    "CO1_svj": {"color": "darkorange", "label": "Brent (CO1) SVJ"},
    "CL1_svj": {"color": "firebrick",  "label": "WTI (CL1) SVJ"},
}

EVENTS = [
    (pd.Timestamp("2008-09-15"), "GFC\n2008",          "firebrick"),
    (pd.Timestamp("2016-01-20"), "Supply\nGlut\n2016", "darkorange"),
    (pd.Timestamp("2020-04-21"), "COVID-19\n2020",     "purple"),
    (pd.Timestamp("2022-03-08"), "Ukraine\n2022",      "navy"),
]

REGIME_SHADE = {
    "Calm":             (None,      0.0),
    "Geopolitical":     ("#4e79a7", 0.20),
    "Financial Stress": ("#f28e2b", 0.22),
    "Oil Stress":       ("#e15759", 0.22),
    "Compound":         ("#b07aa1", 0.30),
}


def _add_events(ax, y_frac=0.97, fontsize=7.5):
    ylims = ax.get_ylim()
    yspan = ylims[1] - ylims[0]
    for dt, label, color in EVENTS:
        ax.axvline(dt, color=color, lw=0.9, ls="--", alpha=0.7)
        ax.text(dt, ylims[0] + y_frac * yspan, label,
                ha="center", va="top", fontsize=fontsize, color=color,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))


def _fmt_x(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=9)


def _shade_regimes(ax, sub):
    dates  = sub["Date"].values
    regime = sub["regime"].values
    i = 0
    while i < len(regime):
        j = i + 1
        while j < len(regime) and regime[j] == regime[i]:
            j += 1
        color, alpha = REGIME_SHADE[regime[i]]
        if color is not None and alpha > 0:
            t0 = pd.Timestamp(dates[i])  - pd.Timedelta("15d")
            t1 = pd.Timestamp(dates[j-1]) + pd.Timedelta("15d")
            ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=0)
        i = j


# ---------------------------------------------------------------------------
# Load and validate rolling SVJ data
# ---------------------------------------------------------------------------

if not os.path.exists(ROLLING_SVJ_CSV):
    print(f"ERROR: {ROLLING_SVJ_CSV} not found — run rolling_svj_hessian.py first.")
    sys.exit(1)

print("Loading rolling SVJ data...")
svj_roll = pd.read_csv(ROLLING_SVJ_CSV, parse_dates=["Date"])
svj_roll = svj_roll[svj_roll["rmse_volpts"] <= RMSE_MAX].copy()
svj_roll["condition_number"] = svj_roll["condition_number"].clip(upper=CN_MAX)
svj_roll["log_cn"] = np.log10(svj_roll["condition_number"])
svj_roll = svj_roll.sort_values(["Underlying", "Date"]).reset_index(drop=True)
print(f"  SVJ rolling: {len(svj_roll)} rows after quality filter")

print("Loading Bloomberg surface data (for surface fit)...")
df = surface_loader.build_options_df()
moneyness = df["Strike"] / df["SpotPrice"]
df = df[(moneyness >= 0.70) & (moneyness <= 1.30)].copy()

# Static SVJ params (CO1, 2021-12-01)
co1_csv = os.path.join(RESULTS_DIR, "CO1_svj_calibration.csv")
if not os.path.exists(co1_csv):
    print(f"ERROR: {co1_csv} not found — run calibrate_svj.py first.")
    sys.exit(1)
row = pd.read_csv(co1_csv).iloc[0]
svj_p_co1 = np.array([
    row["kappa"], row["theta"], row["sigma"], row["rho"], row["v0"],
    row["lam"], row["mu_j"], row["sigma_j"]
])

# Heston rolling (for comparison overlay)
heston_roll = None
if os.path.exists(ROLLING_HESTON_CSV):
    heston_roll = pd.read_csv(ROLLING_HESTON_CSV, parse_dates=["Date"])
    heston_roll = heston_roll[heston_roll["rmse_volpts"] <= RMSE_MAX].copy()
    heston_roll["condition_number"] = heston_roll["condition_number"].clip(upper=CN_MAX)
    heston_roll["log_cn"] = np.log10(heston_roll["condition_number"])
    print(f"  Heston rolling: {len(heston_roll)} rows")


# ---------------------------------------------------------------------------
# Figure A — svj_vol_surface_co1.png
# ---------------------------------------------------------------------------

print("\n[A] svj_vol_surface_co1.png  (3-D surface fit)")

sub_co1 = df[(df["Underlying"] == "CO1") & (df["Date"] == CAL_DATE)].dropna(
    subset=["ImpliedVol"]
)
S_co1 = sub_co1["SpotPrice"].iloc[0]
r_co1 = sub_co1["RiskFreeRate"].iloc[0]
K_mkt = sub_co1["Strike"].values
T_mkt = sub_co1["Maturity"].values
IV_mkt = sub_co1["ImpliedVol"].values

# Dense grid for SVJ model surface
T_grid = np.linspace(T_mkt.min(), T_mkt.max(), 20)
K_grid = np.linspace(0.70 * S_co1, 1.30 * S_co1, 40)
KK, TT  = np.meshgrid(K_grid, T_grid)     # shape (20, 40)

IV_surface = np.zeros_like(KK)
for j, T_val in enumerate(T_grid):
    row_ivols = svj_model.svj_implied_vol(
        svj_p_co1, S_co1, K_grid, np.full_like(K_grid, T_val), r_co1
    )
    IV_surface[j, :] = row_ivols * 100   # %

fig = plt.figure(figsize=(12, 7))
ax3d = fig.add_subplot(111, projection="3d")

ax3d.plot_surface(KK / S_co1, TT, IV_surface,
                  cmap="plasma", alpha=0.75, edgecolor="none")
ax3d.scatter(K_mkt / S_co1, T_mkt, IV_mkt * 100,
             color="red", s=35, zorder=10, label="Market")

ax3d.set_xlabel("Moneyness (K/S)", labelpad=8)
ax3d.set_ylabel("Maturity (years)",   labelpad=8)
ax3d.set_zlabel("Implied Vol (%)",    labelpad=8)
ax3d.set_title("SVJ Implied Volatility Surface — Brent (CO1), 1 December 2021",
               fontsize=11, fontweight="bold")
ax3d.legend(fontsize=9)
ax3d.view_init(elev=25, azim=230)

plt.tight_layout()
plt.savefig("data_plots/svj_vol_surface_co1.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/svj_vol_surface_co1.png")


# ---------------------------------------------------------------------------
# Figure B — svj_condition_number_ts.png  (SVJ + Heston overlay)
# ---------------------------------------------------------------------------

print("[B] svj_condition_number_ts.png")

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle(
    r"Rolling Condition Number $\kappa_H$: SVJ vs Heston — 2006–2026",
    fontsize=12, fontweight="bold"
)

for und in ["CO1", "CL1"]:
    # Heston line (solid)
    if heston_roll is not None:
        h_sub = heston_roll[heston_roll["Underlying"] == und].sort_values("Date")
        ax.semilogy(h_sub["Date"], h_sub["condition_number"],
                    color=STYLE[und]["color"], lw=1.4, ls="-",
                    label=f'{STYLE[und]["label"]} Heston', alpha=0.85)

    # SVJ line (dashed)
    s_sub = svj_roll[svj_roll["Underlying"] == und].sort_values("Date")
    if not s_sub.empty:
        ax.semilogy(s_sub["Date"], s_sub["condition_number"],
                    color=STYLE[und + "_svj"]["color"], lw=1.4, ls="--",
                    label=f'{STYLE[und]["label"]} SVJ')

ax.set_ylabel(r"$\kappa_H$ (log scale)", fontsize=10)
ax.set_xlabel("Date", fontsize=10)
ax.grid(True, alpha=0.25, which="both")
ax.legend(fontsize=9, ncol=2)
_fmt_x(ax)
_add_events(ax)

plt.tight_layout()
plt.savefig("data_plots/svj_condition_number_ts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/svj_condition_number_ts.png")


# ---------------------------------------------------------------------------
# Figure C — svj_cn_regime_ts.png
# ---------------------------------------------------------------------------

print("[C] svj_cn_regime_ts.png")

# Merge SVJ with macro for regime classification
macro_df = macro_analysis.load_macro_indicators()
svj_roll_s = svj_roll.sort_values("Date")
macro_s    = macro_df.sort_values("Date")
merged_svj = pd.merge_asof(
    svj_roll_s, macro_s, on="Date",
    direction="nearest", tolerance=pd.Timedelta("15d"),
)
merged_svj = macro_analysis.classify_regimes(merged_svj)

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig.suptitle(
    r"SVJ Hessian Condition Number $\kappa_H^{\mathrm{SVJ}}$ by Market Regime (2006–2026)",
    fontsize=12, fontweight="bold"
)

legend_patches = [
    mpatches.Patch(color=REGIME_SHADE["Financial Stress"][0],
                   alpha=REGIME_SHADE["Financial Stress"][1]*3, label="Financial Stress"),
    mpatches.Patch(color=REGIME_SHADE["Oil Stress"][0],
                   alpha=REGIME_SHADE["Oil Stress"][1]*3,       label="Oil Stress"),
    mpatches.Patch(color=REGIME_SHADE["Compound"][0],
                   alpha=REGIME_SHADE["Compound"][1]*3,         label="Compound"),
    mpatches.Patch(color=REGIME_SHADE["Geopolitical"][0],
                   alpha=REGIME_SHADE["Geopolitical"][1]*3,     label="Geopolitical"),
]

for ax, und in zip(axes, ["CO1", "CL1"]):
    sub = merged_svj[merged_svj["Underlying"] == und].sort_values("Date")
    _shade_regimes(ax, sub)
    ax.plot(sub["Date"], sub["log_cn"],
            color=STYLE[und + "_svj"]["color"], lw=1.0,
            label=f'SVJ {STYLE[und]["label"]}')
    # Overlay Heston for comparison
    if heston_roll is not None:
        h_sub = heston_roll[heston_roll["Underlying"] == und].sort_values("Date")
        ax.plot(h_sub["Date"], np.log10(h_sub["condition_number"]),
                color=STYLE[und]["color"], lw=0.8, ls="--", alpha=0.6,
                label=f'Heston {STYLE[und]["label"]}')
    ax.set_ylabel(r"$\log_{10}(\kappa_H)$", fontsize=9)
    ax.set_title(STYLE[und]["label"], fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="y", labelsize=9)
    _add_events(ax)
    _fmt_x(ax)

axes[0].legend(handles=legend_patches + [
    plt.Line2D([0], [0], color=STYLE["CO1_svj"]["color"], lw=1.2, label="SVJ"),
    plt.Line2D([0], [0], color=STYLE["CO1"]["color"], lw=0.8, ls="--", label="Heston"),
], fontsize=7.5, loc="upper left", ncol=3, framealpha=0.8)

axes[-1].set_xlabel("Year", fontsize=9)
plt.tight_layout()
plt.savefig("data_plots/svj_cn_regime_ts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/svj_cn_regime_ts.png")


# ---------------------------------------------------------------------------
# Figure D — svj_macro_scatter.png
# ---------------------------------------------------------------------------

print("[D] svj_macro_scatter.png")

SCATTER_COLS   = ["OVX", "VIX", "logGPR", "dInventory_z", "DXY"]
SCATTER_LABELS = ["OVX", "VIX", r"$\log$(GPR)", r"$\Delta$Inventory$_z$", "DXY"]

fig, axes = plt.subplots(2, 5, figsize=(16, 6))
fig.suptitle(
    r"$\log_{10}(\kappa_H^{\mathrm{SVJ}})$ vs Macro Indicators",
    fontsize=12, fontweight="bold"
)

for row_i, und in enumerate(["CO1", "CL1"]):
    sub = merged_svj[
        (merged_svj["Underlying"] == und)
    ].dropna(subset=SCATTER_COLS + ["log_cn"])
    for col_j, (col, xlabel) in enumerate(zip(SCATTER_COLS, SCATTER_LABELS)):
        ax = axes[row_i, col_j]
        x  = sub[col].values
        y  = sub["log_cn"].values
        ax.scatter(x, y, s=8, alpha=0.4, color=STYLE[und + "_svj"]["color"])
        if len(x) > 2:
            m, b, r_val, p_val, _ = stats.linregress(x, y)
            xfit = np.linspace(x.min(), x.max(), 100)
            ax.plot(xfit, m * xfit + b, color="black", lw=1.2)
            r2  = r_val ** 2
            sig = "*" if p_val < 0.05 else ("†" if p_val < 0.10 else "")
            ax.text(0.97, 0.96, f"$R^2$={r2:.2f}{sig}",
                    ha="right", va="top", fontsize=7.5,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        if col_j == 0:
            ax.set_ylabel(
                f"{STYLE[und]['label']}\n" + r"$\log_{10}(\kappa_H^{\mathrm{SVJ}})$",
                fontsize=8
            )
        if row_i == 0:
            ax.set_title(xlabel, fontsize=9, fontweight="bold")
        if row_i == 1:
            ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

fig.text(0.5, -0.01, "† p<0.10,  * p<0.05", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("data_plots/svj_macro_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/svj_macro_scatter.png")


# ---------------------------------------------------------------------------
# Figure E — svj_regression_forest.png
# ---------------------------------------------------------------------------

print("[E] svj_regression_forest.png")

print("  Running SVJ OLS regressions...")
ols_svj = macro_analysis.run_ols_regression(merged_svj)

# Also load Heston OLS for comparison (if Heston rolling available)
ols_heston = None
if heston_roll is not None:
    merged_heston = pd.merge_asof(
        heston_roll.sort_values("Date"),
        macro_s,
        on="Date", direction="nearest", tolerance=pd.Timedelta("15d"),
    )
    merged_heston = macro_analysis.classify_regimes(merged_heston)
    print("  Running Heston OLS regressions (for comparison)...")
    ols_heston = macro_analysis.run_ols_regression(merged_heston)

PRED_LABELS = ["OVX", "VIX", "log(GPR)", r"$\Delta$Inv$_z$", "DXY"]
PRED_VARS   = ["OVX_std", "VIX_std", "logGPR_std", "dInventory_z_std", "DXY_std"]

n_pred = len(PRED_VARS)
y_pos  = np.arange(n_pred)

if ols_heston is not None:
    # Side-by-side: Heston vs SVJ, 2 underlyings → 4 bar groups
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        "Standardised OLS Coefficients — Heston vs SVJ\n"
        r"Dependent variable: $\log_{10}(\kappa_H)$  |  HAC Newey-West SE",
        fontsize=11, fontweight="bold"
    )
    bar_w = 0.35

    for ax_i, und in enumerate(["CO1", "CL1"]):
        ax = axes[ax_i]
        for k, (model_label, ols_res, col, ls) in enumerate([
            ("Heston", ols_heston[und], STYLE[und]["color"],       "-"),
            ("SVJ",    ols_svj[und],    STYLE[und + "_svj"]["color"], "--"),
        ]):
            coefs  = [ols_res.params.get(v, np.nan) for v in PRED_VARS]
            ci_lo  = [ols_res.conf_int().loc[v, 0]  if v in ols_res.conf_int().index else np.nan
                      for v in PRED_VARS]
            ci_hi  = [ols_res.conf_int().loc[v, 1]  if v in ols_res.conf_int().index else np.nan
                      for v in PRED_VARS]
            err_lo = np.array(coefs) - np.array(ci_lo)
            err_hi = np.array(ci_hi) - np.array(coefs)

            offset = (k - 0.5) * bar_w
            ax.barh(y_pos + offset, coefs, height=bar_w,
                    xerr=[err_lo, err_hi], color=col, alpha=0.8,
                    error_kw=dict(elinewidth=1.0, capsize=3),
                    label=model_label)

        ax.axvline(0, color="black", lw=0.9, ls="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(PRED_LABELS, fontsize=10)
        ax.set_xlabel(r"Standardised $\hat{\beta}$", fontsize=10)
        ax.set_title(STYLE[und]["label"], fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)

        r2_h = ols_heston[und].rsquared
        r2_s = ols_svj[und].rsquared
        ax.text(0.02, 0.04,
                f"Heston $R^2$={r2_h:.3f}   SVJ $R^2$={r2_s:.3f}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey", alpha=0.8))
else:
    # SVJ only
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Standardised OLS Coefficients — SVJ\n"
        r"Dependent variable: $\log_{10}(\kappa_H^{\mathrm{SVJ}})$  |  HAC Newey-West SE",
        fontsize=11, fontweight="bold"
    )
    bar_w = 0.32
    for i_und, und in enumerate(["CO1", "CL1"]):
        res    = ols_svj[und]
        coefs  = [res.params.get(v, np.nan) for v in PRED_VARS]
        ci_lo  = [res.conf_int().loc[v, 0]  if v in res.conf_int().index else np.nan
                  for v in PRED_VARS]
        ci_hi  = [res.conf_int().loc[v, 1]  if v in res.conf_int().index else np.nan
                  for v in PRED_VARS]
        err_lo = np.array(coefs) - np.array(ci_lo)
        err_hi = np.array(ci_hi) - np.array(coefs)
        offset = (i_und - 0.5) * bar_w
        ax.barh(y_pos + offset, coefs, height=bar_w,
                xerr=[err_lo, err_hi], color=STYLE[und + "_svj"]["color"], alpha=0.8,
                error_kw=dict(elinewidth=1.0, capsize=3),
                label=STYLE[und]["label"])

    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(PRED_LABELS, fontsize=10)
    ax.set_xlabel(r"Standardised $\hat{\beta}$", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("data_plots/svj_regression_forest.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved data_plots/svj_regression_forest.png")


# ---------------------------------------------------------------------------
# Copy all figures to LaTeX directory
# ---------------------------------------------------------------------------

print("\nCopying to LaTeX_Paper/...")
for fname in [
    "svj_vol_surface_co1.png",
    "svj_condition_number_ts.png",
    "svj_cn_regime_ts.png",
    "svj_macro_scatter.png",
    "svj_regression_forest.png",
]:
    src = os.path.join("data_plots", fname)
    dst = os.path.join(LATEX_DIR, fname)
    shutil.copy(src, dst)
    print(f"  {src} → {dst}")

print("\nAll SVJ rolling figures done.")
