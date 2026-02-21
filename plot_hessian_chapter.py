"""
plot_hessian_chapter.py — Chapter 5 figures for the Hessian spectral geometry analysis.

Produces 6 figures (saved to data_plots/):

  A.  hessian_matrix.png          — 2-panel Hessian heatmaps (CO1 | CL1)
  B.  eigenvalue_spectrum.png     — 2-panel eigenvalue bar charts (log scale)
  C.  eigenvector_composition.png — 2×5 signed loading bar charts
  D.  condition_number_ts.png     — Rolling condition number time series (HEADLINE)
  E.  eigenvalue_ts.png           — Rolling eigenvalue time series
  F.  sloppy_direction_ts.png     — Rolling sloppiest eigenvector composition

Run AFTER hessian.py __main__ (static) AND run_rolling_analysis() have completed.

Run with:
    MPLBACKEND=Agg PYTHONUNBUFFERED=1 conda run python plot_hessian_chapter.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import surface_loader
import heston_model
import hessian as hess_mod

os.makedirs("data_plots", exist_ok=True)

PARAM_NAMES  = [r"$\kappa$", r"$\theta$", r"$\sigma$", r"$\rho$", r"$v_0$"]
PARAM_KEYS   = ["kappa", "theta", "sigma", "rho", "v0"]
PARAM_COLORS = {
    "kappa": "#1f77b4",   # blue
    "theta": "#2ca02c",   # green
    "sigma": "#ff7f0e",   # orange
    "rho":   "#d62728",   # red
    "v0":    "#9467bd",   # purple
}

STYLE = {
    "CO1": {"color": "seagreen",  "label": "Brent (CO1)", "ls": "-"},
    "CL1": {"color": "steelblue", "label": "WTI (CL1)",   "ls": "--"},
}

EVENT_ANNOTATIONS = [
    ("2008-09-15", "GFC\n2008",          "firebrick"),
    ("2016-01-20", "Supply\nGlut\n2016", "darkorange"),
    ("2020-04-21", "COVID-19\n2020",     "purple"),
    ("2022-03-08", "Ukraine\n2022",      "navy"),
]

ROLLING_CSV = "data_plots/rolling_hessian.csv"


# ---------------------------------------------------------------------------
# Load static Hessian results
# ---------------------------------------------------------------------------

def load_static_results():
    """
    Reload Bloomberg data and run static Hessian for CO1 and CL1.
    Returns dict: underlying -> {H, spec, params}.
    """
    print("Loading Bloomberg surface data for static analysis...")
    df_full = surface_loader.build_options_df()
    moneyness = df_full["Strike"] / df_full["SpotPrice"]
    df_full = df_full[(moneyness >= 0.70) & (moneyness <= 1.30)].copy()
    cal_date = df_full["Date"].max()
    print(f"Static date: {cal_date.date()}")

    static = {}
    for und in ["CO1", "CL1"]:
        csv_path = os.path.join("results", f"{und}_calibration.csv")
        if not os.path.exists(csv_path):
            print(f"  Missing {csv_path} — run calibrate.py first.")
            sys.exit(1)
        row = pd.read_csv(csv_path).iloc[0]
        params = np.array([row["kappa"], row["theta"], row["sigma"],
                           row["rho"],   row["v0"]])

        sub = df_full[(df_full["Underlying"] == und) &
                      (df_full["Date"] == cal_date)].dropna(subset=["ImpliedVol"])
        S, r = sub["SpotPrice"].iloc[0], sub["RiskFreeRate"].iloc[0]
        K_array, T_array = sub["Strike"].values, sub["Maturity"].values
        IV_market = sub["ImpliedVol"].values

        print(f"  [{und}] Computing Hessian ...")
        H    = hess_mod.compute_hessian(params, S, K_array, T_array, r, IV_market)
        spec = hess_mod.spectral_decomposition(H)
        print(f"  [{und}] κ_H = {spec['condition_number']:.3e}")

        static[und] = {
            "H": H, "spec": spec, "params": params, "rmse": row["rmse_volpts"]
        }
    return static, cal_date


# ---------------------------------------------------------------------------
# Figure A: Hessian matrix heatmaps
# ---------------------------------------------------------------------------

def plot_hessian_matrix(static):
    print("\nFigure A: hessian_matrix.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Hessian of the Calibration Loss $\\mathcal{L}(\\Theta)$ — "
                 "19 February 2026", fontsize=13, fontweight="bold")

    for ax, und in zip(axes, ["CO1", "CL1"]):
        H = static[und]["H"]
        vmax = np.max(np.abs(H))
        im = ax.imshow(H, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(PARAM_NAMES, fontsize=10)
        ax.set_yticklabels(PARAM_NAMES, fontsize=10)
        ax.set_title(STYLE[und]["label"], fontsize=11, fontweight="bold")

        for i in range(5):
            for j in range(5):
                val = H[i, j]
                txt_color = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(j, i, f"{val:.2e}", ha="center", va="center",
                        fontsize=7.5, color=txt_color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("data_plots/hessian_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved data_plots/hessian_matrix.png")


# ---------------------------------------------------------------------------
# Figure B: Eigenvalue spectrum bar charts
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(static):
    print("\nFigure B: eigenvalue_spectrum.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Eigenvalue Spectrum of the Hessian — 19 February 2026",
                 fontsize=13, fontweight="bold")

    colors_bar = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

    for ax, und in zip(axes, ["CO1", "CL1"]):
        spec   = static[und]["spec"]
        evs    = spec["eigenvalues"]
        evecs  = spec["eigenvectors"]
        kH     = spec["condition_number"]

        bars = ax.bar(range(1, 6), np.abs(evs), color=colors_bar, edgecolor="k",
                      linewidth=0.7, zorder=3)

        # Label bars with dominant parameter loading
        for k in range(5):
            dom_idx = np.argmax(np.abs(evecs[:, k]))
            dom_lbl = PARAM_NAMES[dom_idx]
            ax.text(k + 1, np.abs(evs[k]) * 1.5,
                    f"{dom_lbl}\n{evs[k]:.1e}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_yscale("log")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f"$\\lambda_{k}$" for k in range(1, 6)], fontsize=10)
        ax.set_xlabel("Eigenmode (1 = stiffest)", fontsize=9)
        ax.set_ylabel("|Eigenvalue|", fontsize=9)
        ax.set_title(f"{STYLE[und]['label']}\n$\\kappa_H$ = {kH:.2e}",
                     fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("data_plots/eigenvalue_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved data_plots/eigenvalue_spectrum.png")


# ---------------------------------------------------------------------------
# Figure C: Eigenvector composition (signed loadings)
# ---------------------------------------------------------------------------

def plot_eigenvector_composition(static):
    print("\nFigure C: eigenvector_composition.png")
    mode_labels = [f"$\\lambda_{k}$ (stiffest)" if k == 1
                   else (f"$\\lambda_{k}$ (sloppiest)" if k == 5
                         else f"$\\lambda_{k}$")
                   for k in range(1, 6)]

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=False)
    fig.suptitle("Eigenvector Loadings — Stiff and Sloppy Parameter Directions "
                 "(19 February 2026)", fontsize=12, fontweight="bold")

    for row_idx, und in enumerate(["CO1", "CL1"]):
        evecs = static[und]["spec"]["eigenvectors"]

        for col_idx in range(5):
            ax  = axes[row_idx, col_idx]
            vec = evecs[:, col_idx]

            bar_colors = [PARAM_COLORS[k] for k in PARAM_KEYS]
            ax.barh(range(5), vec, color=bar_colors, edgecolor="k", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_yticks(range(5))
            ax.set_yticklabels(PARAM_NAMES if col_idx == 0 else [], fontsize=9)
            ax.set_xlim(-1.1, 1.1)
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["-1", "0", "1"], fontsize=8)
            ax.grid(True, axis="x", alpha=0.3)

            if row_idx == 0:
                ax.set_title(mode_labels[col_idx], fontsize=8.5, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(STYLE[und]["label"], fontsize=9, fontweight="bold")

    # Shared legend for parameter colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PARAM_COLORS[k], edgecolor="k",
                             label=PARAM_NAMES[i])
                       for i, k in enumerate(PARAM_KEYS)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("data_plots/eigenvector_composition.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved data_plots/eigenvector_composition.png")


# ---------------------------------------------------------------------------
# Figures D, E, F: Rolling analysis
# ---------------------------------------------------------------------------

def load_rolling():
    if not os.path.exists(ROLLING_CSV):
        print(f"\n  Rolling CSV not found: {ROLLING_CSV}")
        print("  Run: python hessian.py --rolling  (or the rolling notebook cell)")
        return None
    df = pd.read_csv(ROLLING_CSV, parse_dates=["Date"])
    print(f"\nLoaded rolling results: {len(df)} rows, "
          f"{df['Date'].min().date()} – {df['Date'].max().date()}")
    return df


def _annotate_crises(ax, y_pos_frac=0.92):
    ylim = ax.get_ylim()
    yspan = ylim[1] - ylim[0]
    for date_str, text, color in EVENT_ANNOTATIONS:
        dt = pd.Timestamp(date_str)
        ax.axvline(dt, color=color, linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
        ax.text(dt, ylim[0] + yspan * y_pos_frac, text,
                color=color, fontsize=7.5, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))


def plot_condition_number_ts(rolling_df):
    print("\nFigure D: condition_number_ts.png  (HEADLINE)")
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        r"Rolling Condition Number $\kappa_H = \lambda_{\max} / |\lambda_{\min}|$ "
        "of the Hessian — 2006–2026",
        fontsize=12, fontweight="bold"
    )

    for und in ["CO1", "CL1"]:
        sub = rolling_df[rolling_df["Underlying"] == und].sort_values("Date")
        ax.semilogy(sub["Date"], sub["condition_number"],
                    color=STYLE[und]["color"],
                    linestyle=STYLE[und]["ls"],
                    linewidth=1.4, label=STYLE[und]["label"], zorder=3)

    ax.set_ylabel(r"$\kappa_H$ (log scale)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    _annotate_crises(ax)

    plt.tight_layout()
    plt.savefig("data_plots/condition_number_ts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved data_plots/condition_number_ts.png")


def plot_eigenvalue_ts(rolling_df):
    print("\nFigure E: eigenvalue_ts.png")
    lam_cols = ["lambda1", "lambda2", "lambda3", "lambda4", "lambda5"]
    lam_labels = [f"$\\lambda_{k}$" for k in range(1, 6)]
    colors_line = plt.cm.plasma(np.linspace(0.1, 0.9, 5))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Rolling Eigenvalue Dynamics — Hessian of Calibration Loss",
                 fontsize=12, fontweight="bold")

    for ax, und in zip(axes, ["CO1", "CL1"]):
        sub = rolling_df[rolling_df["Underlying"] == und].sort_values("Date")
        for k, (col, lbl) in enumerate(zip(lam_cols, lam_labels)):
            vals = sub[col].abs()
            ax.semilogy(sub["Date"], vals, color=colors_line[k],
                        linewidth=1.2, label=lbl, zorder=3)

        ax.set_ylabel("|Eigenvalue| (log scale)", fontsize=9)
        ax.set_title(STYLE[und]["label"], fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=8, ncol=5, loc="upper left")
        _annotate_crises(ax)

    axes[-1].set_xlabel("Date", fontsize=10)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    plt.savefig("data_plots/eigenvalue_ts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved data_plots/eigenvalue_ts.png")


def plot_sloppy_direction_ts(rolling_df):
    print("\nFigure F: sloppy_direction_ts.png")
    ev5_cols   = ["ev5_kappa", "ev5_theta", "ev5_sigma", "ev5_rho", "ev5_v0"]
    ev5_labels = [r"$\kappa$", r"$\theta$", r"$\sigma$", r"$\rho$", r"$v_0$"]
    colors_area = [PARAM_COLORS[k] for k in PARAM_KEYS]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        r"Sloppiest Eigenvector ($\lambda_5$) Loading Composition — Rolling 2006–2026",
        fontsize=12, fontweight="bold"
    )

    for ax, und in zip(axes, ["CO1", "CL1"]):
        sub = rolling_df[rolling_df["Underlying"] == und].sort_values("Date")
        dates = sub["Date"].values
        vecs  = sub[ev5_cols].values  # shape (T, 5)

        # Separate positive and negative loadings for stacked area
        pos = np.maximum(vecs, 0)
        neg = np.minimum(vecs, 0)

        ax.stackplot(dates, pos.T, labels=ev5_labels, colors=colors_area, alpha=0.75)
        ax.stackplot(dates, neg.T, colors=colors_area, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.7)

        ax.set_ylabel("Eigenvector loading", fontsize=9)
        ax.set_title(STYLE[und]["label"], fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-1.1, 1.1)
        _annotate_crises(ax, y_pos_frac=0.85)

        if und == "CO1":
            ax.legend(fontsize=8, ncol=5, loc="upper left")

    axes[-1].set_xlabel("Date", fontsize=10)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    plt.savefig("data_plots/sloppy_direction_ts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved data_plots/sloppy_direction_ts.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chapter 5 Hessian figures")
    parser.add_argument("--static-only", action="store_true",
                        help="Only produce static figures A, B, C (no rolling needed)")
    args = parser.parse_args()

    # --- Static figures (A, B, C) ---
    static, cal_date = load_static_results()
    plot_hessian_matrix(static)
    plot_eigenvalue_spectrum(static)
    plot_eigenvector_composition(static)

    if not args.static_only:
        # --- Rolling figures (D, E, F) ---
        rolling_df = load_rolling()
        if rolling_df is not None:
            plot_condition_number_ts(rolling_df)
            plot_eigenvalue_ts(rolling_df)
            plot_sloppy_direction_ts(rolling_df)
            print("\nAll 6 figures saved to data_plots/")
        else:
            print("\nStatic figures (A, B, C) saved. "
                  "Run rolling analysis first for figures D, E, F.")
    else:
        print("\nStatic figures (A, B, C) saved to data_plots/")
