"""
plot_skew_analysis.py — Smile and skew stylised facts for Chapter 2.

Outputs (to data_plots/):
  avg_smile_shape.png        — Time-averaged IV(m) − IV_ATM per maturity
  crisis_smile_distortion.png — Smile distortion during market stress (2×6 grid)
  risk_reversal_ts.png       — Risk reversal IV(1.10) − IV(0.90) time series

Run with:
  MPLBACKEND=Agg PYTHONUNBUFFERED=1 conda run python plot_skew_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

import surface_loader

os.makedirs("data_plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Load and enrich data
# ---------------------------------------------------------------------------
print("Loading Bloomberg surface data...")
df = surface_loader.build_options_df()

df["Moneyness"] = (df["Strike"] / df["SpotPrice"]).round(3)
df["IV_pct"]    = df["ImpliedVol"] * 100

TENOR_TO_MONTHS = {
    round(1 / 12, 6): 1,
    round(2 / 12, 6): 2,
    round(3 / 12, 6): 3,
    round(6 / 12, 6): 6,
    1.0: 12,
    2.0: 24,
}
df["Maturity_months"] = df["Maturity"].apply(
    lambda x: TENOR_TO_MONTHS.get(round(x, 6), round(x * 12))
)

STYLE = {
    "CO1": {"color": "seagreen",  "label": "Brent (CO1)"},
    "CL1": {"color": "steelblue", "label": "WTI (CL1)"},
}

MONEYNESS_NODES = [0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10]
MATURITIES_MONTHS = [1, 2, 3, 6, 12, 24]

print(f"Loaded {len(df):,} rows. Date range: "
      f"{df['Date'].min().date()} – {df['Date'].max().date()}")


# ---------------------------------------------------------------------------
# Figure A: avg_smile_shape.png
# Time-averaged IV(m) − IV_ATM by maturity, for each underlying
# ---------------------------------------------------------------------------
print("\nComputing average smile shape...")

# For each (Date, Underlying, Maturity), get the relative smile IV(m) − IV(1.00)
# We pivot to wide format first, then subtract the ATM column.

smile_pivot = (
    df.groupby(["Date", "Underlying", "Maturity_months", "Moneyness"])["IV_pct"]
    .mean()
    .reset_index()
    .pivot_table(index=["Date", "Underlying", "Maturity_months"],
                 columns="Moneyness", values="IV_pct")
)

# Only keep rows where ATM (1.000) is present
smile_pivot = smile_pivot.dropna(subset=[1.000])

# Relative smile: IV(m) − IV(1.000)
atm_col = smile_pivot[1.000]
relative = smile_pivot.subtract(atm_col, axis=0)

# Aggregate over dates
rel_mean = relative.groupby(["Underlying", "Maturity_months"]).mean()
rel_std  = relative.groupby(["Underlying", "Maturity_months"]).std()

# Colourmap: one colour per maturity (darker = shorter)
mat_cmap = plt.colormaps["plasma"].resampled(len(MATURITIES_MONTHS))
mat_colors = {m: mat_cmap(i) for i, m in enumerate(MATURITIES_MONTHS)}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle(
    "Time-Averaged Implied Volatility Smile Shape (2006–2026)",
    fontsize=13, fontweight="bold"
)

for ax, underlying in zip(axes, ["CO1", "CL1"]):
    label = STYLE[underlying]["label"]
    for mat in MATURITIES_MONTHS:
        if (underlying, mat) not in rel_mean.index:
            continue
        means = rel_mean.loc[(underlying, mat)]
        stds  = rel_std.loc[(underlying, mat)]
        mon   = means.index.values
        mu    = means.values
        sd    = stds.values
        color = mat_colors[mat]
        mon_label = f"{mat}M"

        ax.plot(mon, mu, color=color, linewidth=1.8, label=mon_label, zorder=3)
        ax.fill_between(mon, mu - sd, mu + sd, color=color, alpha=0.10, zorder=2)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(1.0, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("Moneyness (K/F)")
    ax.set_ylabel("IV − IV$_{\\mathrm{ATM}}$ (vol pts)")
    ax.set_title(label)
    ax.set_xticks(MONEYNESS_NODES)
    ax.set_xticklabels(["90%", "95%", "97.5%", "ATM", "102.5%", "105%", "110%"],
                       fontsize=8, rotation=30)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Maturity", fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.savefig("data_plots/avg_smile_shape.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data_plots/avg_smile_shape.png")


# ---------------------------------------------------------------------------
# Figure B: crisis_smile_distortion.png
# Smile distortion (all maturities) across crisis periods — 2×6 grid
# ---------------------------------------------------------------------------
print("\nComputing crisis smile distortions...")

CRISIS_PERIODS = {
    "Calm (2012)":      ("calm",      "2012-06-01", "2012-08-31"),
    "GFC (2008)":       ("single",    "2008-10-15", None),
    "Supply Glut (2016)": ("single",  "2016-01-20", None),
    "COVID-19 (2020)":  ("single",    "2020-04-21", None),
    "Ukraine (2022)":   ("single",    "2022-03-08", None),
}

CRISIS_STYLE = {
    "Calm (2012)":        {"color": "gray",       "ls": "--", "lw": 1.6},
    "GFC (2008)":         {"color": "firebrick",  "ls": "-",  "lw": 1.6},
    "Supply Glut (2016)": {"color": "darkorange", "ls": "-",  "lw": 1.6},
    "COVID-19 (2020)":    {"color": "purple",     "ls": "-",  "lw": 1.6},
    "Ukraine (2022)":     {"color": "navy",       "ls": "-",  "lw": 1.6},
}

all_dates = df["Date"].sort_values().unique()


def nearest_date(target_str):
    target = pd.Timestamp(target_str)
    idx = np.argmin(np.abs(all_dates - target))
    return all_dates[idx]


def get_crisis_smile(underlying, mat_months, period_label):
    """Return (moneyness_arr, relative_iv_arr) for a given period."""
    kind, d1, d2 = CRISIS_PERIODS[period_label]

    if kind == "calm":
        mask = (
            (df["Underlying"] == underlying) &
            (df["Maturity_months"] == mat_months) &
            (df["Date"] >= d1) & (df["Date"] <= d2)
        )
        sub = df[mask].groupby("Moneyness")["IV_pct"].mean().reset_index()
    else:
        target_date = nearest_date(d1)
        mask = (
            (df["Underlying"] == underlying) &
            (df["Maturity_months"] == mat_months) &
            (df["Date"] == target_date)
        )
        sub = df[mask].groupby("Moneyness")["IV_pct"].mean().reset_index()

    sub = sub.sort_values("Moneyness")
    if sub.empty or 1.000 not in sub["Moneyness"].values:
        return None, None

    atm_iv = sub.loc[sub["Moneyness"] == 1.000, "IV_pct"].values[0]
    return sub["Moneyness"].values, (sub["IV_pct"] - atm_iv).values


# 2 rows (CO1, CL1) × 6 columns (maturities)
fig, axes = plt.subplots(2, 6, figsize=(18, 6), sharey="row")
fig.suptitle(
    "Implied Volatility Smile Distortion During Market Stress  —  "
    r"$\sigma(m) - \sigma_{\mathrm{ATM}}$ [vol pts]",
    fontsize=12, fontweight="bold"
)

for row_idx, underlying in enumerate(["CO1", "CL1"]):
    for col_idx, mat in enumerate(MATURITIES_MONTHS):
        ax = axes[row_idx][col_idx]

        for period_label in CRISIS_PERIODS:
            mon, rel = get_crisis_smile(underlying, mat, period_label)
            if mon is None:
                continue
            st = CRISIS_STYLE[period_label]
            ax.plot(mon, rel,
                    color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                    label=period_label if (row_idx == 0 and col_idx == 0) else "_nolegend_",
                    zorder=3)

        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        ax.set_xticks([0.90, 1.00, 1.10])
        ax.set_xticklabels(["90%", "ATM", "110%"], fontsize=7)
        ax.grid(True, alpha=0.2)

        if col_idx == 0:
            ax.set_ylabel(f"{STYLE[underlying]['label']}\nIV−IV$_{{\\rm ATM}}$ (vol pts)",
                          fontsize=8)
        if row_idx == 0:
            ax.set_title(f"{mat}M", fontsize=9, fontweight="bold")
        if row_idx == 1:
            ax.set_xlabel("Moneyness", fontsize=7)

# Single shared legend at the top
handles = [
    plt.Line2D([0], [0],
               color=CRISIS_STYLE[p]["color"],
               linestyle=CRISIS_STYLE[p]["ls"],
               linewidth=1.8,
               label=p)
    for p in CRISIS_PERIODS
]
fig.legend(handles=handles, loc="upper right", fontsize=8,
           bbox_to_anchor=(0.99, 0.97), framealpha=0.85)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("data_plots/crisis_smile_distortion.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data_plots/crisis_smile_distortion.png")


# ---------------------------------------------------------------------------
# Figure C: risk_reversal_ts.png
# Risk reversal IV(1.10) − IV(0.90) at 1M and 3M, 2006–2026
# ---------------------------------------------------------------------------
print("\nComputing risk reversal time series...")

EVENT_ANNOTATIONS = [
    ("2008-09-15", "GFC\n2008",           "firebrick"),
    ("2014-11-01", "Supply\nGlut\n2014",  "darkorange"),
    ("2020-03-20", "COVID-19\n2020",      "purple"),
    ("2022-03-07", "Ukraine\n2022",       "navy"),
]


def compute_rr(underlying, mat_months):
    """Return a Date-indexed Series of IV(1.10) - IV(0.90)."""
    wing_hi = (
        df[(df["Underlying"] == underlying) &
           (df["Maturity_months"] == mat_months) &
           (df["Moneyness"] == 1.10)]
        .set_index("Date")["IV_pct"]
    )
    wing_lo = (
        df[(df["Underlying"] == underlying) &
           (df["Maturity_months"] == mat_months) &
           (df["Moneyness"] == 0.90)]
        .set_index("Date")["IV_pct"]
    )
    rr = (wing_hi - wing_lo).dropna().sort_index()
    return rr


fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle(
    "Brent and WTI 1M and 3M Risk Reversal  —  "
    r"$\sigma(110\%) - \sigma(90\%)$ [vol pts]",
    fontsize=12, fontweight="bold"
)

for ax, underlying in zip(axes, ["CO1", "CL1"]):
    color = STYLE[underlying]["color"]
    label = STYLE[underlying]["label"]

    rr_1m = compute_rr(underlying, 1)
    rr_3m = compute_rr(underlying, 3)

    ax.plot(rr_1m.index, rr_1m.values,
            color=color, linewidth=0.8, linestyle="-",  label="1M", alpha=0.9)
    ax.plot(rr_3m.index, rr_3m.values,
            color=color, linewidth=0.8, linestyle="--", label="3M", alpha=0.7)

    ax.axhline(0, color="gray", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_ylabel(r"$\sigma(110\%) - \sigma(90\%)$ [vol pts]")
    ax.set_title(label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    ylim_top = ax.get_ylim()[1]
    for date_str, text, color_ev in EVENT_ANNOTATIONS:
        dt = pd.Timestamp(date_str)
        if rr_1m.index.min() <= dt <= rr_1m.index.max():
            ax.axvline(dt, color=color_ev, linewidth=0.9, linestyle="--", alpha=0.7)
            ax.text(dt, ylim_top * 0.88, text,
                    color=color_ev, fontsize=7.5, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
plt.savefig("data_plots/risk_reversal_ts.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data_plots/risk_reversal_ts.png")

print("\nAll 3 figures saved to data_plots/.")
