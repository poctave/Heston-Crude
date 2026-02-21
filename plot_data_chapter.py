"""
plot_data_chapter.py — Generate data visualisation figures for Chapter 2.

Outputs (to data_plots/):
  market_surface_brent.png  — 3D market IV surface for CO1 on latest date
  market_surface_wti.png    — 3D market IV surface for CL1 on latest date
  atm_term_structure.png    — ATM term structure + 1M smile (latest date)
  atm_vol_history.png       — Historical ATM 1M vol for CO1 and CL1

Run with:
  MPLBACKEND=Agg PYTHONUNBUFFERED=1 conda run python plot_data_chapter.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import surface_loader

os.makedirs("data_plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading Bloomberg surface data (full history)...")
df = surface_loader.build_options_df()

# Recover moneyness from Strike / SpotPrice and round to 3 decimal places
df["Moneyness"] = (df["Strike"] / df["SpotPrice"]).round(3)
df["IV_pct"]    = df["ImpliedVol"] * 100

# Tenor label map: maturity in years → months (for axis labels)
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

LATEST_DATE = df["Date"].max()
print(f"Latest date: {LATEST_DATE.date()}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STYLE = {
    "CO1": {"color": "seagreen",   "label": "Brent (CO1)"},
    "CL1": {"color": "steelblue",  "label": "WTI (CL1)"},
}

MATURITIES_MONTHS = [1, 2, 3, 6, 12, 24]
MONEYNESS_LABELS  = [0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10]


def get_latest_surface(underlying):
    """Return (moneyness_arr, maturities_arr, IV_pct_grid) for the latest date."""
    sub = df[(df["Underlying"] == underlying) & (df["Date"] == LATEST_DATE)].copy()
    pivot = (
        sub.pivot_table(index="Moneyness", columns="Maturity_months",
                        values="IV_pct", aggfunc="first")
           .reindex(index=sorted(sub["Moneyness"].unique()),
                    columns=sorted(sub["Maturity_months"].unique()))
    )
    mon  = pivot.index.values
    mats = pivot.columns.values
    grid = pivot.values
    return mon, mats, grid


# ---------------------------------------------------------------------------
# Figure 1 & 2: 3D Market Implied Volatility Surface
# ---------------------------------------------------------------------------

for underlying in ["CO1", "CL1"]:
    label = STYLE[underlying]["label"]
    fname = f"data_plots/market_surface_{'brent' if underlying == 'CO1' else 'wti'}.png"

    mon, mats, grid = get_latest_surface(underlying)
    M, K = np.meshgrid(mats, mon)  # M = maturity months, K = moneyness

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(K, M, grid, cmap="viridis", alpha=0.85, edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.1, label="Implied Vol (%)")

    # Scatter the actual market data points
    sub = df[(df["Underlying"] == underlying) & (df["Date"] == LATEST_DATE)]
    ax.scatter(
        sub["Moneyness"], sub["Maturity_months"], sub["IV_pct"],
        color="red", s=30, zorder=10, label="Market data points"
    )

    ax.set_xlabel("Moneyness (K/F)", labelpad=10)
    ax.set_ylabel("Maturity (months)",  labelpad=10)
    ax.set_zlabel("Implied Vol (%)",    labelpad=10)
    ax.set_title(
        f"{label} — Market Implied Volatility Surface ({LATEST_DATE.date()})",
        fontsize=12, fontweight="bold", pad=15
    )
    ax.set_xticks([0.90, 0.95, 1.00, 1.05, 1.10])
    ax.set_yticks(MATURITIES_MONTHS)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 3: ATM Term Structure + 1M Smile (latest date)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Market Implied Volatility Structure — {LATEST_DATE.date()}",
    fontsize=13, fontweight="bold"
)

# Panel 1: ATM (moneyness ≈ 1.00) term structure across maturities
ax = axes[0]
for underlying in ["CO1", "CL1"]:
    sub = df[(df["Underlying"] == underlying) & (df["Date"] == LATEST_DATE)].copy()
    atm = sub[sub["Moneyness"] == 1.000].sort_values("Maturity_months")
    ax.plot(
        atm["Maturity_months"], atm["IV_pct"],
        marker="o", linewidth=1.8, markersize=6,
        color=STYLE[underlying]["color"], label=STYLE[underlying]["label"]
    )

ax.set_xlabel("Maturity (months)")
ax.set_ylabel("ATM Implied Vol (%)")
ax.set_title("ATM Term Structure")
ax.set_xticks(MATURITIES_MONTHS)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Moneyness smile at 1M maturity
ax = axes[1]
for underlying in ["CO1", "CL1"]:
    sub = df[
        (df["Underlying"] == underlying) &
        (df["Date"] == LATEST_DATE) &
        (df["Maturity_months"] == 1)
    ].sort_values("Moneyness")
    ax.plot(
        sub["Moneyness"] * 100, sub["IV_pct"],
        marker="o", linewidth=1.8, markersize=6,
        color=STYLE[underlying]["color"], label=STYLE[underlying]["label"]
    )

ax.set_xlabel("Moneyness K/F (%)")
ax.set_ylabel("Implied Vol (%)")
ax.set_title("Volatility Smile — 1-Month Maturity")
ax.set_xticks([90, 95, 97.5, 100, 102.5, 105, 110])
ax.axvline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="ATM")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data_plots/atm_term_structure.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data_plots/atm_term_structure.png")


# ---------------------------------------------------------------------------
# Figure 4: ATM 1M Implied Volatility History
# ---------------------------------------------------------------------------

# Filter to ATM (moneyness = 1.000) at 1M maturity for all dates
atm_hist = df[
    (df["Moneyness"] == 1.000) & (df["Maturity_months"] == 1)
].copy().sort_values("Date")

fig, ax = plt.subplots(figsize=(14, 5))

for underlying in ["CO1", "CL1"]:
    sub = atm_hist[atm_hist["Underlying"] == underlying]
    ax.plot(
        sub["Date"], sub["IV_pct"],
        linewidth=0.7, color=STYLE[underlying]["color"],
        label=STYLE[underlying]["label"]
    )

# Annotate key market events
events = [
    ("2008-09-15", "GFC\n2008",           "firebrick"),
    ("2014-11-01", "Supply\nGlut\n2014",  "darkorange"),
    ("2020-03-20", "COVID-19\n2020",      "purple"),
    ("2022-03-07", "Ukraine\n2022",       "navy"),
]
ylim_top = atm_hist["IV_pct"].quantile(0.995) * 1.05

for date_str, text, color in events:
    dt = pd.Timestamp(date_str)
    if atm_hist["Date"].min() <= dt <= atm_hist["Date"].max():
        ax.axvline(dt, color=color, linewidth=0.9, linestyle="--", alpha=0.7)
        ax.text(
            dt, ylim_top * 0.88, text,
            color=color, fontsize=7.5, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none")
        )

ax.set_xlabel("Date")
ax.set_ylabel("1-Month ATM Implied Vol (%)")
ax.set_title(
    "Brent and WTI 1-Month ATM Implied Volatility — 2006–2026",
    fontsize=12, fontweight="bold"
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data_plots/atm_vol_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data_plots/atm_vol_history.png")

print("\nAll 4 figures saved to data_plots/.")
