"""
Heston model calibration script for crude oil futures options.

Calibrates separately for:
    - CO1 (Brent crude)
    - CL1 (WTI crude)

Outputs (in results/ folder):
    - {underlying}_calibration.csv   : calibrated parameters
    - {underlying}_vol_surface.png   : model vs market implied vol surface
    - {underlying}_vol_smile.png     : smile cross-sections per maturity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import data_loader
import heston_model

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

USE_BLOOMBERG_DATA = True       # True = parse real Bloomberg files via surface_loader
                                # False = use legacy options_data.csv

FUTURES_FILE     = "data/Heston-crude-HArdcodded.xlsx"
OPTIONS_FILE     = "data/options_data.csv"
RISK_FREE_RATE   = 0.053        # fallback rate (used only when USE_BLOOMBERG_DATA=False)
CALIBRATION_DATE = None         # None = use most recent date in options data
UNDERLYINGS      = ["CO1", "CL1"]
OUTPUT_DIR       = "results"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
if USE_BLOOMBERG_DATA:
    import surface_loader
    options_df = surface_loader.build_options_df()
    options_df["needs_iv_inversion"] = False   # surface provides ImpliedVol directly
    moneyness = options_df["Strike"] / options_df["SpotPrice"]
    options_df = options_df[(moneyness >= 0.7) & (moneyness <= 1.3)].copy()
else:
    futures_df = data_loader.load_futures(FUTURES_FILE)
    options_df = data_loader.load_options(OPTIONS_FILE)
    options_df = data_loader.merge_spot_into_options(options_df, futures_df)

# Fill missing implied vols by inverting option prices
inv_mask = options_df["needs_iv_inversion"]
if inv_mask.any():
    print(f"\nInverting {inv_mask.sum()} option prices to implied vols...")
    for idx in options_df[inv_mask].index:
        row = options_df.loc[idx]
        iv = heston_model.black76_implied_vol(
            price=row["OptionPrice"],
            F=row["SpotPrice"],
            K=row["Strike"],
            T=row["Maturity"],
            r=row["RiskFreeRate"],
            option_type=row["OptionType"]
        )
        options_df.at[idx, "ImpliedVol"] = iv
    options_df = options_df.dropna(subset=["ImpliedVol"]).copy()

# Select calibration date
dates = data_loader.get_calibration_dates(options_df)
cal_date = pd.Timestamp(CALIBRATION_DATE) if CALIBRATION_DATE else max(dates)
print(f"\nCalibration date: {cal_date.date()}")

# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------

all_results = {}

for underlying in UNDERLYINGS:
    print(f"\n{'='*55}")
    print(f" Calibrating Heston model — {underlying} ({'Brent' if underlying == 'CO1' else 'WTI'})")
    print(f"{'='*55}")

    # Filter options to this underlying and calibration date
    sub = options_df[
        (options_df["Underlying"] == underlying) &
        (options_df["Date"] == cal_date)
    ].dropna(subset=["ImpliedVol"]).copy()

    if len(sub) < 6:
        print(f"  Not enough data for {underlying} on {cal_date.date()} "
              f"(found {len(sub)} rows, need at least 6). Skipping.")
        continue

    S = sub["SpotPrice"].iloc[0]
    r = sub["RiskFreeRate"].iloc[0] if "RiskFreeRate" in sub else RISK_FREE_RATE
    K_array = sub["Strike"].values
    T_array = sub["Maturity"].values
    IV_market = sub["ImpliedVol"].values

    print(f"  Spot price : {S:.2f} USD")
    print(f"  Options    : {len(sub)} observations across {sub['Maturity'].nunique()} maturities")

    # Run calibration
    result = heston_model.calibrate_heston(S, K_array, T_array, r, IV_market)
    all_results[underlying] = result

    # Print results table
    label = "Brent (CO1)" if underlying == "CO1" else "WTI (CL1)"
    print(f"\n  === {label} Heston Calibration Results ===")
    print(f"  {'Parameter':<10} {'Value':>8}   Description")
    print(f"  {'-'*50}")
    print(f"  {'kappa':<10} {result['kappa']:>8.4f}   Mean-reversion speed")
    print(f"  {'theta':<10} {result['theta']:>8.4f}   Long-run variance "
          f"({np.sqrt(result['theta'])*100:.1f}% long-run vol)")
    print(f"  {'sigma':<10} {result['sigma']:>8.4f}   Vol of vol")
    print(f"  {'rho':<10} {result['rho']:>8.4f}   Spot-vol correlation")
    if result['rho'] > 0:
        print(f"  *** NOTE: rho > 0 (call-skew regime — atypical for oil; observed since mid-2022)")
    print(f"  {'v0':<10} {result['v0']:>8.4f}   Initial variance "
          f"({np.sqrt(result['v0'])*100:.1f}% initial vol)")
    print(f"\n  Feller condition (2κθ ≥ σ²) : "
          f"{'SATISFIED' if result['feller'] else 'VIOLATED'} "
          f"(value: {result['feller_value']:.4f})")
    print(f"  Calibration MSE  : {result['mse']:.6f}")
    print(f"  RMSE (vol pts)   : {result['rmse_volpts']:.2f}%")

    # Save parameters to CSV
    params_dict = {
        "Underlying": underlying,
        "Date": cal_date.date(),
        "kappa": result["kappa"],
        "theta": result["theta"],
        "sigma": result["sigma"],
        "rho": result["rho"],
        "v0": result["v0"],
        "mse": result["mse"],
        "rmse_volpts": result["rmse_volpts"],
        "feller_satisfied": result["feller"],
    }
    pd.DataFrame([params_dict]).to_csv(
        os.path.join(OUTPUT_DIR, f"{underlying}_calibration.csv"), index=False
    )

    # -----------------------------------------------------------------------
    # Plot 1: Volatility smile cross-sections per maturity
    # -----------------------------------------------------------------------
    maturities = sorted(sub["Maturity"].unique())
    ncols = min(len(maturities), 3)
    nrows = (len(maturities) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"{label} — Heston Vol Smile (RMSE={result['rmse_volpts']:.2f} vol pts)",
                 fontsize=13, fontweight="bold")

    for i, T in enumerate(maturities):
        ax = axes[i // ncols][i % ncols]
        mask = sub["Maturity"] == T
        K_sub = sub.loc[mask, "Strike"].values
        IV_mkt = sub.loc[mask, "ImpliedVol"].values

        # Model vols for this maturity
        IV_model = heston_model.heston_implied_vol(
            result["params"], S, K_sub, np.full_like(K_sub, T), r
        )

        moneyness = K_sub / S
        ax.scatter(moneyness, IV_mkt * 100, color="steelblue", s=40,
                   label="Market", zorder=5)
        ax.plot(moneyness, IV_model * 100, color="firebrick", linewidth=1.5,
                label="Heston model")
        months = round(T * 12)
        ax.set_title(f"T = {months}M ({T:.3f}y)")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Vol (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(maturities), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    plt.tight_layout()
    smile_path = os.path.join(OUTPUT_DIR, f"{underlying}_vol_smile.png")
    plt.savefig(smile_path, dpi=150)
    plt.show()
    print(f"\n  Smile plot saved to {smile_path}")

    # -----------------------------------------------------------------------
    # Plot 2: 3D volatility surface
    # -----------------------------------------------------------------------
    K_grid = np.linspace(K_array.min(), K_array.max(), 40)
    T_grid = np.linspace(T_array.min(), T_array.max(), 30)
    KK, TT = np.meshgrid(K_grid, T_grid)

    IV_surface = np.zeros_like(KK)
    for j, T in enumerate(T_grid):
        row_ivols = heston_model.heston_implied_vol(
            result["params"], S, K_grid, np.full_like(K_grid, T), r
        )
        IV_surface[j, :] = row_ivols * 100  # to percentage

    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.plot_surface(KK / S, TT, IV_surface, cmap="viridis", alpha=0.75)
    ax3d.scatter(
        K_array / S, T_array, IV_market * 100,
        color="red", s=30, label="Market", zorder=10
    )
    ax3d.set_xlabel("Moneyness (K/S)")
    ax3d.set_ylabel("Maturity (years)")
    ax3d.set_zlabel("Implied Vol (%)")
    ax3d.set_title(f"{label} — Heston Implied Vol Surface")
    ax3d.legend()

    surface_path = os.path.join(OUTPUT_DIR, f"{underlying}_vol_surface.png")
    plt.savefig(surface_path, dpi=150)
    plt.show()
    print(f"  Surface plot saved to {surface_path}")

print(f"\n{'='*55}")
print(" Calibration complete. Results saved to results/")
print(f"{'='*55}")
