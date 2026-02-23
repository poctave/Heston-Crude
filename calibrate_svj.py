"""
calibrate_svj.py — Bates (1996) SVJ calibration for crude oil futures options.

Calibrates the 8-parameter SVJ model to CO1 (Brent) and CL1 (WTI) on the
static analysis date 2021-12-01, using the same Bloomberg OVDV surface data
and moneyness filter as calibrate.py.

Outputs (in results/ folder):
    - {underlying}_svj_calibration.csv  : 8 calibrated SVJ parameters
    - {underlying}_svj_smile.png        : smile cross-sections: market / Heston / SVJ

Run with:
    MPLBACKEND=Agg PYTHONUNBUFFERED=1 conda run python calibrate_svj.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import surface_loader
import heston_model
import svj_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CALIBRATION_DATE = "2021-12-01"
UNDERLYINGS      = ["CO1", "CL1"]
OUTPUT_DIR       = "results"
ROLLING_CSV      = "data_plots/rolling_hessian.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading Bloomberg surface data...")
options_df = surface_loader.build_options_df()
moneyness  = options_df["Strike"] / options_df["SpotPrice"]
options_df = options_df[(moneyness >= 0.7) & (moneyness <= 1.3)].copy()

cal_date = pd.Timestamp(CALIBRATION_DATE)
print(f"Calibration date: {cal_date.date()}")

# Load rolling Hessian CSV for Heston reference parameters
rolling = pd.read_csv(ROLLING_CSV, parse_dates=["Date"])

# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------

all_results = {}

for underlying in UNDERLYINGS:
    print(f"\n{'='*60}")
    print(f" Calibrating SVJ model — {underlying} ({'Brent' if underlying == 'CO1' else 'WTI'})")
    print(f"{'='*60}")

    sub = options_df[
        (options_df["Underlying"] == underlying) &
        (options_df["Date"] == cal_date)
    ].dropna(subset=["ImpliedVol"]).copy()

    if len(sub) < 6:
        print(f"  Not enough data for {underlying} on {cal_date.date()} "
              f"(found {len(sub)} rows). Skipping.")
        continue

    S = sub["SpotPrice"].iloc[0]
    r = sub["RiskFreeRate"].iloc[0] if "RiskFreeRate" in sub else 0.053
    K_array  = sub["Strike"].values
    T_array  = sub["Maturity"].values
    IV_market = sub["ImpliedVol"].values

    print(f"  Spot price : {S:.2f} USD")
    print(f"  Options    : {len(sub)} observations across {sub['Maturity'].nunique()} maturities")

    # Retrieve Heston reference parameters from rolling CSV
    heston_row = rolling[(rolling["Date"] == cal_date) &
                         (rolling["Underlying"] == underlying)]
    if heston_row.empty:
        print(f"  WARNING: No Heston entry in rolling CSV for {underlying} on {cal_date.date()}")
        heston_params = None
        heston_rmse   = np.nan
    else:
        h = heston_row.iloc[0]
        heston_params = np.array([h["kappa"], h["theta"], h["sigma"], h["rho"], h["v0"]])
        heston_rmse   = h["rmse_volpts"]

    # Run SVJ calibration
    result = svj_model.calibrate_svj(S, K_array, T_array, r, IV_market)
    all_results[underlying] = result

    # Print results table
    label = "Brent (CO1)" if underlying == "CO1" else "WTI (CL1)"
    print(f"\n  === {label} SVJ Calibration Results ===")
    print(f"  {'Parameter':<12} {'Value':>10}   Description")
    print(f"  {'-'*55}")
    print(f"  {'kappa':<12} {result['kappa']:>10.4f}   Mean-reversion speed")
    print(f"  {'theta':<12} {result['theta']:>10.4f}   Long-run variance "
          f"({np.sqrt(result['theta'])*100:.1f}% long-run vol)")
    print(f"  {'sigma':<12} {result['sigma']:>10.4f}   Vol of vol (ξ)")
    print(f"  {'rho':<12} {result['rho']:>10.4f}   Spot-vol correlation")
    print(f"  {'v0':<12} {result['v0']:>10.4f}   Initial variance "
          f"({np.sqrt(result['v0'])*100:.1f}% initial vol)")
    print(f"  {'lam':<12} {result['lam']:>10.4f}   Jump intensity (λ)")
    print(f"  {'mu_j':<12} {result['mu_j']:>10.4f}   Mean log-jump size (μ_J)")
    print(f"  {'sigma_j':<12} {result['sigma_j']:>10.4f}   Jump size volatility (σ_J)")
    print(f"\n  Feller condition (2κθ ≥ ξ²) : "
          f"{'SATISFIED' if result['feller'] else 'VIOLATED'} "
          f"(value: {result['feller_value']:.4f})")
    print(f"  Heston RMSE (reference)    : {heston_rmse:.2f} vol pts")
    print(f"  SVJ    RMSE                : {result['rmse_volpts']:.2f} vol pts")
    if result['rho'] > 0:
        print(f"  *** NOTE: rho > 0 (call-skew regime)")

    # Save parameters to CSV
    params_dict = {
        "Underlying": underlying,
        "Date":       cal_date.date(),
        "kappa":      result["kappa"],
        "theta":      result["theta"],
        "sigma":      result["sigma"],
        "rho":        result["rho"],
        "v0":         result["v0"],
        "lam":        result["lam"],
        "mu_j":       result["mu_j"],
        "sigma_j":    result["sigma_j"],
        "mse":        result["mse"],
        "rmse_volpts": result["rmse_volpts"],
        "feller_satisfied": result["feller"],
        "heston_rmse_volpts": heston_rmse,
    }
    csv_path = os.path.join(OUTPUT_DIR, f"{underlying}_svj_calibration.csv")
    pd.DataFrame([params_dict]).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Plot: smile cross-sections — market / Heston / SVJ
    # -----------------------------------------------------------------------
    maturities = sorted(sub["Maturity"].unique())
    ncols = min(len(maturities), 3)
    nrows = (len(maturities) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(
        f"{label} — Smile Fit: Market / Heston / SVJ  "
        f"({cal_date.date()})\n"
        f"SVJ RMSE={result['rmse_volpts']:.2f} vpts   "
        f"Heston RMSE={heston_rmse:.2f} vpts",
        fontsize=12, fontweight="bold"
    )

    for i, T in enumerate(maturities):
        ax = axes[i // ncols][i % ncols]
        mask = sub["Maturity"] == T
        K_sub = sub.loc[mask, "Strike"].values
        IV_mkt = sub.loc[mask, "ImpliedVol"].values
        mon = K_sub / S

        # SVJ model vols
        IV_svj = svj_model.svj_implied_vol(result["params"], S, K_sub,
                                            np.full_like(K_sub, T), r)

        ax.scatter(mon, IV_mkt * 100, color="steelblue", s=30,
                   label="Market", zorder=5)

        # Heston model vols (if available)
        if heston_params is not None:
            IV_heston = heston_model.heston_implied_vol(
                heston_params, S, K_sub, np.full_like(K_sub, T), r
            )
            ax.plot(mon, IV_heston * 100, color="firebrick", linewidth=1.4,
                    linestyle="--", label="Heston", zorder=4)

        ax.plot(mon, IV_svj * 100, color="darkorange", linewidth=1.6,
                linestyle="-", label="SVJ (Bates)", zorder=6)

        months = round(T * 12)
        ax.set_title(f"T = {months}M", fontsize=9)
        ax.set_xlabel("Moneyness (K/S)", fontsize=8)
        ax.set_ylabel("Implied Vol (%)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i in range(len(maturities), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    plt.tight_layout()
    smile_path = os.path.join(OUTPUT_DIR, f"{underlying}_svj_smile.png")
    plt.savefig(smile_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Smile plot saved: {smile_path}")

print(f"\n{'='*60}")
print(" SVJ calibration complete. Results saved to results/")
print(f"{'='*60}")
