"""
rolling_svj_hessian.py — Rolling 8×8 SVJ Hessian spectral analysis.

Mirrors hessian.py but for the Bates (1996) SVJ model (8 parameters).

Strategy (identical warm-start logic to hessian.py):
  • First date per underlying  : lite Differential Evolution (maxiter=50, popsize=8)
                                  initialised from Heston rolling params + jump defaults
  • Subsequent dates           : L-BFGS-B warm-started from previous SVJ params
  • Fallback (RMSE > 8 vpts)   : re-run lite DE

Output
------
data_plots/rolling_svj_hessian.csv  (checkpointed every 10 rows)

Columns
-------
Date, Underlying,
kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j,
rmse_volpts, feller,
lambda1 … lambda8,
condition_number,
ev1_kappa … ev1_sigma_j,    (stiffest eigenvector, 8 components)
ev8_kappa … ev8_sigma_j     (sloppiest eigenvector, 8 components)

Run with:
    MPLBACKEND=Agg PYTHONUNBUFFERED=1 conda run --no-capture-output python rolling_svj_hessian.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

import surface_loader
import svj_model

warnings.filterwarnings("ignore")

OUTPUT_CSV   = "data_plots/rolling_svj_hessian.csv"
ROLLING_CSV  = "data_plots/rolling_hessian.csv"   # Heston rolling (for warm-start init)
RMSE_THRESHOLD = 8.0    # vol pts; fallback to DE if warm-start exceeds this

SVJ_BOUNDS = [
    (0.01, 20.0),    # kappa
    (0.001, 2.0),    # theta
    (0.01,  2.0),    # sigma (ξ)
    (-0.999, 0.999), # rho
    (0.001, 2.0),    # v0
    (0.01, 10.0),    # lam
    (-2.0,  2.0),    # mu_j
    (0.01,  2.0),    # sigma_j
]

# Default jump parameters used when warm-starting from Heston
DEFAULT_JUMP_PARAMS = np.array([0.50, -0.10, 0.15])   # [lam, mu_j, sigma_j]

PARAM_KEYS = ["kappa", "theta", "sigma", "rho", "v0", "lam", "mu_j", "sigma_j"]


# ---------------------------------------------------------------------------
# Calibration helpers (mirror hessian.py pattern)
# ---------------------------------------------------------------------------

def _fast_de_svj(S, K_array, T_array, r, market_ivols, init=None):
    """
    Lite Differential Evolution for SVJ.  Used on first date (per underlying)
    or as fallback when warm-start quality is poor.
    """
    obj = lambda p: svj_model.calibration_objective_svj(
        p, S, K_array, T_array, r, market_ivols
    )

    de_res = differential_evolution(
        obj, SVJ_BOUNDS,
        maxiter=50, tol=1e-5,
        seed=42, workers=1, polish=False,
        popsize=8, mutation=(0.5, 1.5), recombination=0.7,
        init="latinhypercube" if init is None else "latinhypercube",
    )

    # Local refinement after DE
    local = minimize(obj, de_res.x, method="L-BFGS-B", bounds=SVJ_BOUNDS,
                     options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-8})

    best = local.x if local.fun < de_res.fun else de_res.x
    rmse = np.sqrt(min(local.fun, de_res.fun)) * 100
    return best, rmse


def _warm_start_svj(params0, S, K_array, T_array, r, market_ivols):
    """
    L-BFGS-B from previous date's SVJ params.
    Falls back to lite DE if RMSE exceeds RMSE_THRESHOLD.
    """
    obj = lambda p: svj_model.calibration_objective_svj(
        p, S, K_array, T_array, r, market_ivols
    )

    res = minimize(obj, params0, method="L-BFGS-B", bounds=SVJ_BOUNDS,
                   options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7})

    rmse = np.sqrt(res.fun) * 100
    if rmse > RMSE_THRESHOLD:
        return _fast_de_svj(S, K_array, T_array, r, market_ivols)

    return res.x, rmse


def _nearest_date(target, available_dates):
    """Snap target to nearest available date (same helper as hessian.py)."""
    available = pd.to_datetime(available_dates)
    diffs = np.abs((available - pd.Timestamp(target)).total_seconds())
    return available[np.argmin(diffs)]


# ---------------------------------------------------------------------------
# Main rolling loop
# ---------------------------------------------------------------------------

def run_rolling_svj(df, underlyings=("CO1", "CL1"),
                    output_csv=OUTPUT_CSV,
                    sample_freq="MS", verbose=True):
    """
    Roll the SVJ Hessian analysis across monthly snapshots.

    Parameters
    ----------
    df          : DataFrame from surface_loader.build_options_df()
    underlyings : tickers to process
    output_csv  : checkpoint / output path
    sample_freq : pandas date offset (default "MS" = month-start)
    verbose     : print progress
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # --- Checkpoint ---
    if os.path.exists(output_csv):
        existing  = pd.read_csv(output_csv, parse_dates=["Date"])
        done_keys = set(zip(existing["Date"].dt.date.astype(str),
                            existing["Underlying"]))
        print(f"[rolling_svj] Loaded {len(existing)} existing rows from {output_csv}")
    else:
        existing  = pd.DataFrame()
        done_keys = set()

    # --- Load Heston rolling params for warm-start initialisation ---
    heston_rolling = None
    if os.path.exists(ROLLING_CSV):
        heston_rolling = pd.read_csv(ROLLING_CSV, parse_dates=["Date"])
        print(f"[rolling_svj] Loaded Heston rolling CSV ({len(heston_rolling)} rows)")

    # --- Monthly dates ---
    moneyness = df["Strike"] / df["SpotPrice"]
    df = df[(moneyness >= 0.70) & (moneyness <= 1.30)].copy()

    all_dates    = pd.to_datetime(df["Date"].sort_values().unique())
    target_dates = pd.date_range(all_dates.min(), all_dates.max(), freq=sample_freq)

    new_rows          = []
    checkpoint_counter = 0

    for underlying in underlyings:
        df_u  = df[df["Underlying"] == underlying]
        avail = pd.to_datetime(df_u["Date"].sort_values().unique())
        prev_svj_params = None

        for i, tgt in enumerate(target_dates):
            snap_date = _nearest_date(tgt, avail)
            key = (str(snap_date.date()), underlying)
            if key in done_keys:
                continue

            sub = df_u[df_u["Date"] == snap_date].dropna(subset=["ImpliedVol"])
            if len(sub) < 6:
                if verbose:
                    print(f"  [{underlying}] {snap_date.date()} — skip ({len(sub)} obs)")
                continue

            S         = sub["SpotPrice"].iloc[0]
            r         = sub["RiskFreeRate"].iloc[0]
            K_array   = sub["Strike"].values
            T_array   = sub["Maturity"].values
            IV_market = sub["ImpliedVol"].values

            # --- Calibrate ---
            if prev_svj_params is None:
                # First date: try to init from Heston rolling + default jumps
                if heston_rolling is not None:
                    h_row = heston_rolling[
                        (heston_rolling["Date"] == snap_date) &
                        (heston_rolling["Underlying"] == underlying)
                    ]
                    if h_row.empty:
                        # Find nearest Heston row
                        h_sub = heston_rolling[heston_rolling["Underlying"] == underlying]
                        idx   = (h_sub["Date"] - snap_date).abs().idxmin()
                        h_row = h_sub.loc[[idx]]
                    heston_p = np.array([
                        h_row.iloc[0]["kappa"], h_row.iloc[0]["theta"],
                        h_row.iloc[0]["sigma"], h_row.iloc[0]["rho"],
                        h_row.iloc[0]["v0"]
                    ])
                    init_params = np.concatenate([heston_p, DEFAULT_JUMP_PARAMS])
                    # Clip to bounds
                    for k, (lo, hi) in enumerate(SVJ_BOUNDS):
                        init_params[k] = np.clip(init_params[k], lo, hi)
                else:
                    init_params = None

                if verbose:
                    print(f"  [{underlying}] {snap_date.date()} — lite DE (first date)")
                params_t, rmse_t = _fast_de_svj(
                    S, K_array, T_array, r, IV_market, init=init_params
                )
            else:
                params_t, rmse_t = _warm_start_svj(
                    prev_svj_params, S, K_array, T_array, r, IV_market
                )
                if verbose:
                    print(f"  [{underlying}] {snap_date.date()} "
                          f"RMSE={rmse_t:.2f}%")

            prev_svj_params = params_t.copy()

            # --- 8×8 Hessian ---
            try:
                H    = svj_model.compute_svj_hessian(
                    params_t, S, K_array, T_array, r, IV_market
                )
                spec = svj_model.spectral_decomposition_svj(H)
            except Exception as exc:
                print(f"  [{underlying}] {snap_date.date()} — Hessian failed: {exc}")
                continue

            evs = spec["eigenvalues"]         # shape (8,) descending
            ev1 = spec["eigenvectors"][:, 0]  # stiffest
            ev8 = spec["eigenvectors"][:, -1] # sloppiest

            kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j = params_t
            feller = (2 * kappa * theta - sigma**2) >= 0

            row = {
                "Date":       snap_date,
                "Underlying": underlying,
                "kappa": kappa, "theta": theta, "sigma": sigma,
                "rho":   rho,   "v0":    v0,
                "lam":   lam,   "mu_j":  mu_j,  "sigma_j": sigma_j,
                "rmse_volpts":      rmse_t,
                "feller":           feller,
                "lambda1": evs[0], "lambda2": evs[1], "lambda3": evs[2],
                "lambda4": evs[3], "lambda5": evs[4], "lambda6": evs[5],
                "lambda7": evs[6], "lambda8": evs[7],
                "condition_number": spec["condition_number"],
                # Stiffest eigenvector (8 components)
                "ev1_kappa": ev1[0], "ev1_theta": ev1[1], "ev1_sigma": ev1[2],
                "ev1_rho":   ev1[3], "ev1_v0":    ev1[4],
                "ev1_lam":   ev1[5], "ev1_mu_j":  ev1[6], "ev1_sigma_j": ev1[7],
                # Sloppiest eigenvector (8 components)
                "ev8_kappa": ev8[0], "ev8_theta": ev8[1], "ev8_sigma": ev8[2],
                "ev8_rho":   ev8[3], "ev8_v0":    ev8[4],
                "ev8_lam":   ev8[5], "ev8_mu_j":  ev8[6], "ev8_sigma_j": ev8[7],
            }
            new_rows.append(row)
            checkpoint_counter += 1

            # --- Checkpoint every 10 new rows ---
            if checkpoint_counter % 10 == 0:
                chunk    = pd.DataFrame(new_rows)
                combined = pd.concat([existing, chunk], ignore_index=True)
                combined.to_csv(output_csv, index=False)
                if verbose:
                    print(f"  [checkpoint] {len(combined)} rows → {output_csv}")

    # --- Final save ---
    if new_rows:
        chunk      = pd.DataFrame(new_rows)
        results_df = pd.concat([existing, chunk], ignore_index=True)
    else:
        results_df = existing.copy()

    results_df = results_df.sort_values(
        ["Underlying", "Date"]
    ).reset_index(drop=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\n[rolling_svj] Done. {len(results_df)} rows → {output_csv}")
    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[rolling_svj] Loading Bloomberg surface data...")
    df = surface_loader.build_options_df()
    run_rolling_svj(df, underlyings=("CO1", "CL1"), verbose=True)
