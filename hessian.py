"""
hessian.py — Hessian spectral geometry analysis for the Heston calibration landscape.

Computes the 5×5 Hessian of the weighted-MSE calibration loss L(Θ) at the optimal
parameter vector Θ̂ via central finite differences, then decomposes it spectrally
(H = V Λ Vᵀ) to identify stiff and sloppy parameter directions.

A rolling version repeats this across monthly snapshots from 2006 to 2026.

Functions
---------
compute_hessian          : numerical Hessian via central finite differences
spectral_decomposition   : eigendecomposition of the Hessian
static_hessian_analysis  : static analysis for a single date/underlying
run_rolling_analysis     : monthly rolling pipeline with checkpointing

References
----------
Transtrum, Machta & Sethna (2010), Phys. Rev. Lett. 104, 060201.
    Origin of "stiff / sloppy" terminology in nonlinear least-squares.
Lord & Kahl (2010), Mathematical Finance 20(4), 671-694.
    Numerically stable Heston characteristic function.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import heston_model
import surface_loader

PARAM_NAMES = ["kappa", "theta", "sigma", "rho", "v0"]

# ---------------------------------------------------------------------------
# 1.  Numerical Hessian
# ---------------------------------------------------------------------------

def compute_hessian(params, S, K_array, T_array, r, market_ivols,
                    weights=None, rel_eps=1e-4):
    """
    Compute the 5×5 Hessian of the Heston calibration loss L(Θ) at `params`
    via central finite differences.

    Parameters
    ----------
    params       : array-like, length 5 — (kappa, theta, sigma, rho, v0)
    S            : float  — current futures price
    K_array      : np.ndarray — option strikes
    T_array      : np.ndarray — option maturities in years
    r            : float  — risk-free rate
    market_ivols : np.ndarray — market Black-76 implied vols (decimal)
    weights      : np.ndarray or None — per-observation weights
    rel_eps      : float  — relative step size (adaptive per parameter)

    Returns
    -------
    H : np.ndarray, shape (5, 5), symmetric Hessian matrix
    """
    params = np.asarray(params, dtype=float)
    n = len(params)

    # Adaptive step: ε_i = max(1e-5, |θ_i| × rel_eps)
    eps = np.maximum(1e-5, np.abs(params) * rel_eps)

    def L(p):
        return heston_model.calibration_objective(
            p, S, K_array, T_array, r, market_ivols, weights
        )

    L0 = L(params)
    H  = np.zeros((n, n))

    # --- Diagonal elements ---
    for i in range(n):
        ei        = np.zeros(n); ei[i] = eps[i]
        L_plus    = L(params + ei)
        L_minus   = L(params - ei)
        H[i, i]   = (L_plus - 2.0 * L0 + L_minus) / eps[i]**2

    # --- Off-diagonal elements (symmetric) ---
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = eps[i]
            ej = np.zeros(n); ej[j] = eps[j]
            Lpp = L(params + ei + ej)
            Lpm = L(params + ei - ej)
            Lmp = L(params - ei + ej)
            Lmm = L(params - ei - ej)
            H[i, j] = (Lpp - Lpm - Lmp + Lmm) / (4.0 * eps[i] * eps[j])
            H[j, i] = H[i, j]

    return H


# ---------------------------------------------------------------------------
# 2.  Spectral decomposition
# ---------------------------------------------------------------------------

def spectral_decomposition(H):
    """
    Eigendecompose a symmetric Hessian matrix H = V Λ Vᵀ.

    Eigenvalues are sorted in **descending** order (largest = stiffest first).

    Parameters
    ----------
    H : np.ndarray, shape (5, 5)

    Returns
    -------
    dict with keys:
        eigenvalues    : np.ndarray (5,) — sorted descending
        eigenvectors   : np.ndarray (5,5) — columns are eigenvectors
        condition_number : float — |λ_max| / max(|λ_min|, 1e-12)
        is_psd         : bool — True if all eigenvalues ≥ 0
    """
    # eigh exploits symmetry and guarantees real eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Sort descending (stiffest direction first)
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    lambda_max = eigenvalues[0]
    lambda_min = eigenvalues[-1]
    condition_number = abs(lambda_max) / max(abs(lambda_min), 1e-12)

    return {
        "eigenvalues":     eigenvalues,
        "eigenvectors":    eigenvectors,
        "condition_number": condition_number,
        "is_psd":          bool(lambda_min >= -1e-8),
    }


# ---------------------------------------------------------------------------
# 3.  Static analysis helper (single date / underlying)
# ---------------------------------------------------------------------------

def static_hessian_analysis(underlying, params, S, K_array, T_array, r,
                             market_ivols, weights=None, verbose=True):
    """
    Compute and print the Hessian spectral decomposition for a single snapshot.

    Returns (H, spec_dict).
    """
    params = np.asarray(params, dtype=float)

    print(f"\n[static] Computing Hessian for {underlying} ...")
    H    = compute_hessian(params, S, K_array, T_array, r, market_ivols, weights)
    spec = spectral_decomposition(H)

    if verbose:
        print(f"\n  Hessian matrix ({underlying}):")
        print(f"  {'':>6} " + "  ".join(f"{p:>8}" for p in PARAM_NAMES))
        for i, pi in enumerate(PARAM_NAMES):
            row = "  ".join(f"{H[i, j]:>8.3e}" for j in range(5))
            print(f"  {pi:>6}  {row}")

        print(f"\n  Eigenvalue spectrum ({underlying}):")
        print(f"  {'Mode':<5} {'Eigenvalue':>14}  {'Dominant loading':>18}  "
              f"{'Eigenvector (κ, θ, σ, ρ, v0)'}")
        for k in range(5):
            ev  = spec["eigenvalues"][k]
            vec = spec["eigenvectors"][:, k]
            dom = PARAM_NAMES[np.argmax(np.abs(vec))]
            vec_str = "  ".join(f"{v:+.3f}" for v in vec)
            print(f"  λ_{k+1:<3} {ev:>14.4e}  {dom:>18}  [{vec_str}]")

        kH = spec["condition_number"]
        print(f"\n  Condition number κ_H = {kH:.3e}")
        print(f"  Positive semi-definite: {spec['is_psd']}")

    return H, spec


# ---------------------------------------------------------------------------
# 4.  Rolling analysis
# ---------------------------------------------------------------------------

def _nearest_date(target, available_dates):
    """Snap target date to the nearest date present in available_dates."""
    available = pd.to_datetime(available_dates)
    diffs = np.abs((available - pd.Timestamp(target)).total_seconds())
    idx = np.argmin(diffs)
    return available[idx]


def _fast_de_calibrate(S, K_array, T_array, r, market_ivols):
    """
    Rapid differential evolution calibration for rolling analysis.
    Uses smaller population and fewer iterations than the full calibration.
    """
    from scipy.optimize import differential_evolution
    bounds = [
        (0.01, 20.0),    # kappa
        (0.001, 2.0),    # theta
        (0.01, 2.0),     # sigma
        (-0.999, 0.999), # rho
        (0.001, 2.0),    # v0
    ]
    obj = lambda p: heston_model.calibration_objective(
        p, S, K_array, T_array, r, market_ivols
    )
    de_res = differential_evolution(
        obj, bounds,
        maxiter=80, tol=1e-5,
        seed=42, workers=1, polish=True,
        popsize=8, mutation=(0.5, 1.5), recombination=0.7,
    )
    rmse = np.sqrt(de_res.fun) * 100
    return de_res.x, rmse


def _calibrate_local(params0, S, K_array, T_array, r, market_ivols,
                     rmse_threshold=8.0):
    """
    Local-only calibration: L-BFGS-B warm-started from params0.
    Falls back to fast DE if RMSE exceeds threshold (vol pts).
    """
    bounds = [
        (0.01, 20.0),    # kappa
        (0.001, 2.0),    # theta
        (0.01, 2.0),     # sigma
        (-0.999, 0.999), # rho
        (0.001, 2.0),    # v0
    ]
    obj = lambda p: heston_model.calibration_objective(
        p, S, K_array, T_array, r, market_ivols
    )

    res = minimize(obj, params0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7})

    rmse = np.sqrt(res.fun) * 100
    if rmse > rmse_threshold:
        # Bad local minimum — run fast global search
        return _fast_de_calibrate(S, K_array, T_array, r, market_ivols)

    return res.x, rmse


def run_rolling_analysis(df, underlyings=("CO1", "CL1"),
                         output_csv="data_plots/rolling_hessian.csv",
                         sample_freq="MS", verbose=True):
    """
    Roll the Hessian spectral analysis across monthly snapshots 2006–2026.

    Parameters
    ----------
    df           : DataFrame from surface_loader.build_options_df()
    underlyings  : list of underlying tickers to process
    output_csv   : path to checkpoint / output CSV
    sample_freq  : pandas date offset for sampling (default "MS" = month-start)
    verbose      : print progress

    Returns
    -------
    results_df : DataFrame with all rolling results
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # --- Load existing checkpoint ---
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv, parse_dates=["Date"])
        done_keys = set(zip(existing["Date"].dt.date.astype(str),
                            existing["Underlying"]))
        print(f"[rolling] Loaded {len(existing)} existing rows from {output_csv}")
    else:
        existing  = pd.DataFrame()
        done_keys = set()

    # --- Monthly target dates ---
    all_dates = pd.to_datetime(df["Date"].sort_values().unique())
    target_dates = pd.date_range(all_dates.min(), all_dates.max(), freq=sample_freq)

    # Filter moneyness to [0.70, 1.30] (same as calibrate.py)
    moneyness = df["Strike"] / df["SpotPrice"]
    df = df[(moneyness >= 0.70) & (moneyness <= 1.30)].copy()

    new_rows = []
    checkpoint_counter = 0

    for underlying in underlyings:
        df_u = df[df["Underlying"] == underlying]
        avail = pd.to_datetime(df_u["Date"].sort_values().unique())
        prev_params = None

        for i, tgt in enumerate(target_dates):
            snap_date = _nearest_date(tgt, avail)
            key = (str(snap_date.date()), underlying)
            if key in done_keys:
                continue

            sub = df_u[df_u["Date"] == snap_date].dropna(subset=["ImpliedVol"])
            if len(sub) < 6:
                if verbose:
                    print(f"  [{underlying}] {snap_date.date()} — skipped "
                          f"(only {len(sub)} obs)")
                continue

            S          = sub["SpotPrice"].iloc[0]
            r          = sub["RiskFreeRate"].iloc[0]
            K_array    = sub["Strike"].values
            T_array    = sub["Maturity"].values
            IV_market  = sub["ImpliedVol"].values

            # --- Calibration ---
            if prev_params is None:
                if verbose:
                    print(f"  [{underlying}] {snap_date.date()} — fast DE (first date)")
                params_t, rmse_t = _fast_de_calibrate(
                    S, K_array, T_array, r, IV_market
                )
            else:
                params_t, rmse_t = _calibrate_local(
                    prev_params, S, K_array, T_array, r, IV_market
                )
                if verbose:
                    print(f"  [{underlying}] {snap_date.date()} — warm-start  "
                          f"RMSE={rmse_t:.2f}%")

            prev_params = params_t.copy()

            # --- Hessian ---
            try:
                H    = compute_hessian(params_t, S, K_array, T_array, r, IV_market)
                spec = spectral_decomposition(H)
            except Exception as exc:
                print(f"  [{underlying}] {snap_date.date()} — Hessian failed: {exc}")
                continue

            evs = spec["eigenvalues"]
            ev1 = spec["eigenvectors"][:, 0]   # stiffest
            ev5 = spec["eigenvectors"][:, -1]  # sloppiest
            kappa, theta, sigma, rho, v0 = params_t
            feller = (2 * kappa * theta - sigma**2) >= 0

            row = {
                "Date":       snap_date,
                "Underlying": underlying,
                "kappa": kappa, "theta": theta, "sigma": sigma,
                "rho":   rho,   "v0":    v0,
                "rmse_volpts":    rmse_t,
                "feller":         feller,
                "lambda1": evs[0], "lambda2": evs[1], "lambda3": evs[2],
                "lambda4": evs[3], "lambda5": evs[4],
                "condition_number": spec["condition_number"],
                # Stiffest eigenvector
                "ev1_kappa": ev1[0], "ev1_theta": ev1[1], "ev1_sigma": ev1[2],
                "ev1_rho":   ev1[3], "ev1_v0":    ev1[4],
                # Sloppiest eigenvector
                "ev5_kappa": ev5[0], "ev5_theta": ev5[1], "ev5_sigma": ev5[2],
                "ev5_rho":   ev5[3], "ev5_v0":    ev5[4],
            }
            new_rows.append(row)
            checkpoint_counter += 1

            # --- Checkpoint every 10 new rows ---
            if checkpoint_counter % 10 == 0:
                chunk = pd.DataFrame(new_rows)
                combined = pd.concat([existing, chunk], ignore_index=True)
                combined.to_csv(output_csv, index=False)
                if verbose:
                    print(f"  [checkpoint] Saved {len(combined)} rows → {output_csv}")

    # --- Final save ---
    if new_rows:
        chunk = pd.DataFrame(new_rows)
        results_df = pd.concat([existing, chunk], ignore_index=True)
    else:
        results_df = existing.copy()

    results_df = results_df.sort_values(["Underlying", "Date"]).reset_index(drop=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\n[rolling] Done. {len(results_df)} total rows saved to {output_csv}")
    return results_df


# ---------------------------------------------------------------------------
# Self-test / quick static run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  Hessian spectral analysis — static test (Feb 2026)")
    print("=" * 60)

    # Load existing calibrated params from CSV
    results_dir = "results"
    all_static  = {}
    for und in ["CO1", "CL1"]:
        csv_path = os.path.join(results_dir, f"{und}_calibration.csv")
        if not os.path.exists(csv_path):
            print(f"  {csv_path} not found — run calibrate.py first.")
            sys.exit(1)
        row = pd.read_csv(csv_path).iloc[0]
        all_static[und] = row

    # Load market data for the calibration date
    print("\nLoading Bloomberg surface data ...")
    df_full = surface_loader.build_options_df()
    moneyness = df_full["Strike"] / df_full["SpotPrice"]
    df_full = df_full[(moneyness >= 0.70) & (moneyness <= 1.30)].copy()

    cal_date = df_full["Date"].max()
    print(f"Using calibration date: {cal_date.date()}\n")

    static_results = {}
    for und in ["CO1", "CL1"]:
        row = all_static[und]
        params = np.array([row["kappa"], row["theta"], row["sigma"],
                           row["rho"],   row["v0"]])

        sub = df_full[(df_full["Underlying"] == und) &
                      (df_full["Date"] == cal_date)].dropna(subset=["ImpliedVol"])
        S         = sub["SpotPrice"].iloc[0]
        r         = sub["RiskFreeRate"].iloc[0]
        K_array   = sub["Strike"].values
        T_array   = sub["Maturity"].values
        IV_market = sub["ImpliedVol"].values

        H, spec = static_hessian_analysis(und, params, S, K_array, T_array, r,
                                          IV_market, verbose=True)
        static_results[und] = {"H": H, "spec": spec, "params": params,
                                "S": S, "K": K_array, "T": T_array,
                                "r": r, "IV": IV_market}

    # Symmetry check
    for und, res in static_results.items():
        H = res["H"]
        asym = np.max(np.abs(H - H.T))
        print(f"\n  [{und}] Symmetry check: max|H - Hᵀ| = {asym:.2e}  "
              f"({'PASS' if asym < 1e-8 else 'FAIL'})")

    print("\nStatic test complete. Run plot_hessian_chapter.py for figures.")
