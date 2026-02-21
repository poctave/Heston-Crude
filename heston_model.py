"""
Heston (1993) stochastic volatility model — pricing and calibration.

Model dynamics:
    dS = mu*S*dt + sqrt(V)*S*dW1
    dV = kappa*(theta - V)*dt + sigma*sqrt(V)*dW2
    corr(dW1, dW2) = rho

Parameters:
    kappa  : mean-reversion speed of variance
    theta  : long-run variance  (long-run vol = sqrt(theta))
    sigma  : volatility of volatility
    rho    : spot-variance correlation  (typically negative for commodities)
    v0     : initial variance  (initial vol = sqrt(v0))

References:
    Heston (1993), Review of Financial Studies 6(2), 327-343.
    Carr & Madan (1999), Journal of Computational Finance 2(4), 61-73.
    Lord & Kahl (2010), Mathematical Finance 20(4), 671-694.
"""

import numpy as np
from scipy.optimize import brentq, differential_evolution, minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Characteristic function  (Lord & Kahl 2010 numerically stable rotation)
# ---------------------------------------------------------------------------

def heston_char_func(u, S, T, r, kappa, theta, sigma, rho, v0):
    """
    Characteristic function of log(S_T) under the Heston model.

    Uses the Lord & Kahl (2010) formulation to avoid branch-cut
    discontinuities in complex logarithms for large |u| or long T.

    Parameters
    ----------
    u     : complex array  — integration frequencies
    S     : float          — current futures price
    T     : float          — time to maturity in years
    r     : float          — risk-free rate (decimal)
    kappa, theta, sigma, rho, v0 : Heston parameters

    Returns
    -------
    phi : complex array of same shape as u
    """
    i = 1j
    xi = kappa - rho * sigma * i * u
    d = np.sqrt(xi**2 + sigma**2 * u * (u + i))

    # Lord & Kahl rotation: use r_minus to stay on the correct branch
    g = (xi - d) / (xi + d)
    exp_dT = np.exp(-d * T)

    # Avoid division by zero when g*exp_dT == 1
    denom = 1.0 - g * exp_dT
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)

    C = kappa * (
        (xi - d) * T - 2.0 * np.log(denom / (1.0 - g))
    ) / sigma**2

    D = ((xi - d) / sigma**2) * (1.0 - exp_dT) / denom

    phi = np.exp(C * theta + D * v0 + i * u * (np.log(S) + r * T))
    return phi


# ---------------------------------------------------------------------------
# Black-76 formula for futures options
# ---------------------------------------------------------------------------

def black76_call(F, K, T, r, sigma):
    """European call price on a futures contract (Black 1976)."""
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0) * np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black76_put(F, K, T, r, sigma):
    """European put price on a futures contract (Black 1976)."""
    if sigma <= 0 or T <= 0:
        return max(K - F, 0.0) * np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_implied_vol(price, F, K, T, r, option_type="C"):
    """
    Invert Black-76 to implied volatility using Brent's root-finding method.

    Returns np.nan if no solution is found (deep ITM/OTM or bad data).
    """
    if option_type.upper() == "C":
        pricer = lambda sig: black76_call(F, K, T, r, sig) - price
        intrinsic = max(F - K, 0.0) * np.exp(-r * T)
    else:
        pricer = lambda sig: black76_put(F, K, T, r, sig) - price
        intrinsic = max(K - F, 0.0) * np.exp(-r * T)

    if price <= intrinsic + 1e-8:
        return np.nan

    try:
        return brentq(pricer, 1e-6, 5.0, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------------------
# Carr-Madan FFT pricer
# ---------------------------------------------------------------------------

def heston_call_price_fft(S, K_array, T, r, kappa, theta, sigma, rho, v0,
                           N=4096, eta=0.25, alpha=1.5):
    """
    Price European calls at an array of strikes using the Carr-Madan (1999) FFT.

    One FFT call prices the entire strike grid — critical for calibration speed.

    Parameters
    ----------
    S       : float        — current futures/spot price
    K_array : array-like   — array of strikes
    T       : float        — maturity in years
    r       : float        — risk-free rate
    N, eta, alpha : FFT parameters (defaults work for most calibrations)

    Returns
    -------
    prices : np.ndarray of call prices, same length as K_array
    """
    K_array = np.asarray(K_array, dtype=float)

    # Frequency grid
    v = np.arange(N) * eta
    v[0] = 1e-14  # avoid division by zero at v=0

    # Characteristic function evaluated on the Carr-Madan contour
    u = v - (alpha + 1) * 1j
    phi = heston_char_func(u, S, T, r, kappa, theta, sigma, rho, v0)

    # Carr-Madan dampened call transform
    psi = (np.exp(-r * T) * phi /
           (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v))

    # Simpson's rule weights for integration accuracy
    simpson = (3 + (-1)**np.arange(N) - np.where(np.arange(N) == 0, 1, 0)) / 3.0
    x = np.exp(1j * np.arange(N) * np.pi) * psi * simpson * eta

    # FFT
    fft_vals = np.real(np.fft.fft(x))

    # Log-strike grid produced by FFT
    lambda_ = 2 * np.pi / (N * eta)
    b = N * lambda_ / 2
    log_strikes_fft = -b + lambda_ * np.arange(N)

    # Recover call prices from FFT output
    call_prices_fft = np.exp(-alpha * log_strikes_fft) / np.pi * fft_vals

    # Interpolate to requested strikes
    log_K = np.log(K_array)
    prices = np.interp(log_K, log_strikes_fft, call_prices_fft)

    # Floor at intrinsic value
    intrinsic = np.maximum(S - K_array, 0.0) * np.exp(-r * T)
    prices = np.maximum(prices, intrinsic)

    return prices


# ---------------------------------------------------------------------------
# Model implied volatilities
# ---------------------------------------------------------------------------

def heston_implied_vol(params, S, K_array, T_array, r):
    """
    Compute Heston model implied Black-76 vols for each (K, T) pair.

    Parameters
    ----------
    params  : (kappa, theta, sigma, rho, v0)
    S       : float     — spot/futures price
    K_array : np.array  — strikes
    T_array : np.array  — maturities in years (same length as K_array)
    r       : float     — risk-free rate

    Returns
    -------
    ivols : np.array of implied vols (np.nan where inversion fails)
    """
    kappa, theta, sigma, rho, v0 = params
    K_array = np.asarray(K_array, dtype=float)
    T_array = np.asarray(T_array, dtype=float)
    ivols = np.full(len(K_array), np.nan)

    for T in np.unique(T_array):
        mask = T_array == T
        K_sub = K_array[mask]
        prices = heston_call_price_fft(S, K_sub, T, r, kappa, theta, sigma, rho, v0)
        for j, (K, price) in enumerate(zip(K_sub, prices)):
            iv = black76_implied_vol(price, S, K, T, r, option_type="C")
            ivols[np.where(mask)[0][j]] = iv

    return ivols


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_objective(params, S, K_array, T_array, r, market_ivols, weights=None):
    """
    Weighted mean-squared error between Heston and market implied vols.

    Weights default to vega-weighting (ATM options weighted more),
    following Cont & Tankov (2004).
    """
    model_ivols = heston_implied_vol(params, S, K_array, T_array, r)
    valid = ~np.isnan(model_ivols) & ~np.isnan(market_ivols)

    if valid.sum() < 3:
        return 1e6  # not enough valid points

    diff = model_ivols[valid] - market_ivols[valid]

    if weights is None:
        # Vega-like weights: highest at ATM (K ~ S)
        moneyness = K_array[valid] / S
        weights_use = np.exp(-0.5 * ((moneyness - 1.0) / 0.1)**2)
    else:
        weights_use = weights[valid]

    return np.sum(weights_use * diff**2) / np.sum(weights_use)


def calibrate_heston(S, K_array, T_array, r, market_ivols, weights=None):
    """
    Calibrate Heston parameters to market implied volatilities.

    Uses differential_evolution for global search followed by L-BFGS-B
    for local refinement.

    Parameters
    ----------
    S            : float     — current futures price
    K_array      : np.array  — option strikes
    T_array      : np.array  — option maturities in years
    r            : float     — risk-free rate
    market_ivols : np.array  — market Black-76 implied vols
    weights      : np.array or None — optional per-observation weights

    Returns
    -------
    result : dict with keys:
        params        — (kappa, theta, sigma, rho, v0)
        kappa, theta, sigma, rho, v0 — individual calibrated values
        mse           — final weighted MSE
        rmse_volpts   — RMSE in vol points (percentage)
        feller        — True if 2*kappa*theta >= sigma^2
        feller_value  — 2*kappa*theta - sigma^2
        message       — optimizer convergence message
    """
    bounds = [
        (0.01, 20.0),    # kappa
        (0.001, 2.0),    # theta
        (0.01, 2.0),     # sigma
        (-0.999, 0.999), # rho
        (0.001, 2.0),    # v0
    ]

    obj = lambda p: calibration_objective(p, S, K_array, T_array, r, market_ivols, weights)

    # --- Global search ---
    print("[calibrate] Running global search (differential_evolution)...")
    de_result = differential_evolution(
        obj, bounds,
        maxiter=300, tol=1e-7,
        seed=42, workers=1, polish=False,
        popsize=15, mutation=(0.5, 1.5), recombination=0.7
    )

    # --- Local refinement ---
    print("[calibrate] Refining with L-BFGS-B...")
    local_result = minimize(
        obj, de_result.x,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8}
    )

    kappa, theta, sigma, rho, v0 = local_result.x
    mse = local_result.fun
    rmse_volpts = np.sqrt(mse) * 100  # convert to vol points (%)

    feller_val = 2 * kappa * theta - sigma**2

    return {
        "params": local_result.x,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho,
        "v0": v0,
        "mse": mse,
        "rmse_volpts": rmse_volpts,
        "feller": feller_val >= 0,
        "feller_value": feller_val,
        "message": local_result.message,
    }


# ---------------------------------------------------------------------------
# Quick self-test (run: python3 heston_model.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Heston model self-test ===")

    # Test parameters (from Heston 1993, Table 1 approximately)
    S, K, T, r = 100.0, 100.0, 1.0, 0.0
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04

    # FFT price
    price_fft = heston_call_price_fft(S, np.array([K]), T, r, kappa, theta, sigma, rho, v0)[0]
    iv_fft = black76_implied_vol(price_fft, S, K, T, r)
    print(f"ATM call price (FFT) : {price_fft:.4f}")
    print(f"Implied vol from FFT : {iv_fft:.4f}  (should be ~0.20)")

    # Feller condition check
    feller = 2 * kappa * theta - sigma**2
    print(f"Feller condition 2κθ - σ² = {feller:.4f}  ({'SATISFIED' if feller >= 0 else 'VIOLATED'})")
    print("Self-test complete.")
