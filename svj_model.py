"""
svj_model.py — Bates (1996) Stochastic Volatility with Jumps (SVJ) model.

Extends the Heston (1993) model by adding a compound Poisson log-normal
jump component to the log-price dynamics:

    dS/S = (r - q - λμ̄) dt + sqrt(V) dW^S + (e^J - 1) dN

    dV   = κ(θ - V) dt + ξ sqrt(V) dW^V

    corr(dW^S, dW^V) = ρ
    J ~ N(μ_J, σ_J²)         (log-jump size)
    N_t ~ Poisson(λ)          (jump arrival)
    μ̄ = exp(μ_J + σ_J²/2) - 1  (compensator)

Parameters (8 total):
    kappa   : variance mean-reversion speed
    theta   : long-run variance  (long-run vol = sqrt(theta))
    sigma   : volatility of variance (vol-of-vol, written ξ in the paper)
    rho     : spot-variance correlation  (historically negative for oil)
    v0      : initial variance  (initial vol = sqrt(v0))
    lam     : jump intensity  (expected jumps per year)
    mu_j    : mean log-jump size  (negative → downside jumps)
    sigma_j : jump size volatility

References:
    Bates (1996), Review of Financial Studies 9(1), 69-107.
    Heston (1993), Review of Financial Studies 6(2), 327-343.
    Carr & Madan (1999), Journal of Computational Finance 2(4), 61-73.
    Lord & Kahl (2010), Mathematical Finance 20(4), 671-694.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

import heston_model


# ---------------------------------------------------------------------------
# Bates (1996) characteristic function
# ---------------------------------------------------------------------------

def svj_char_func(u, S, T, r, kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j):
    """
    Characteristic function of log(S_T) under the Bates (1996) SVJ model.

    Factorises as:
        φ_SVJ(u) = φ_Heston(u) × exp(jump_term)

    where the jump correction is:
        jump_term = λT [exp(iuμ_J - ½σ_J²u²) - 1 - iu(exp(μ_J + ½σ_J²) - 1)]

    Parameters
    ----------
    u       : complex array — integration frequencies
    S       : float  — current futures price
    T       : float  — time to maturity in years
    r       : float  — risk-free rate (decimal)
    kappa, theta, sigma, rho, v0 : Heston parameters
    lam     : float  — jump intensity (λ)
    mu_j    : float  — mean log-jump size (μ_J)
    sigma_j : float  — jump size volatility (σ_J)

    Returns
    -------
    phi : complex array of same shape as u
    """
    # Heston component
    phi_heston = heston_model.heston_char_func(
        u, S, T, r, kappa, theta, sigma, rho, v0
    )

    # Jump correction (Bates 1996, eq. 11)
    # E[e^{iuJ}] = exp(iu*mu_j - 0.5*sigma_j^2*u^2)
    cf_jump = np.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2)

    # Compensator: mu_bar = exp(mu_j + 0.5*sigma_j^2) - 1
    mu_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    # Jump term contribution to CF
    jump_term = lam * T * (cf_jump - 1.0 - 1j * u * mu_bar)

    return phi_heston * np.exp(jump_term)


# ---------------------------------------------------------------------------
# Carr-Madan FFT pricer (SVJ)
# ---------------------------------------------------------------------------

def svj_call_price_fft(S, K_array, T, r, kappa, theta, sigma, rho, v0,
                       lam, mu_j, sigma_j,
                       N=4096, eta=0.25, alpha=1.5):
    """
    Price a strip of European calls under the SVJ model via the
    Carr-Madan (1999) FFT method.

    Parameters
    ----------
    S       : float  — current futures price
    K_array : array  — option strikes
    T       : float  — maturity in years
    r       : float  — risk-free rate
    kappa, theta, sigma, rho, v0 : Heston parameters
    lam, mu_j, sigma_j           : jump parameters
    N, eta, alpha                : FFT grid parameters

    Returns
    -------
    prices : array of call prices, same length as K_array
    """
    K_array = np.asarray(K_array, dtype=float)

    # FFT grid
    lam_fft = 2 * np.pi / (N * eta)
    b = 0.5 * N * lam_fft
    u_arr = np.arange(N) * eta

    # Modified characteristic function for the damped call
    i = 1j
    phi_args = u_arr - (alpha + 1) * i
    phi = svj_char_func(phi_args, S, T, r, kappa, theta, sigma, rho, v0,
                        lam, mu_j, sigma_j)

    # Denominator: alpha^2 + alpha - u^2 + i*(2*alpha+1)*u
    denom = alpha**2 + alpha - u_arr**2 + i * (2 * alpha + 1) * u_arr
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)

    psi = np.exp(-r * T) * phi / denom

    # Simpson weights
    weights = np.ones(N)
    weights[0] = 1 / 3
    weights[-1] = 1 / 3
    weights[1:-1:2] = 4 / 3
    weights[2:-2:2] = 2 / 3

    # FFT
    x = psi * np.exp(i * b * u_arr) * eta * weights
    fft_vals = np.real(np.fft.fft(x))

    # Log-strike grid
    k_grid = -b + lam_fft * np.arange(N)
    call_vals = np.exp(-alpha * k_grid) / np.pi * fft_vals

    # Interpolate to requested strikes
    K_arr = np.log(K_array)
    prices = np.interp(K_arr, k_grid, call_vals)
    prices = np.maximum(prices, 0.0)
    return prices


# ---------------------------------------------------------------------------
# Implied volatility (SVJ → Black-76)
# ---------------------------------------------------------------------------

def svj_implied_vol(params, S, K_array, T_array, r):
    """
    Convert SVJ call prices to Black-76 implied vols.

    Parameters
    ----------
    params  : array-like, length 8 — (kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j)
    S       : float
    K_array : np.ndarray of strikes
    T_array : np.ndarray of maturities
    r       : float

    Returns
    -------
    ivols : np.ndarray, same shape as K_array (NaN where inversion fails)
    """
    kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j = params
    K_array = np.asarray(K_array, dtype=float)
    T_array = np.asarray(T_array, dtype=float)
    ivols = np.full(len(K_array), np.nan)

    for T in np.unique(T_array):
        mask = T_array == T
        K_sub = K_array[mask]
        prices = svj_call_price_fft(S, K_sub, T, r,
                                    kappa, theta, sigma, rho, v0,
                                    lam, mu_j, sigma_j)
        for j, (K, price) in enumerate(zip(K_sub, prices)):
            iv = heston_model.black76_implied_vol(price, S, K, T, r,
                                                  option_type="C")
            ivols[np.where(mask)[0][j]] = iv

    return ivols


# ---------------------------------------------------------------------------
# Calibration objective
# ---------------------------------------------------------------------------

def calibration_objective_svj(params, S, K_array, T_array, r, market_ivols,
                               weights=None):
    """
    Weighted MSE between SVJ and market Black-76 implied vols.

    Weights default to vega-weighting (ATM options weighted more),
    following Cont & Tankov (2004), consistent with heston_model.py.
    """
    model_ivols = svj_implied_vol(params, S, K_array, T_array, r)
    valid = ~np.isnan(model_ivols) & ~np.isnan(market_ivols)

    if valid.sum() < 3:
        return 1e6

    diff = model_ivols[valid] - market_ivols[valid]

    if weights is None:
        moneyness = K_array[valid] / S
        weights_use = np.exp(-0.5 * ((moneyness - 1.0) / 0.1)**2)
    else:
        weights_use = weights[valid]

    return np.sum(weights_use * diff**2) / np.sum(weights_use)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_svj(S, K_array, T_array, r, market_ivols, weights=None):
    """
    Calibrate Bates (1996) SVJ parameters to market implied volatilities.

    Two-stage: differential_evolution (global) + L-BFGS-B (local).

    Parameters
    ----------
    S            : float     — current futures price
    K_array      : np.array  — option strikes
    T_array      : np.array  — option maturities in years
    r            : float     — risk-free rate
    market_ivols : np.array  — market Black-76 implied vols
    weights      : np.array or None

    Returns
    -------
    result : dict with keys:
        params        — (kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j)
        kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j — individual values
        mse           — final weighted MSE
        rmse_volpts   — RMSE in vol points (percentage)
        feller        — True if 2*kappa*theta >= sigma^2
        feller_value  — 2*kappa*theta - sigma^2
        message       — optimizer convergence message
    """
    bounds = [
        (0.01, 20.0),    # kappa
        (0.001, 2.0),    # theta
        (0.01, 2.0),     # sigma  (vol-of-vol ξ)
        (-0.999, 0.999), # rho
        (0.001, 2.0),    # v0
        (0.01, 10.0),    # lam    (jump intensity)
        (-2.0,  2.0),    # mu_j   (mean log-jump)
        (0.01,  2.0),    # sigma_j (jump size vol)
    ]

    obj = lambda p: calibration_objective_svj(
        p, S, K_array, T_array, r, market_ivols, weights
    )

    # --- Global search ---
    print("[calibrate_svj] Running global search (differential_evolution)...")
    de_result = differential_evolution(
        obj, bounds,
        maxiter=400, tol=1e-7,
        seed=42, workers=1, polish=False,
        popsize=15, mutation=(0.5, 1.5), recombination=0.7
    )

    # --- Local refinement ---
    print("[calibrate_svj] Refining with L-BFGS-B...")
    local_result = minimize(
        obj, de_result.x,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8}
    )

    kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j = local_result.x
    mse = local_result.fun
    rmse_volpts = np.sqrt(mse) * 100

    feller_val = 2 * kappa * theta - sigma**2

    return {
        "params":    local_result.x,
        "kappa":     kappa,
        "theta":     theta,
        "sigma":     sigma,
        "rho":       rho,
        "v0":        v0,
        "lam":       lam,
        "mu_j":      mu_j,
        "sigma_j":   sigma_j,
        "mse":       mse,
        "rmse_volpts": rmse_volpts,
        "feller":    feller_val >= 0,
        "feller_value": feller_val,
        "message":   local_result.message,
    }


# ---------------------------------------------------------------------------
# Numerical Hessian (8×8) for SVJ
# ---------------------------------------------------------------------------

def compute_svj_hessian(params, S, K_array, T_array, r, market_ivols,
                        weights=None, rel_eps=1e-4):
    """
    Compute the 8×8 Hessian of the SVJ calibration loss L(Θ) at `params`
    via central finite differences.

    Parameters
    ----------
    params       : array-like, length 8 — (κ, θ, ξ, ρ, v₀, λ, μ_J, σ_J)
    S, K_array, T_array, r, market_ivols : calibration data
    weights      : optional per-observation weights
    rel_eps      : relative step size for finite differences

    Returns
    -------
    H : np.ndarray, shape (8, 8), symmetric
    """
    params = np.asarray(params, dtype=float)
    n = len(params)
    eps = np.maximum(1e-5, np.abs(params) * rel_eps)

    def L(p):
        return calibration_objective_svj(
            p, S, K_array, T_array, r, market_ivols, weights
        )

    L0 = L(params)
    H  = np.zeros((n, n))

    for i in range(n):
        ei = np.zeros(n); ei[i] = eps[i]
        H[i, i] = (L(params + ei) - 2.0 * L0 + L(params - ei)) / eps[i]**2

    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = eps[i]
            ej = np.zeros(n); ej[j] = eps[j]
            H[i, j] = (
                L(params + ei + ej) - L(params + ei - ej)
                - L(params - ei + ej) + L(params - ei - ej)
            ) / (4.0 * eps[i] * eps[j])
            H[j, i] = H[i, j]

    return H


def spectral_decomposition_svj(H):
    """
    Eigendecomposition of the 8×8 SVJ Hessian.

    Returns dict with eigenvalues (descending), eigenvectors, and condition number.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pos_eigs = eigenvalues[eigenvalues > 0]
    cond = float(pos_eigs[0] / pos_eigs[-1]) if len(pos_eigs) >= 2 else np.nan

    return {
        "eigenvalues":  eigenvalues,
        "eigenvectors": eigenvectors,
        "condition_number": cond,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("SVJ model self-test (synthetic data)...")
    np.random.seed(42)

    S0  = 75.0
    r0  = 0.04
    T0  = 0.25

    # True parameters (moderate negative skew + downside jumps)
    TRUE = dict(kappa=2.0, theta=0.04, sigma=0.40, rho=-0.50, v0=0.04,
                lam=1.0, mu_j=-0.05, sigma_j=0.10)

    K_test = S0 * np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    T_test = np.full_like(K_test, T0)

    ivols = svj_implied_vol(list(TRUE.values()), S0, K_test, T_test, r0)
    print("Implied vols:", np.round(ivols * 100, 2))
    print("Self-test passed." if not np.any(np.isnan(ivols)) else "WARNING: NaN ivols")
