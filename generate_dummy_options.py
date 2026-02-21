"""
Generates realistic dummy options data for testing the Heston calibration pipeline.

Implied vols are produced by the Heston model itself (with known parameters)
plus small random noise — so the calibration should recover values close to
the true parameters below.
"""

import numpy as np
import pandas as pd
import heston_model

np.random.seed(42)

# --- True Heston parameters used to generate the data ---
TRUE_PARAMS = {
    "CO1": dict(kappa=3.0, theta=0.09, sigma=0.40, rho=-0.60, v0=0.10),  # Brent
    "CL1": dict(kappa=2.5, theta=0.08, sigma=0.35, rho=-0.55, v0=0.09),  # WTI
}

SPOT_PRICES = {"CO1": 82.0, "CL1": 78.5}
RISK_FREE   = 0.053
CAL_DATE    = "2024-01-15"

# Strikes: 7 per maturity (moneyness 0.80 to 1.20)
MATURITIES  = [1/12, 3/12, 6/12, 1.0]   # 1M, 3M, 6M, 12M

rows = []

for underlying, p in TRUE_PARAMS.items():
    S = SPOT_PRICES[underlying]
    params = (p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"])

    for T in MATURITIES:
        # Strikes centred on spot, moneyness 0.82 to 1.18
        strikes = np.round(S * np.linspace(0.82, 1.18, 7), 1)

        # Model implied vols
        iv_model = heston_model.heston_implied_vol(
            params, S, strikes, np.full_like(strikes, T), RISK_FREE
        )

        # Add small bid-ask noise (±0.5 vol pts)
        noise = np.random.normal(0, 0.005, size=len(strikes))
        iv_noisy = np.clip(iv_model + noise, 0.05, 1.5)

        for K, iv in zip(strikes, iv_noisy):
            rows.append({
                "Date":         CAL_DATE,
                "Underlying":   underlying,
                "Strike":       K,
                "Maturity":     round(T, 6),
                "OptionType":   "C",
                "ImpliedVol":   round(iv, 5),
                "OptionPrice":  "",          # not needed since ImpliedVol is provided
                "SpotPrice":    S,
                "RiskFreeRate": RISK_FREE,
            })

df = pd.DataFrame(rows)
df.to_csv("options_data.csv", index=False)

print(f"Generated {len(df)} dummy option rows -> options_data.csv")
print(df.head(10).to_string(index=False))
print("\nTrue parameters used:")
for u, p in TRUE_PARAMS.items():
    print(f"  {u}: kappa={p['kappa']}, theta={p['theta']}, sigma={p['sigma']}, "
          f"rho={p['rho']}, v0={p['v0']}")
