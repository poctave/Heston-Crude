"""
macro_analysis.py — Chapter 6 data pipeline.

Provides:
  load_macro_indicators()  — monthly macro panel (VIX, OVX, GPR, Inventories, DXY)
  merge_with_rolling()     — join macro panel with rolling_hessian.csv
  classify_regimes()       — rule-based regime labels
  run_ols_regression()     — HAC OLS: log10(kappa_H) ~ macro indicators

Run as __main__ to print diagnostics (regime counts + regression summaries).
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE   = "data/Data-Hardcodded.xlsx"
ROLLING_CSV = "data_plots/rolling_hessian.csv"


# ---------------------------------------------------------------------------
# 1.  Load and resample macro indicators
# ---------------------------------------------------------------------------

def _parse_sheet(sheet, value_col, value_name):
    """Read one sheet (skip 2 header rows, take date + value columns)."""
    df = pd.read_excel(DATA_FILE, sheet_name=sheet, header=None, skiprows=2)
    df = df.iloc[:, [1, value_col]].copy()
    df.columns = ["Date", value_name]
    df["Date"]       = pd.to_datetime(df["Date"], errors="coerce")
    df[value_name]   = pd.to_numeric(df[value_name], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def load_macro_indicators():
    """
    Load all five macro sheets and resample to monthly frequency (MS = month-start).

    Returns
    -------
    macro : DataFrame, index=Date (monthly MS), columns:
            VIX, OVX, GPR, Inventories, dInventory, dInventory_z, DXY
    """
    vix  = _parse_sheet("VIX",         2, "VIX")
    ovx  = _parse_sheet("OVX",         2, "OVX")
    gpr  = _parse_sheet("GPR",         2, "GPR")
    inv  = _parse_sheet("Inventories", 2, "Inventories")
    dxy  = _parse_sheet("DXY",         2, "DXY")

    # Set DatetimeIndex for resampling
    for df in (vix, ovx, gpr, inv, dxy):
        df.set_index("Date", inplace=True)

    # Monthly aggregation
    vix_m  = vix["VIX"].resample("MS").mean()
    ovx_m  = ovx["OVX"].resample("MS").mean()          # NaN before May 2007
    gpr_m  = gpr["GPR"].resample("MS").mean()
    inv_m  = inv["Inventories"].resample("MS").last()   # last weekly obs in month
    dxy_m  = dxy["DXY"].resample("MS").mean()

    macro = pd.concat([vix_m, ovx_m, gpr_m, inv_m, dxy_m], axis=1)
    macro.index.name = "Date"

    # Inventory change (month-over-month) and z-score
    macro["dInventory"]   = macro["Inventories"].diff()
    mu  = macro["dInventory"].mean()
    sig = macro["dInventory"].std()
    macro["dInventory_z"] = (macro["dInventory"] - mu) / sig

    # Log GPR for regression (skewed distribution)
    macro["logGPR"] = np.log(macro["GPR"])

    return macro.reset_index()


# ---------------------------------------------------------------------------
# 2.  Merge macro with rolling Hessian results
# ---------------------------------------------------------------------------

def _load_rolling(rmse_max=8.0, cn_max=1e8):
    """Load and quality-filter rolling_hessian.csv."""
    if not os.path.exists(ROLLING_CSV):
        print(f"ERROR: {ROLLING_CSV} not found.  Run hessian.py first.")
        sys.exit(1)
    df = pd.read_csv(ROLLING_CSV, parse_dates=["Date"])
    df = df[df["rmse_volpts"] <= rmse_max].copy()
    df["condition_number"] = df["condition_number"].clip(upper=cn_max)
    df = df.sort_values(["Underlying", "Date"]).reset_index(drop=True)
    return df


def merge_with_rolling(rmse_max=8.0, cn_max=1e8):
    """
    Join rolling Hessian results with monthly macro indicators.

    Each rolling date is snapped to the nearest macro month-start (tolerance ±15 d).

    Returns
    -------
    merged : DataFrame with all rolling columns plus macro columns and log_cn.
    """
    roll  = _load_rolling(rmse_max, cn_max)
    macro = load_macro_indicators()

    roll["log_cn"] = np.log10(roll["condition_number"])

    # merge_asof requires sorted keys
    roll  = roll.sort_values("Date")
    macro = macro.sort_values("Date")

    merged = pd.merge_asof(
        roll, macro,
        on="Date",
        direction="nearest",
        tolerance=pd.Timedelta("15d"),
    )
    merged = merged.sort_values(["Underlying", "Date"]).reset_index(drop=True)

    n_macro_nan = merged["VIX"].isna().sum()
    print(f"[macro_analysis] Merged {len(merged)} rows. "
          f"{n_macro_nan} have missing macro data (dropped for regression).")
    return merged


# ---------------------------------------------------------------------------
# 3.  Regime classification
# ---------------------------------------------------------------------------

REGIME_ORDER = ["Calm", "Geopolitical", "Financial Stress", "Oil Stress", "Compound"]

REGIME_COLORS = {
    "Calm":             "white",
    "Geopolitical":     "#4e79a7",   # muted blue
    "Financial Stress": "#f28e2b",   # amber
    "Oil Stress":       "#e15759",   # salmon-red
    "Compound":         "#b07aa1",   # purple
}

# Threshold constants (exposed for LaTeX table)
THRESH_VIX = 30.0
THRESH_OVX = 60.0
THRESH_GPR = 200.0


def classify_regimes(df):
    """
    Add a 'regime' column using rule-based thresholds.

    Priority (highest wins):
      5 Compound         VIX > 30  AND  OVX > 60
      4 Oil Stress       OVX > 60
      3 Financial Stress VIX > 30
      2 Geopolitical     GPR > 200
      1 Calm             (default)

    Parameters
    ----------
    df : DataFrame with columns VIX, OVX, GPR (monthly-level values).

    Returns
    -------
    df with added 'regime' column (string).
    """
    df = df.copy()
    regime = pd.Series("Calm", index=df.index)

    regime[df["GPR"] > THRESH_GPR]                               = "Geopolitical"
    regime[df["VIX"] > THRESH_VIX]                               = "Financial Stress"
    regime[df["OVX"] > THRESH_OVX]                               = "Oil Stress"
    regime[(df["VIX"] > THRESH_VIX) & (df["OVX"] > THRESH_OVX)] = "Compound"

    df["regime"] = pd.Categorical(regime, categories=REGIME_ORDER, ordered=True)
    return df


# ---------------------------------------------------------------------------
# 4.  OLS regression with HAC standard errors
# ---------------------------------------------------------------------------

def _standardise(df, cols):
    """Return copy with cols standardised (zero mean, unit variance)."""
    df = df.copy()
    for c in cols:
        mu, sg = df[c].mean(), df[c].std()
        df[c + "_std"] = (df[c] - mu) / sg
    return df


REGRESSORS_RAW = ["OVX", "VIX", "logGPR", "dInventory_z", "DXY"]
REGRESSOR_LABELS = {
    "OVX_std":          "OVX",
    "VIX_std":          "VIX",
    "logGPR_std":       "log(GPR)",
    "dInventory_z_std": r"$\Delta$Inventory$_z$",
    "DXY_std":          "DXY",
}


def run_ols_regression(merged, maxlags=3):
    """
    Fit HAC OLS: log10(kappa_H) ~ OVX + VIX + logGPR + dInventory_z + DXY
    Regressors are standardised so coefficients are directly comparable.

    Runs separately for CO1, CL1, and pooled (with CL1 dummy).

    Parameters
    ----------
    merged  : DataFrame from merge_with_rolling() + classify_regimes()
    maxlags : int  — Newey-West lag truncation (default 3)

    Returns
    -------
    dict: {"CO1": RegressionResultsWrapper,
           "CL1": RegressionResultsWrapper,
           "pooled": RegressionResultsWrapper}
    """
    # Drop rows with any NaN in required columns
    req_cols = ["log_cn"] + REGRESSORS_RAW
    df_clean = merged.dropna(subset=req_cols).copy()

    # Standardise regressors
    df_clean = _standardise(df_clean, REGRESSORS_RAW)
    std_cols  = [c + "_std" for c in REGRESSORS_RAW]
    formula   = "log_cn ~ " + " + ".join(std_cols)

    results = {}
    for und in ["CO1", "CL1"]:
        sub = df_clean[df_clean["Underlying"] == und].copy()
        model = smf.ols(formula, data=sub)
        res   = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        results[und] = res
        print(f"\n{'='*60}")
        print(f" OLS Results — {und}  (N={len(sub)}, HAC maxlags={maxlags})")
        print(f"{'='*60}")
        print(res.summary2(float_format="%.4f"))

    # Pooled with CL1 dummy
    df_clean["isCL1"] = (df_clean["Underlying"] == "CL1").astype(float)
    pooled_formula = formula + " + isCL1"
    model_p  = smf.ols(pooled_formula, data=df_clean)
    res_p    = model_p.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    results["pooled"] = res_p
    print(f"\n{'='*60}")
    print(f" OLS Results — Pooled  (N={len(df_clean)}, HAC maxlags={maxlags})")
    print(f"{'='*60}")
    print(res_p.summary2(float_format="%.4f"))

    return results


# ---------------------------------------------------------------------------
# __main__: diagnostic run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[1] Loading macro indicators...")
    macro = load_macro_indicators()
    print(f"  Macro panel: {len(macro)} months, "
          f"{macro['Date'].min().date()} – {macro['Date'].max().date()}")
    print(f"  OVX NaN months: {macro['OVX'].isna().sum()}")
    print(macro[["Date", "VIX", "OVX", "GPR", "dInventory_z", "DXY"]].tail(6).to_string(index=False))

    print("\n[2] Merging with rolling Hessian data...")
    merged = merge_with_rolling()

    print("\n[3] Classifying regimes...")
    merged = classify_regimes(merged)
    print("\nRegime counts (all underlyings):")
    print(merged["regime"].value_counts().reindex(REGIME_ORDER))
    print("\nRegime counts — CO1:")
    print(merged[merged["Underlying"] == "CO1"]["regime"].value_counts().reindex(REGIME_ORDER))
    print("\nRegime counts — CL1:")
    print(merged[merged["Underlying"] == "CL1"]["regime"].value_counts().reindex(REGIME_ORDER))

    print("\n[4] Running OLS regressions...")
    results = run_ols_regression(merged)

    print("\n[5] Summary — R² values:")
    for k, r in results.items():
        print(f"  {k}: R²={r.rsquared:.3f},  adj-R²={r.rsquared_adj:.3f}")
