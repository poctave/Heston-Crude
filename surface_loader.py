"""
surface_loader.py — Bloomberg vol surface + macro data parser.

Parses:
  - COA_Historical_surface (Brent implied vol surface, 6 tenors × 7 moneyness)
  - CLA_Historical_surface (WTI implied vol surface, same structure)
  - Data-Hardcodded.xlsx (futures prices CO1/CL1, UST yield curve)

Main entry point:
  df = build_options_df()

Returns a DataFrame ready for calibrate.py with columns:
  Date, Underlying, Strike, Maturity, OptionType,
  ImpliedVol, OptionPrice, SpotPrice, RiskFreeRate
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default file paths
# ---------------------------------------------------------------------------

COA_FILE  = "data/COA_Historical_surface - Copy.xlsx"
CLA_FILE  = "data/CLA_Historical_surface - Copy.xlsx"
MACRO_FILE = "data/Data-Hardcodded.xlsx"

# Tenor suffix → maturity in years
TENOR_MAP = {
    "1M":  1 / 12,
    "2M":  2 / 12,
    "3M":  3 / 12,
    "6M":  6 / 12,
    "12M": 1.0,
    "24M": 2.0,
}

# UST tenors to load (in order) and their maturity in years
UST_TENORS      = ["1M", "3M", "6M", "1Y", "2Y"]
UST_TENOR_YEARS = [1 / 12, 3 / 12, 6 / 12, 1.0, 2.0]

# Column pairs (date_col_idx, val_col_idx), 0-based from column A,
# matching Bloomberg BDH multi-series interleaved export pattern:
#   B=Date_1M, C=1M, D=gap, E=Date_3M, F=3M, G=gap, H=Date_6M, ...
UST_COL_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_moneyness(val) -> float:
    """
    Normalize a moneyness column header to a decimal fraction.
    Handles: 0.975, '0.975', '97.5%', 97.5, '97.5'
    Returns np.nan if unparseable.
    """
    try:
        s = str(val).strip().replace('%', '').replace('\xa0', '')
        f = float(s)
        return f / 100 if f > 5 else f  # "97.5" → 0.975; "0.975" → 0.975
    except (ValueError, TypeError):
        return np.nan


def _excel_serial_to_date(series: pd.Series) -> pd.Series:
    """
    Convert Excel serial integers to Timestamps (origin = 1899-12-30).
    Also handles strings that look like dates and actual datetime objects.
    """
    numeric = pd.to_numeric(series, errors='coerce')
    is_numeric = numeric.notna()

    result = pd.Series(pd.NaT, index=series.index)

    # Handle numeric (Excel serial) values
    if is_numeric.any():
        valid_int = numeric[is_numeric].astype(int)
        parsed = pd.to_datetime(valid_int, unit='D', origin='1899-12-30', errors='coerce')
        result[is_numeric] = parsed.values

    # Handle non-numeric (already strings / datetimes)
    if (~is_numeric).any():
        parsed_str = pd.to_datetime(series[~is_numeric], errors='coerce')
        result[~is_numeric] = parsed_str.values

    return pd.to_datetime(result).dt.normalize()  # drop time component


# ---------------------------------------------------------------------------
# Vol surface loader
# ---------------------------------------------------------------------------

def load_vol_surface(filepath: str, underlying_code: str) -> pd.DataFrame:
    """
    Parse one Bloomberg vol surface Excel file (6 sheets × 7 moneyness).

    Parameters
    ----------
    filepath        : path to COA or CLA surface Excel file
    underlying_code : "CO1" (Brent) or "CL1" (WTI)

    Returns
    -------
    DataFrame with columns: Date, Underlying, Maturity, Moneyness, ImpliedVol
      - ImpliedVol in decimal (0.3281 for 32.81%)
      - Maturity in years (0.0833 for 1M)
      - Rows with NaN ImpliedVol already dropped
    """
    prefix = "COA" if underlying_code == "CO1" else "CLA"
    tenors = list(TENOR_MAP.keys())  # ["1M","2M","3M","6M","12M","24M"]
    sheet_names = [f"{prefix} {t}" for t in tenors]

    xls = pd.ExcelFile(filepath)
    available = set(xls.sheet_names)
    frames = []

    for tenor, sheet in zip(tenors, sheet_names):
        if sheet not in available:
            print(f"  [surface_loader] Sheet '{sheet}' not found, skipping.")
            continue

        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        raw = raw.dropna(how='all').reset_index(drop=True)

        if raw.shape[0] < 2:
            print(f"  [surface_loader] Sheet '{sheet}' has too few rows, skipping.")
            continue

        # After dropna(how='all'), the all-NaN Bloomberg metadata row is gone.
        # Row 0: column headers (Date, 0.90, 0.95, ...)
        # Row 1+: data
        header_row = raw.iloc[0]
        data = raw.iloc[1:].reset_index(drop=True)

        # Column 1 (B) = Date; columns 2–8 (C–I) = moneyness implied vols
        date_col_idx = 1
        mon_col_idxs = list(range(2, min(9, raw.shape[1])))

        # Parse moneyness headers and filter valid ones
        moneyness_vals = []
        valid_col_idxs = []
        for ci in mon_col_idxs:
            m = _parse_moneyness(header_row.iloc[ci])
            if not np.isnan(m) and 0.5 <= m <= 2.0:
                moneyness_vals.append(round(m, 4))
                valid_col_idxs.append(ci)

        if not moneyness_vals:
            print(f"  [surface_loader] Sheet '{sheet}': no valid moneyness columns found.")
            continue

        # Parse dates
        dates = _excel_serial_to_date(data.iloc[:, date_col_idx])

        # Build long-format rows: one column per moneyness level
        for m_val, ci in zip(moneyness_vals, valid_col_idxs):
            ivol_pct = pd.to_numeric(data.iloc[:, ci], errors='coerce')
            ivol_dec = ivol_pct / 100  # e.g. 32.81 → 0.3281

            sub = pd.DataFrame({
                "Date":       dates.values,
                "Underlying": underlying_code,
                "Maturity":   TENOR_MAP[tenor],
                "Moneyness":  m_val,
                "ImpliedVol": ivol_dec.values,
            })
            frames.append(sub)

    if not frames:
        raise ValueError(f"No data loaded from {filepath}. Check sheet names.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Date", "ImpliedVol"])
    df = df[df["Date"] >= "2000-01-01"]  # drop any obviously wrong dates
    df = df.sort_values(["Date", "Maturity", "Moneyness"]).reset_index(drop=True)

    print(f"[surface_loader] {underlying_code} surface: {len(df):,} rows | "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Futures price loader
# ---------------------------------------------------------------------------

def load_new_futures(filepath: str) -> pd.DataFrame:
    """
    Parse Brent (CO1) and WTI (CL1) front-month futures prices
    from Data-Hardcodded.xlsx sheets 'Brent Futures' and 'WTI Futures'.

    Returns DataFrame with columns: Date, CO1, CL1
    """

    def _load_sheet(sheet_name, price_col_label):
        raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
        raw = raw.dropna(how='all').reset_index(drop=True)
        # Row 0: Bloomberg metadata; Row 1: column headers; Rows 2+: data
        data = raw.iloc[2:].reset_index(drop=True)
        dates  = pd.to_datetime(data.iloc[:, 1], errors='coerce')
        prices = pd.to_numeric(data.iloc[:, 2], errors='coerce')
        sub = pd.DataFrame({"Date": dates, price_col_label: prices})
        sub = sub.dropna(subset=["Date", price_col_label])
        sub = sub[sub["Date"] >= "1980-01-01"]  # drop corrupt serial-date artefacts
        return sub

    brent = _load_sheet("Brent Futures", "CO1")
    wti   = _load_sheet("WTI Futures",   "CL1")

    futures = (
        brent.merge(wti, on="Date", how="outer")
             .sort_values("Date")
             .reset_index(drop=True)
    )

    print(f"[surface_loader] Futures (CO1/CL1): {len(futures):,} rows | "
          f"{futures['Date'].min().date()} → {futures['Date'].max().date()}")
    return futures


# ---------------------------------------------------------------------------
# UST yield curve loader
# ---------------------------------------------------------------------------

def load_ust_curve(filepath: str) -> pd.DataFrame:
    """
    Parse UST yield curve from the 'USTs' sheet in Data-Hardcodded.xlsx.

    Bloomberg BDH multi-series export: each tenor has its own Date column,
    interleaved in groups of three (Date | Rate | empty gap).

    Returns long-format DataFrame: Date, tenor, rate
      - tenor: one of "1M", "3M", "6M", "1Y", "2Y"
      - rate: decimal (0.04 = 4%)
    """
    raw = pd.read_excel(filepath, sheet_name="USTs", header=None)
    raw = raw.dropna(how='all').reset_index(drop=True)

    # Row 0: Bloomberg metadata; Row 1: column headers; Rows 2+: data
    data = raw.iloc[2:].reset_index(drop=True)

    frames = []
    for tenor, (date_col, val_col) in zip(UST_TENORS, UST_COL_PAIRS):
        if max(date_col, val_col) >= data.shape[1]:
            print(f"  [surface_loader] UST tenor {tenor}: columns {date_col},{val_col} "
                  f"out of range (shape {data.shape}), skipping.")
            continue

        dates = pd.to_datetime(data.iloc[:, date_col], errors='coerce')
        rates = pd.to_numeric(data.iloc[:, val_col], errors='coerce') / 100  # % → decimal

        sub = pd.DataFrame({"Date": dates, "tenor": tenor, "rate": rates})
        sub = sub.dropna(subset=["Date", "rate"])
        sub = sub[sub["Date"] >= "1990-01-01"]
        frames.append(sub)

    if not frames:
        raise ValueError("No UST data loaded. Check sheet 'USTs' in Data-Hardcodded.xlsx.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Date", "tenor"]).reset_index(drop=True)

    print(f"[surface_loader] UST curve: {len(df):,} rows | "
          f"{df['Date'].min().date()} → {df['Date'].max().date()} | "
          f"tenors: {df['tenor'].unique().tolist()}")
    return df


# ---------------------------------------------------------------------------
# RFR interpolation (vectorised)
# ---------------------------------------------------------------------------

def _build_rfr_pivot(ust_df: pd.DataFrame):
    """
    Pivot UST long-format → wide date-indexed DataFrame.
    Returns (pivot_df, tenor_years_array).
    """
    pivot = ust_df.pivot_table(
        index="Date", columns="tenor", values="rate", aggfunc="last"
    )
    ordered = [t for t in UST_TENORS if t in pivot.columns]
    pivot = pivot[ordered]
    t_years = np.array([UST_TENOR_YEARS[UST_TENORS.index(t)] for t in ordered])
    return pivot, t_years


def _interp_rfr_for_maturity(pivot: pd.DataFrame, t_years: np.ndarray,
                              mat_years: float) -> pd.Series:
    """
    For a given option maturity (scalar, years), interpolate the UST rate
    linearly across the yield curve, for every date in pivot.index.
    Boundary maturities are clamped (np.interp default).
    """
    rates = pivot.values  # shape (n_dates, n_tenors)
    result = np.array([np.interp(mat_years, t_years, row) for row in rates])
    return pd.Series(result, index=pivot.index, name="RiskFreeRate")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_options_df(
    coa_surface_file: str = COA_FILE,
    cla_surface_file: str = CLA_FILE,
    macro_file: str = MACRO_FILE,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Assemble the complete options DataFrame for calibrate.py.

    Steps
    -----
    1. Parse COA and CLA vol surfaces → long-format (Date, Underlying, Maturity,
       Moneyness, ImpliedVol)
    2. Parse futures prices (CO1 for COA, CL1 for CLA)
    3. Merge spot: Strike = Moneyness × FuturesPrice
    4. Parse UST yield curve
    5. Interpolate RiskFreeRate for each (Date, Maturity) combination
    6. Fill OptionType = "C", OptionPrice = NaN
    7. Apply optional date range filter
    8. Return DataFrame matching the 9-column schema expected by calibrate.py

    Returns
    -------
    DataFrame with columns:
        Date, Underlying, Strike, Maturity, OptionType,
        ImpliedVol, OptionPrice, SpotPrice, RiskFreeRate
    """
    print("\n[surface_loader] ── Loading Bloomberg data ──────────────────────")

    # ── 1. Vol surfaces ───────────────────────────────────────────────────
    coa_df = load_vol_surface(coa_surface_file, "CO1")
    cla_df = load_vol_surface(cla_surface_file, "CL1")
    surface_df = pd.concat([coa_df, cla_df], ignore_index=True)

    # ── 2. Futures prices ─────────────────────────────────────────────────
    futures_df = load_new_futures(macro_file)

    # ── 3. Merge spot price; compute strikes ──────────────────────────────
    fut_long = pd.concat([
        futures_df[["Date", "CO1"]].rename(columns={"CO1": "SpotPrice"}).assign(Underlying="CO1"),
        futures_df[["Date", "CL1"]].rename(columns={"CL1": "SpotPrice"}).assign(Underlying="CL1"),
    ], ignore_index=True).dropna(subset=["SpotPrice"])

    surface_df = surface_df.merge(fut_long, on=["Date", "Underlying"], how="left")
    surface_df["Strike"] = surface_df["Moneyness"] * surface_df["SpotPrice"]

    n_missing_spot = surface_df["SpotPrice"].isna().sum()
    if n_missing_spot:
        print(f"  Warning: {n_missing_spot:,} rows have no matching futures price "
              f"(date mismatch). These will be dropped.")
    surface_df = surface_df.dropna(subset=["SpotPrice", "Strike"])

    # ── 4. UST yield curve ────────────────────────────────────────────────
    ust_df = load_ust_curve(macro_file)
    pivot, t_years = _build_rfr_pivot(ust_df)

    # ── 5. Interpolate RFR for each option maturity ───────────────────────
    maturities = surface_df["Maturity"].unique()
    rfr_frames = []
    for mat in maturities:
        rfr_series = _interp_rfr_for_maturity(pivot, t_years, mat)
        rfr_frames.append(
            pd.DataFrame({
                "Date":         rfr_series.index,
                "Maturity":     mat,
                "RiskFreeRate": rfr_series.values,
            })
        )
    rfr_df = pd.concat(rfr_frames, ignore_index=True)
    surface_df = surface_df.merge(rfr_df, on=["Date", "Maturity"], how="left")

    n_missing_rfr = surface_df["RiskFreeRate"].isna().sum()
    if n_missing_rfr:
        # Forward-fill per (Underlying, Maturity) — handles weekends/holidays
        surface_df = surface_df.sort_values(["Underlying", "Maturity", "Date"])
        surface_df["RiskFreeRate"] = (
            surface_df.groupby(["Underlying", "Maturity"])["RiskFreeRate"]
                      .transform(lambda s: s.ffill().bfill())
        )
        remaining = surface_df["RiskFreeRate"].isna().sum()
        print(f"  [surface_loader] Filled {n_missing_rfr - remaining:,} NaN RFR via "
              f"forward-fill ({remaining} remain unfilled).")

    # ── 6. Static columns ─────────────────────────────────────────────────
    surface_df["OptionType"] = "C"
    surface_df["OptionPrice"] = np.nan

    # ── 7. Date filter ────────────────────────────────────────────────────
    if start_date is not None:
        surface_df = surface_df[surface_df["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        surface_df = surface_df[surface_df["Date"] <= pd.Timestamp(end_date)]

    # ── 8. Final output ───────────────────────────────────────────────────
    out_cols = [
        "Date", "Underlying", "Strike", "Maturity", "OptionType",
        "ImpliedVol", "OptionPrice", "SpotPrice", "RiskFreeRate",
    ]
    result = surface_df[out_cols].copy()
    result = result.sort_values(
        ["Date", "Underlying", "Maturity", "Strike"]
    ).reset_index(drop=True)

    print(f"\n[surface_loader] ── Build complete ──────────────────────────────")
    print(f"  Total rows      : {len(result):,}")
    print(f"  Date range      : {result['Date'].min().date()} → {result['Date'].max().date()}")
    print(f"  Underlyings     : {result['Underlying'].unique().tolist()}")
    print(f"  Maturities (yr) : {sorted(result['Maturity'].unique())}")
    print(f"  NaN SpotPrice   : {result['SpotPrice'].isna().sum()}")
    print(f"  NaN RiskFreeRate: {result['RiskFreeRate'].isna().sum()}")
    print(f"  NaN ImpliedVol  : {result['ImpliedVol'].isna().sum()}")

    return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_options_df()
    print("\nSample rows (first 14):")
    print(df.head(14).to_string())
    print("\nRows per (Underlying, Maturity):")
    print(df.groupby(["Underlying", "Maturity"]).size().to_string())
