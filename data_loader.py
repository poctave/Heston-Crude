import pandas as pd
import numpy as np

REQUIRED_OPTION_COLS = [
    "Date", "Underlying", "Strike", "Maturity",
    "OptionType", "ImpliedVol", "OptionPrice", "SpotPrice", "RiskFreeRate"
]


def load_futures(filepath="data/Heston-crude-HArdcodded.xlsx"):
    """Load CO1 (Brent) and CL1 (WTI) daily closing prices from the Futures sheet."""
    raw = pd.read_excel(filepath, sheet_name="Futures", header=None)
    df = raw.iloc[3:, [1, 2, 6]].copy()
    df.columns = ["Date", "CO1", "CL1"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["CO1"] = pd.to_numeric(df["CO1"], errors="coerce")
    df["CL1"] = pd.to_numeric(df["CL1"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"[data_loader] Futures loaded: {len(df)} rows | "
          f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def load_options(filepath="data/options_data.csv", underlying=None):
    """
    Load and validate options data from CSV.

    Expected columns:
        Date, Underlying, Strike, Maturity, OptionType,
        ImpliedVol, OptionPrice, SpotPrice, RiskFreeRate

    - Maturity must be in years (e.g. 1 month = 0.0833)
    - ImpliedVol as decimal (30% = 0.30); leave blank if only OptionPrice is available
    - Moneyness filter applied: keeps rows where 0.7 <= Strike/SpotPrice <= 1.3
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])

    # Validate columns
    missing = [c for c in REQUIRED_OPTION_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"options_data.csv is missing columns: {missing}\n"
            f"Required: {REQUIRED_OPTION_COLS}"
        )

    # Coerce numeric columns
    for col in ["Strike", "Maturity", "ImpliedVol", "OptionPrice", "SpotPrice", "RiskFreeRate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag rows that need IV inversion (have OptionPrice but no ImpliedVol)
    df["needs_iv_inversion"] = df["ImpliedVol"].isna() & df["OptionPrice"].notna()

    # Drop rows where both ImpliedVol and OptionPrice are missing
    df = df[df["ImpliedVol"].notna() | df["OptionPrice"].notna()].copy()

    # Moneyness filter: 0.7 <= K/S <= 1.3
    moneyness = df["Strike"] / df["SpotPrice"]
    df = df[(moneyness >= 0.7) & (moneyness <= 1.3)].copy()

    # Filter by underlying if requested
    if underlying is not None:
        df = df[df["Underlying"] == underlying].copy()

    df = df.sort_values(["Date", "Maturity", "Strike"]).reset_index(drop=True)
    print(f"[data_loader] Options loaded: {len(df)} rows"
          + (f" for {underlying}" if underlying else "")
          + f" | {df['needs_iv_inversion'].sum()} rows need IV inversion")
    return df


def get_calibration_dates(options_df):
    """Return sorted list of unique calibration dates in the options data."""
    return sorted(options_df["Date"].unique())


def merge_spot_into_options(options_df, futures_df):
    """
    If SpotPrice is missing in the options data, fill it from the futures DataFrame
    by matching on Date and Underlying (CO1 or CL1).
    """
    missing_spot = options_df["SpotPrice"].isna()
    if not missing_spot.any():
        return options_df

    futures_long = futures_df.melt(
        id_vars="Date", value_vars=["CO1", "CL1"],
        var_name="Underlying", value_name="FuturesSpot"
    )
    merged = options_df.merge(futures_long, on=["Date", "Underlying"], how="left")
    merged.loc[missing_spot, "SpotPrice"] = merged.loc[missing_spot, "FuturesSpot"]
    merged = merged.drop(columns=["FuturesSpot"])
    n_filled = missing_spot.sum() - merged["SpotPrice"].isna().sum()
    print(f"[data_loader] Filled {n_filled} missing SpotPrice values from futures data.")
    return merged


if __name__ == "__main__":
    futures = load_futures()
    print(futures.head())
