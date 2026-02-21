import pandas as pd
import matplotlib.pyplot as plt

FILE = "data/Heston-crude-HArdcodded.xlsx"

# --- Load Futures sheet ---
# Col 1 = Date, Col 2 = CO1 price, Col 6 = CL1 price
# Skip rows 0-2 (empty row + header + bloomberg formula row)
futures_raw = pd.read_excel(FILE, sheet_name="Futures", header=None)
df = futures_raw.iloc[3:, [1, 2, 6]].copy()
df.columns = ["Date", "CO1", "CL1"]
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["CO1"] = pd.to_numeric(df["CO1"], errors="coerce")
df["CL1"] = pd.to_numeric(df["CL1"], errors="coerce")
df = df.dropna().sort_values("Date").reset_index(drop=True)

print(f"Futures: {len(df)} rows | {df['Date'].min().date()} to {df['Date'].max().date()}")
print(df.head())

# --- Load Sheet3 for VIX and OVX ---
# Skip row 0 (empty) and row 1 (header), dates are Excel serial numbers
s3_raw = pd.read_excel(FILE, sheet_name="Sheet3", header=None)
s3 = s3_raw.iloc[2:, [1, 4, 5]].copy()
s3.columns = ["Date", "VIX", "OVX"]
s3["Date"] = pd.to_numeric(s3["Date"], errors="coerce")
s3 = s3.dropna(subset=["Date"])
s3["Date"] = pd.to_datetime(s3["Date"].astype(int), unit="D", origin="1899-12-30")
s3["VIX"] = pd.to_numeric(s3["VIX"], errors="coerce")
s3["OVX"] = pd.to_numeric(s3["OVX"], errors="coerce")
s3 = s3.sort_values("Date").reset_index(drop=True)

print(f"\nSheet3: {len(s3)} rows | {s3['Date'].min().date()} to {s3['Date'].max().date()}")
print(s3.head())

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Crude Oil Market Data", fontsize=14, fontweight="bold")

# Panel 1: Crude oil prices
axes[0].plot(df["Date"], df["CO1"], label="CO1 (Brent)", color="steelblue", linewidth=0.8)
axes[0].plot(df["Date"], df["CL1"], label="CL1 (WTI)", color="darkorange", linewidth=0.8)
axes[0].set_ylabel("Price (USD)")
axes[0].set_title("Futures Prices")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel 2: VIX
axes[1].plot(s3["Date"], s3["VIX"], label="VIX", color="crimson", linewidth=0.8)
axes[1].set_ylabel("Index")
axes[1].set_title("VIX (Equity Volatility)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Panel 3: OVX
axes[2].plot(s3["Date"], s3["OVX"], label="OVX", color="seagreen", linewidth=0.8)
axes[2].set_ylabel("Index")
axes[2].set_title("OVX (Oil Volatility)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("price_evolution.png", dpi=150)
plt.show()
print("\nPlot saved to price_evolution.png")
