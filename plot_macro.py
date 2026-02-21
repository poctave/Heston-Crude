import pandas as pd
import matplotlib.pyplot as plt

FILE = "data/Heston-crude-HArdcodded.xlsx"

# --- Load Sheet3 ---
s3_raw = pd.read_excel(FILE, sheet_name="Sheet3", header=None)
s3 = s3_raw.iloc[2:, [1, 5, 7, 8, 9]].copy()
s3.columns = ["Date", "OVX", "Geo_Risk", "Inventory_Changes", "Inventory"]

# Convert Excel serial dates
s3["Date"] = pd.to_numeric(s3["Date"], errors="coerce")
s3 = s3.dropna(subset=["Date"])
s3["Date"] = pd.to_datetime(s3["Date"].astype(int), unit="D", origin="1899-12-30")

# Convert all numeric columns
for col in ["OVX", "Geo_Risk", "Inventory_Changes", "Inventory"]:
    s3[col] = pd.to_numeric(s3[col], errors="coerce")

s3 = s3.sort_values("Date").reset_index(drop=True)

# --- Plot ---
fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
fig.suptitle("Macro & Oil Market Indicators", fontsize=14, fontweight="bold")

# Panel 1: OVX
axes[0].plot(s3["Date"], s3["OVX"], color="seagreen", linewidth=0.8)
axes[0].set_ylabel("Index")
axes[0].set_title("OVX — Oil Volatility Index")
axes[0].grid(True, alpha=0.3)

# Panel 2: Geopolitical Risk
axes[1].plot(s3["Date"], s3["Geo_Risk"], color="firebrick", linewidth=0.8)
axes[1].set_ylabel("Index")
axes[1].set_title("Geopolitical Risk (Caldara & Iacoviello)")
axes[1].grid(True, alpha=0.3)

# Panel 3: Inventory Changes
axes[2].bar(s3["Date"], s3["Inventory_Changes"], color="steelblue", width=1.5, label="Weekly change")
axes[2].axhline(0, color="black", linewidth=0.6, linestyle="--")
axes[2].set_ylabel("Barrels")
axes[2].set_title("Oil Inventory Changes")
axes[2].grid(True, alpha=0.3)

# Panel 4: Total Inventory
axes[3].plot(s3["Date"], s3["Inventory"], color="goldenrod", linewidth=0.8)
axes[3].set_ylabel("Barrels")
axes[3].set_title("Total Oil Inventory")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("macro_indicators.png", dpi=150)
plt.show()
print("Plot saved to macro_indicators.png")
