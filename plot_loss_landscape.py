"""
plot_loss_landscape.py — 3D loss landscape slices of the Heston calibration surface.

For each of four parameter pairs, all other parameters are fixed at their optimal
values (CO1, 2026-02-19) and the weighted-MSE loss is evaluated on a 2D grid.
Z-axis: log10(RMSE in vol points) for readability.

Outputs (to data_plots/):
  loss_kappa_sigma.png   — (κ, σ): mean-reversion vs vol-of-vol
  loss_rho_sigma.png     — (ρ, σ): correlation vs vol-of-vol
  loss_kappa_theta.png   — (κ, θ): mean-reversion vs long-run variance
  loss_rho_v0.png        — (ρ, v0): correlation vs initial variance

Run with:
  MPLBACKEND=Agg PYTHONUNBUFFERED=1 conda run python plot_loss_landscape.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import warnings

import surface_loader
import heston_model

os.makedirs("data_plots", exist_ok=True)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Calibration data for CO1 on 2026-02-19
# ---------------------------------------------------------------------------
CAL_DATE    = pd.Timestamp("2026-02-19")
UNDERLYING  = "CO1"

print("Loading surface data...")
df = surface_loader.build_options_df()
sub = df[
    (df["Underlying"] == UNDERLYING) &
    (df["Date"] == CAL_DATE)
].copy()

S = sub["SpotPrice"].iloc[0]
K = sub["Strike"].values
T = sub["Maturity"].values
r = sub["RiskFreeRate"].values.mean()
mkt_ivols = sub["ImpliedVol"].values

# Optimal parameters (CO1, 2026-02-19)
OPT = {
    "kappa": 4.7748788165989,
    "theta": 0.036272111964295056,
    "sigma": 1.588887335782593,
    "rho":   0.7699427543219018,
    "v0":    0.21433645304159257,
}
PARAM_ORDER = ["kappa", "theta", "sigma", "rho", "v0"]

print(f"Loaded {len(sub)} options for {UNDERLYING} on {CAL_DATE.date()}")
print(f"S={S:.2f}, r={r:.4f}")


# ---------------------------------------------------------------------------
# Helper: evaluate loss on a 2D grid
# ---------------------------------------------------------------------------

def eval_grid(ax1_name, ax2_name, ax1_vals, ax2_vals):
    """
    Evaluate RMSE (vol points) on a grid of (ax1, ax2), fixing all
    other parameters at their optimal values.

    Returns (GRID of shape [len(ax1), len(ax2)]) where rows=ax1, cols=ax2.
    NaN where the model fails.
    """
    base = [OPT[p] for p in PARAM_ORDER]
    idx1 = PARAM_ORDER.index(ax1_name)
    idx2 = PARAM_ORDER.index(ax2_name)

    n1, n2 = len(ax1_vals), len(ax2_vals)
    grid = np.full((n1, n2), np.nan)

    for i, v1 in enumerate(ax1_vals):
        for j, v2 in enumerate(ax2_vals):
            p = base.copy()
            p[idx1] = v1
            p[idx2] = v2
            try:
                loss = heston_model.calibration_objective(
                    p, S, K, T, r, mkt_ivols
                )
                if np.isfinite(loss) and loss < 1e5:
                    grid[i, j] = np.sqrt(loss) * 100   # RMSE in vol points
            except Exception:
                pass

    return grid


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

LABEL = {
    "kappa": r"$\kappa$ (mean-reversion speed)",
    "theta": r"$\theta$ (long-run variance)",
    "sigma": r"$\xi$ (vol-of-vol)",
    "rho":   r"$\rho$ (correlation)",
    "v0":    r"$v_0$ (initial variance)",
}

TICK_LABEL = {
    "kappa": r"$\kappa$",
    "theta": r"$\theta$",
    "sigma": r"$\xi$",
    "rho":   r"$\rho$",
    "v0":    r"$v_0$",
}


def plot_landscape(ax1_name, ax2_name, grid,
                   ax1_vals, ax2_vals, fname, elev=28, azim=225):
    """Plot the 3D log10(RMSE) landscape for one parameter pair."""
    A1, A2 = np.meshgrid(ax1_vals, ax2_vals, indexing="ij")  # [n1, n2]

    log_grid = np.log10(np.where(grid > 0, grid, np.nan))

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        A1, A2, log_grid,
        cmap="plasma", alpha=0.90, edgecolor="none",
        vmin=np.nanpercentile(log_grid, 5),
        vmax=np.nanpercentile(log_grid, 98),
    )
    fig.colorbar(surf, ax=ax, shrink=0.42, pad=0.10,
                 label=r"$\log_{10}$(RMSE, vol pts)")

    # Mark the optimal point
    opt1 = OPT[ax1_name]
    opt2 = OPT[ax2_name]
    opt_loss = heston_model.calibration_objective(
        [OPT[p] for p in PARAM_ORDER], S, K, T, r, mkt_ivols
    )
    opt_z = np.log10(np.sqrt(opt_loss) * 100)
    ax.scatter([opt1], [opt2], [opt_z],
               color="red", s=60, zorder=10, label="Optimal $\\hat{\\Theta}$")

    ax.set_xlabel(LABEL[ax1_name], labelpad=8)
    ax.set_ylabel(LABEL[ax2_name], labelpad=8)
    ax.set_zlabel(r"$\log_{10}$(RMSE, vol pts)", labelpad=8)
    ax.set_title(
        f"Heston Loss Landscape — {TICK_LABEL[ax1_name]} vs {TICK_LABEL[ax2_name]}\n"
        f"Brent (CO1), {CAL_DATE.date()}",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Define the four slices and their grids (40×40 each)
# ---------------------------------------------------------------------------

GRID_SIZE = 40

SLICES = [
    {
        "ax1": "kappa", "ax2": "sigma",
        "ax1_range": (0.3, 14.0),
        "ax2_range": (0.3,  2.0),
        "fname": "data_plots/loss_kappa_sigma.png",
        "elev": 28, "azim": 220,
    },
    {
        "ax1": "rho",   "ax2": "sigma",
        "ax1_range": (0.1,  0.98),
        "ax2_range": (0.3,  2.0),
        "fname": "data_plots/loss_rho_sigma.png",
        "elev": 28, "azim": 210,
    },
    {
        "ax1": "kappa", "ax2": "theta",
        "ax1_range": (0.3, 14.0),
        "ax2_range": (0.005, 0.15),
        "fname": "data_plots/loss_kappa_theta.png",
        "elev": 28, "azim": 230,
    },
    {
        "ax1": "rho",   "ax2": "v0",
        "ax1_range": (0.1,  0.98),
        "ax2_range": (0.05, 0.50),
        "fname": "data_plots/loss_rho_v0.png",
        "elev": 28, "azim": 215,
    },
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

for sl in SLICES:
    ax1, ax2 = sl["ax1"], sl["ax2"]
    print(f"\nComputing grid: {ax1} × {ax2}  ({GRID_SIZE}×{GRID_SIZE})...")

    ax1_vals = np.linspace(*sl["ax1_range"], GRID_SIZE)
    ax2_vals = np.linspace(*sl["ax2_range"], GRID_SIZE)

    grid = eval_grid(ax1, ax2, ax1_vals, ax2_vals)

    n_valid = np.sum(np.isfinite(grid))
    print(f"  Valid cells: {n_valid}/{GRID_SIZE**2}  "
          f"  RMSE range: [{np.nanmin(grid):.2f}, {np.nanmax(grid):.2f}] vol pts")

    plot_landscape(
        ax1, ax2, grid, ax1_vals, ax2_vals,
        fname=sl["fname"],
        elev=sl["elev"], azim=sl["azim"],
    )

print("\nAll 4 loss landscape figures saved to data_plots/.")
