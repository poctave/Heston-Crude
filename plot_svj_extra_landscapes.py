"""
plot_svj_extra_landscapes.py — Additional SVJ loss-landscape cross-sections.

Four parameter-pair slices of the SVJ calibration loss L(Θ), each fixing all
other parameters at the CO1 calibrated optimum (2021-12-01).

Chosen pairs (all economically interesting):
  1. rho  vs sigma   — the classic Heston ridge: does SVJ change it?
  2. kappa vs lam    — does jump intensity absorb κ's sloppy direction?
  3. rho  vs lam     — both control skew; substitution ridge expected
  4. mu_j vs sigma   — jump mean vs diffusion vol-of-vol: cross-model skew trade-off

Outputs (saved to data_plots/ and copied to Latex_Paper/):
  svj_loss_rho_sigma.png
  svj_loss_kappa_lam.png
  svj_loss_rho_lam.png
  svj_loss_muj_sigma.png

Run with:
    MPLBACKEND=Agg PYTHONUNBUFFERED=1 conda run --no-capture-output python plot_svj_extra_landscapes.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import surface_loader
import svj_model

warnings.filterwarnings("ignore")
os.makedirs("data_plots", exist_ok=True)

CAL_DATE    = pd.Timestamp("2021-12-01")
RESULTS_DIR = "results"
LATEX_DIR   = "Dissertation_LaTeX/Latex_Paper"

PARAM_KEYS_SVJ = ["kappa", "theta", "sigma", "rho", "v0", "lam", "mu_j", "sigma_j"]
PARAM_NAMES_SVJ = {
    "kappa":   r"$\kappa$",
    "theta":   r"$\theta$",
    "sigma":   r"$\xi$",
    "rho":     r"$\rho$",
    "v0":      r"$v_0$",
    "lam":     r"$\lambda$",
    "mu_j":    r"$\mu_J$",
    "sigma_j": r"$\sigma_J$",
}

GRID_SIZE = 32        # 32×32 = 1 024 evaluations per plot — fast yet smooth

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading Bloomberg surface data...")
df = surface_loader.build_options_df()
moneyness = df["Strike"] / df["SpotPrice"]
df = df[(moneyness >= 0.70) & (moneyness <= 1.30)].copy()

# CO1 slice on calibration date
sub = df[(df["Underlying"] == "CO1") & (df["Date"] == CAL_DATE)].dropna(
    subset=["ImpliedVol"]
)
if sub.empty:
    print("No CO1 data on calibration date — aborting.")
    sys.exit(1)

S_h  = sub["SpotPrice"].iloc[0]
r_h  = sub["RiskFreeRate"].iloc[0]
K_h  = sub["Strike"].values
T_h  = sub["Maturity"].values
IV_h = sub["ImpliedVol"].values

# Load calibrated SVJ optimum for CO1
csv_path = os.path.join(RESULTS_DIR, "CO1_svj_calibration.csv")
if not os.path.exists(csv_path):
    print(f"Missing {csv_path} — run calibrate_svj.py first.")
    sys.exit(1)
row = pd.read_csv(csv_path).iloc[0]
opt_full = np.array([
    row["kappa"], row["theta"], row["sigma"], row["rho"], row["v0"],
    row["lam"], row["mu_j"], row["sigma_j"]
])
print(f"Loaded CO1 SVJ optimum: {dict(zip(PARAM_KEYS_SVJ, opt_full))}")

# Baseline loss at optimum
opt_loss = svj_model.calibration_objective_svj(opt_full, S_h, K_h, T_h, r_h, IV_h)
opt_rmse_log = np.log10(np.sqrt(opt_loss) * 100)


# ---------------------------------------------------------------------------
# Helper: compute 2-D grid and draw 3D surface
# ---------------------------------------------------------------------------

def make_landscape(ax1_key, ax2_key, ax1_range, ax2_range, title,
                   elev=28, azim=225):
    """
    Compute and plot a 3D loss landscape for two chosen parameters.

    Returns the figure.
    """
    ax1_idx = PARAM_KEYS_SVJ.index(ax1_key)
    ax2_idx = PARAM_KEYS_SVJ.index(ax2_key)

    ax1_vals = np.linspace(*ax1_range, GRID_SIZE)
    ax2_vals = np.linspace(*ax2_range, GRID_SIZE)
    AX1, AX2 = np.meshgrid(ax1_vals, ax2_vals, indexing="ij")

    grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)

    for i, v1 in enumerate(ax1_vals):
        for j, v2 in enumerate(ax2_vals):
            p = opt_full.copy()
            p[ax1_idx] = v1
            p[ax2_idx] = v2
            try:
                loss = svj_model.calibration_objective_svj(
                    p, S_h, K_h, T_h, r_h, IV_h
                )
                if np.isfinite(loss) and loss < 1e5:
                    grid[i, j] = np.sqrt(loss) * 100
            except Exception:
                pass

    log_grid = np.log10(np.where(grid > 0, grid, np.nan))

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        AX1, AX2, log_grid,
        cmap="plasma", alpha=0.90, edgecolor="none",
        vmin=np.nanpercentile(log_grid, 5),
        vmax=np.nanpercentile(log_grid, 98),
    )
    fig.colorbar(surf, ax=ax, shrink=0.42, pad=0.10,
                 label=r"$\log_{10}$(RMSE, vol pts)")

    # Mark optimal point
    opt_v1 = opt_full[ax1_idx]
    opt_v2 = opt_full[ax2_idx]
    # Check optimal is inside the grid before plotting
    if ax1_range[0] <= opt_v1 <= ax1_range[1] and ax2_range[0] <= opt_v2 <= ax2_range[1]:
        ax.scatter([opt_v1], [opt_v2], [opt_rmse_log],
                   color="red", s=80, zorder=10, label=r"Optimal $\hat{\Theta}$")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel(PARAM_NAMES_SVJ[ax1_key], labelpad=8, fontsize=11)
    ax.set_ylabel(PARAM_NAMES_SVJ[ax2_key], labelpad=8, fontsize=11)
    ax.set_zlabel(r"$\log_{10}$(RMSE, vol pts)", labelpad=8, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 1. rho vs sigma  — does SVJ change the Heston ρ–ξ ridge?
# ---------------------------------------------------------------------------

print("\n[1/4] rho vs sigma (xi)...")
fig1 = make_landscape(
    "rho", "sigma",
    ax1_range=(-0.99, -0.05),
    ax2_range=(0.30, 2.00),
    title=r"SVJ Loss: $\rho$ vs $\xi$ — Brent (CO1), 1 Dec 2021"
    "\n(all other params at calibrated optimum)",
    elev=28, azim=220,
)
fname1 = "data_plots/svj_loss_rho_sigma.png"
fig1.savefig(fname1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved {fname1}")

# ---------------------------------------------------------------------------
# 2. kappa vs lam  — does jump intensity absorb the κ-sloppy direction?
# ---------------------------------------------------------------------------

print("[2/4] kappa vs lam...")
fig2 = make_landscape(
    "kappa", "lam",
    ax1_range=(0.01, 6.0),
    ax2_range=(0.01, 3.0),
    title=r"SVJ Loss: $\kappa$ vs $\lambda$ — Brent (CO1), 1 Dec 2021"
    "\n(does $\lambda$ absorb $\kappa$'s sloppy direction?)",
    elev=28, azim=225,
)
fname2 = "data_plots/svj_loss_kappa_lam.png"
fig2.savefig(fname2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved {fname2}")

# ---------------------------------------------------------------------------
# 3. rho vs lam  — skew substitution: continuous-path vs jump skew
# ---------------------------------------------------------------------------

print("[3/4] rho vs lam...")
fig3 = make_landscape(
    "rho", "lam",
    ax1_range=(-0.99, -0.05),
    ax2_range=(0.01, 3.0),
    title=r"SVJ Loss: $\rho$ vs $\lambda$ — Brent (CO1), 1 Dec 2021"
    "\n(diffusion skew vs jump-intensity skew trade-off)",
    elev=28, azim=225,
)
fname3 = "data_plots/svj_loss_rho_lam.png"
fig3.savefig(fname3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved {fname3}")

# ---------------------------------------------------------------------------
# 4. mu_j vs sigma  — jump mean vs diffusion vol-of-vol (cross-model skew)
# ---------------------------------------------------------------------------

print("[4/4] mu_j vs sigma (xi)...")
fig4 = make_landscape(
    "mu_j", "sigma",
    ax1_range=(-1.20, 0.30),
    ax2_range=(0.30, 2.00),
    title=r"SVJ Loss: $\mu_J$ vs $\xi$ — Brent (CO1), 1 Dec 2021"
    "\n(jump mean-size vs diffusion vol-of-vol cross-dependency)",
    elev=28, azim=220,
)
fname4 = "data_plots/svj_loss_muj_sigma.png"
fig4.savefig(fname4, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"  Saved {fname4}")

# ---------------------------------------------------------------------------
# Copy to LaTeX directory
# ---------------------------------------------------------------------------

print("\nCopying to LaTeX_Paper/...")
import shutil
for fname in [fname1, fname2, fname3, fname4]:
    dst = os.path.join(LATEX_DIR, os.path.basename(fname))
    shutil.copy(fname, dst)
    print(f"  {fname} → {dst}")

print("\nAll 4 extra SVJ landscapes done.")
