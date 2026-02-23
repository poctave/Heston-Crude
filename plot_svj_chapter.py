"""
plot_svj_chapter.py — Chapter 7 figures for the SVJ (Bates 1996) analysis.

Produces 5 figures (saved to data_plots/):

  A.  svj_smile_co1.png          — 2×3 smile grid: market / Heston / SVJ (CO1)
  B.  svj_smile_cl1.png          — 2×3 smile grid: market / Heston / SVJ (CL1)
  C.  svj_hessian_matrix.png     — 8×8 Hessian heatmap (CO1 SVJ)
  D.  svj_eigenvalue_spectrum.png — SVJ vs Heston eigenvalue comparison
  E.  svj_loss_lam_muj.png       — 3D loss landscape: λ vs μ_J

Run AFTER calibrate_svj.py has completed (reads results/CO1_svj_calibration.csv
and results/CL1_svj_calibration.csv).

Run with:
    MPLBACKEND=Agg PYTHONUNBUFFERED=1 conda run python plot_svj_chapter.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import surface_loader
import heston_model
import svj_model

warnings.filterwarnings("ignore")
os.makedirs("data_plots", exist_ok=True)

CAL_DATE    = pd.Timestamp("2021-12-01")
ROLLING_CSV = "data_plots/rolling_hessian.csv"
RESULTS_DIR = "results"

PARAM_NAMES_SVJ = [
    r"$\kappa$", r"$\theta$", r"$\xi$", r"$\rho$", r"$v_0$",
    r"$\lambda$", r"$\mu_J$", r"$\sigma_J$"
]
PARAM_KEYS_SVJ = ["kappa", "theta", "sigma", "rho", "v0", "lam", "mu_j", "sigma_j"]

STYLE = {
    "CO1": {"color": "seagreen",  "label": "Brent (CO1)"},
    "CL1": {"color": "steelblue", "label": "WTI (CL1)"},
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading Bloomberg surface data...")
df = surface_loader.build_options_df()
moneyness = df["Strike"] / df["SpotPrice"]
df = df[(moneyness >= 0.70) & (moneyness <= 1.30)].copy()

rolling = pd.read_csv(ROLLING_CSV, parse_dates=["Date"])

# Load SVJ calibration results
svj_params = {}
heston_params_dict = {}
for und in ["CO1", "CL1"]:
    csv_path = os.path.join(RESULTS_DIR, f"{und}_svj_calibration.csv")
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path} — run calibrate_svj.py first.")
        sys.exit(1)
    row = pd.read_csv(csv_path).iloc[0]
    svj_params[und] = np.array([
        row["kappa"], row["theta"], row["sigma"], row["rho"], row["v0"],
        row["lam"], row["mu_j"], row["sigma_j"]
    ])

    # Heston params from rolling CSV
    h = rolling[(rolling["Date"] == CAL_DATE) & (rolling["Underlying"] == und)].iloc[0]
    heston_params_dict[und] = {
        "params": np.array([h["kappa"], h["theta"], h["sigma"], h["rho"], h["v0"]]),
        "rmse":   h["rmse_volpts"],
    }

print(f"Loaded SVJ params for CO1 and CL1 (date: {CAL_DATE.date()})")


# ---------------------------------------------------------------------------
# Figure A/B: svj_smile_co1.png / svj_smile_cl1.png
# ---------------------------------------------------------------------------

def plot_svj_smile(underlying):
    """2×3 grid of smile cross-sections: market / Heston / SVJ."""
    und_label = STYLE[underlying]["label"]
    sub = df[(df["Underlying"] == underlying) & (df["Date"] == CAL_DATE)].copy()
    if sub.empty:
        print(f"  No data for {underlying} on {CAL_DATE.date()}")
        return

    S = sub["SpotPrice"].iloc[0]
    r = sub["RiskFreeRate"].iloc[0]
    maturities = sorted(sub["Maturity"].unique())[:6]

    svj_p = svj_params[underlying]
    heston_p = heston_params_dict[underlying]["params"]
    heston_rmse = heston_params_dict[underlying]["rmse"]

    svj_rmse = np.sqrt(svj_model.calibration_objective_svj(
        svj_p, S, sub["Strike"].values, sub["Maturity"].values,
        r, sub["ImpliedVol"].values
    )) * 100

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8), sharey=False)
    fig.suptitle(
        f"{und_label} — Implied Volatility Smile: Market / Heston / SVJ\n"
        f"1 December 2021   |   "
        f"Heston RMSE = {heston_rmse:.2f} vpts   "
        f"SVJ RMSE = {svj_rmse:.2f} vpts",
        fontsize=12, fontweight="bold"
    )

    for idx, T in enumerate(maturities):
        ax = axes[idx // ncols][idx % ncols]
        mask = sub["Maturity"] == T
        K_sub = sub.loc[mask, "Strike"].values
        IV_mkt = sub.loc[mask, "ImpliedVol"].values
        mon = K_sub / S

        IV_svj = svj_model.svj_implied_vol(svj_p, S, K_sub,
                                            np.full_like(K_sub, T), r)
        IV_heston = heston_model.heston_implied_vol(
            heston_p, S, K_sub, np.full_like(K_sub, T), r
        )

        ax.scatter(mon, IV_mkt * 100, color="steelblue", s=35, zorder=5, label="Market")
        ax.plot(mon, IV_heston * 100, color="firebrick", lw=1.4, ls="--",
                label="Heston", zorder=4)
        ax.plot(mon, IV_svj * 100, color="darkorange", lw=1.8, ls="-",
                label="SVJ (Bates)", zorder=6)

        months = round(T * 12)
        ax.set_title(f"T = {months}M", fontsize=10, fontweight="bold")
        ax.set_xlabel("Moneyness (K/S)", fontsize=8)
        ax.set_ylabel("Implied Vol (%)", fontsize=8)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused panels
    for idx in range(len(maturities), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    fname = f"data_plots/svj_smile_{underlying.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


print("\nFigures A/B: smile grids...")
plot_svj_smile("CO1")
plot_svj_smile("CL1")


# ---------------------------------------------------------------------------
# Figure C: svj_hessian_matrix.png
# ---------------------------------------------------------------------------

print("\nFigure C: SVJ Hessian matrix...")

und_focus = "CO1"
sub_h = df[(df["Underlying"] == und_focus) & (df["Date"] == CAL_DATE)].dropna(
    subset=["ImpliedVol"]
)
S_h   = sub_h["SpotPrice"].iloc[0]
r_h   = sub_h["RiskFreeRate"].iloc[0]
K_h   = sub_h["Strike"].values
T_h   = sub_h["Maturity"].values
IV_h  = sub_h["ImpliedVol"].values
svj_p_h = svj_params[und_focus]

print(f"  Computing 8×8 Hessian for {und_focus} SVJ (this may take ~2 min)...")
H_svj = svj_model.compute_svj_hessian(svj_p_h, S_h, K_h, T_h, r_h, IV_h)
spec_svj = svj_model.spectral_decomposition_svj(H_svj)
print(f"  SVJ κ_H = {spec_svj['condition_number']:.3e}")

fig, ax = plt.subplots(1, 1, figsize=(8, 7))
vmax = np.max(np.abs(H_svj))
im = ax.imshow(H_svj, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.75, label="Hessian entry value")
ax.set_xticks(range(8)); ax.set_xticklabels(PARAM_NAMES_SVJ, fontsize=9)
ax.set_yticks(range(8)); ax.set_yticklabels(PARAM_NAMES_SVJ, fontsize=9)
ax.set_title(
    f"SVJ Hessian of Calibration Loss — Brent (CO1)\n"
    f"1 December 2021  |  κ_H = {spec_svj['condition_number']:.2e}",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("data_plots/svj_hessian_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data_plots/svj_hessian_matrix.png")


# ---------------------------------------------------------------------------
# Figure D: svj_eigenvalue_spectrum.png
# ---------------------------------------------------------------------------

print("\nFigure D: SVJ vs Heston eigenvalue comparison...")

# Heston Hessian for CO1 (recompute with same data)
heston_p_h = heston_params_dict[und_focus]["params"]
import hessian as hess_mod
H_heston = hess_mod.compute_hessian(heston_p_h, S_h, K_h, T_h, r_h, IV_h)
spec_heston = hess_mod.spectral_decomposition(H_heston)

PARAM_NAMES_HESTON = [r"$\kappa$", r"$\theta$", r"$\xi$", r"$\rho$", r"$v_0$"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Eigenvalue Spectrum: Heston vs SVJ — Brent (CO1), 1 December 2021",
    fontsize=12, fontweight="bold"
)

# Heston panel
ax_h = axes[0]
eigs_h = spec_heston["eigenvalues"]
colors_h = ["firebrick" if e < 0 else "steelblue" for e in eigs_h]
ax_h.bar(range(1, 6), np.abs(eigs_h), color=colors_h, edgecolor="white", linewidth=0.5)
ax_h.set_yscale("log")
ax_h.set_xticks(range(1, 6)); ax_h.set_xticklabels(PARAM_NAMES_HESTON, fontsize=10)
ax_h.set_ylabel(r"$|\lambda_i|$ (log scale)", fontsize=10)
ax_h.set_title(
    f"Heston (5 params)\nκ_H = {spec_heston['condition_number']:.2e}",
    fontsize=10
)
ax_h.grid(True, axis="y", alpha=0.3)

# SVJ panel
ax_s = axes[1]
eigs_s = spec_svj["eigenvalues"]
colors_s = ["firebrick" if e < 0 else "darkorange" for e in eigs_s]
ax_s.bar(range(1, 9), np.abs(eigs_s), color=colors_s, edgecolor="white", linewidth=0.5)
ax_s.set_yscale("log")
ax_s.set_xticks(range(1, 9)); ax_s.set_xticklabels(PARAM_NAMES_SVJ, fontsize=9,
                                                     rotation=20)
ax_s.set_ylabel(r"$|\lambda_i|$ (log scale)", fontsize=10)
ax_s.set_title(
    f"SVJ / Bates (8 params)\nκ_H = {spec_svj['condition_number']:.2e}",
    fontsize=10
)
ax_s.grid(True, axis="y", alpha=0.3)

# Legend for negative eigenvalues
from matplotlib.patches import Patch
legend_els = [Patch(facecolor="firebrick", label="Negative eigenvalue"),
              Patch(facecolor="steelblue", label="Heston positive"),
              Patch(facecolor="darkorange", label="SVJ positive")]
fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("data_plots/svj_eigenvalue_spectrum.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data_plots/svj_eigenvalue_spectrum.png")


# ---------------------------------------------------------------------------
# Figure E: svj_loss_lam_muj.png  (3D loss landscape: λ vs μ_J)
# ---------------------------------------------------------------------------

print("\nFigure E: 3D loss landscape λ vs μ_J ...")

GRID_SIZE = 35
lam_vals  = np.linspace(0.05, 6.0, GRID_SIZE)
muj_vals  = np.linspace(-1.0, 0.5, GRID_SIZE)

opt_full = svj_p_h.copy()
lam_idx  = PARAM_KEYS_SVJ.index("lam")
muj_idx  = PARAM_KEYS_SVJ.index("mu_j")

grid_lm = np.full((GRID_SIZE, GRID_SIZE), np.nan)

for i, lam_v in enumerate(lam_vals):
    for j, muj_v in enumerate(muj_vals):
        p = opt_full.copy()
        p[lam_idx] = lam_v
        p[muj_idx] = muj_v
        try:
            loss = svj_model.calibration_objective_svj(
                p, S_h, K_h, T_h, r_h, IV_h
            )
            if np.isfinite(loss) and loss < 1e5:
                grid_lm[i, j] = np.sqrt(loss) * 100
        except Exception:
            pass

LAM_GRID, MUJ_GRID = np.meshgrid(lam_vals, muj_vals, indexing="ij")
log_grid = np.log10(np.where(grid_lm > 0, grid_lm, np.nan))

fig_lm = plt.figure(figsize=(10, 6))
ax_lm  = fig_lm.add_subplot(111, projection="3d")
surf = ax_lm.plot_surface(
    LAM_GRID, MUJ_GRID, log_grid,
    cmap="plasma", alpha=0.90, edgecolor="none",
    vmin=np.nanpercentile(log_grid, 5),
    vmax=np.nanpercentile(log_grid, 98),
)
fig_lm.colorbar(surf, ax=ax_lm, shrink=0.42, pad=0.10,
                label=r"$\log_{10}$(RMSE, vol pts)")

# Mark optimal point
opt_loss = svj_model.calibration_objective_svj(
    opt_full, S_h, K_h, T_h, r_h, IV_h
)
ax_lm.scatter([opt_full[lam_idx]], [opt_full[muj_idx]],
              [np.log10(np.sqrt(opt_loss) * 100)],
              color="red", s=60, zorder=10, label=r"Optimal $\hat{\Theta}$")

ax_lm.set_xlabel(r"$\lambda$ (jump intensity)", labelpad=8)
ax_lm.set_ylabel(r"$\mu_J$ (mean log-jump)", labelpad=8)
ax_lm.set_zlabel(r"$\log_{10}$(RMSE, vol pts)", labelpad=8)
ax_lm.set_title(
    r"SVJ Loss Landscape — $\lambda$ vs $\mu_J$" + "\nBrent (CO1), 1 December 2021",
    fontsize=11, fontweight="bold", pad=12
)
ax_lm.view_init(elev=28, azim=225)
ax_lm.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("data_plots/svj_loss_lam_muj.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data_plots/svj_loss_lam_muj.png")

print("\nAll 5 SVJ chapter figures saved to data_plots/.")
