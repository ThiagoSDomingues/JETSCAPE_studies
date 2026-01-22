# =============================================================================
# EMULATOR PRIOR PREDICTIONS — 90% CREDIBLE BANDS
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CENTRALITY_INDEX = 7 
N_PRIOR_SAMPLES  = 2000
N_PC_USE         = 5
CONF_LEVEL       = 0.90

PCA_RESULTS_FILE = "pca_results/pca_results_all_corrections.pkl"
EMULATOR_FOLDER  = "emulators_v0pt"

VISCOUS_CORRECTIONS = ["Grad", "CE", "PTM", "PTB"]
color_names = {"Grad": 'blue', "CE": 'red', "PTM": 'magenta', "PTB": 'green'}
exp_centrality_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", 
                         "40-50%", "50-60%", "60-70%", "70-80%", "80-90%"]  

RANGE_FILE = (
    "design_pts_Pb_Pb_2760_production/"
    "design_ranges_main_PbPb-2760.dat"
)

# =============================================================================
# LOAD DESIGN RANGES
# =============================================================================

design_range = pd.read_csv(RANGE_FILE, index_col=0)
param_min = design_range["min"].values
param_max = design_range["max"].values
n_params  = len(param_min)

# =============================================================================
# SAMPLE PRIORS
# =============================================================================

np.random.seed(1)
n_pr_samples = 2000

# Load the original posterior samples; try first without index
orig_data_df = pd.read_csv("new_LHC_posterior_samples.csv")
orig_df = orig_data_df.iloc[:, :-1]
orig_samples = orig_df.values

# =============================================================================
# LOAD PCA RESULTS
# =============================================================================

with open(PCA_RESULTS_FILE, "rb") as f:
    pca_results = pickle.load(f)

# =============================================================================
# PCA RECONSTRUCTION
# =============================================================================

def reconstruct_from_pc_scores(pc_scores, inverse_tf_matrix, scaler):
    Y_scaled = pc_scores @ inverse_tf_matrix
    return Y_scaled * scaler.scale_ + scaler.mean_

# =============================================================================
# EMULATOR'S PREDICTION
# =============================================================================

def predict_observables(model_parameters, Emulators, inverse_tf_matrix, SS):
    model_parameters = np.array(model_parameters).flatten()
    if model_parameters.shape[0] != 17:
        raise ValueError("Input model parameters must be a 17-dimensional array.")
    theta = model_parameters.reshape(1, -1)
    n_pc = len(Emulators)
    pc_means = []
    pc_vars = []
    for emulator in Emulators:
        mn, std = emulator.predict(theta, return_std=True)
        pc_means.append(mn.flatten()[0])
        pc_vars.append(std.flatten()[0]**2)
    pc_means = np.array(pc_means).reshape(1, -1)
    variance_matrix = np.diag(np.array(pc_vars))
    inverse_transformed_mean = pc_means @ inverse_tf_matrix[:n_pc, :] + SS.mean_.reshape(1, -1)
    A = inverse_tf_matrix[:n_pc, :]
    inverse_transformed_variance = np.einsum('ik,kl,lj->ij', A.T, variance_matrix, A)
    return inverse_transformed_mean.flatten(), inverse_transformed_variance

# =============================================================================
# FIGURE
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                         sharex=True, sharey=True)
axes = axes.flatten()

# =============================================================================
# LOOP OVER VISCOUS CORRECTIONS
# =============================================================================

for ax, corr in zip(axes, VISCOUS_CORRECTIONS):

    print(f"\n{'='*60}")
    print(f"Processing {corr}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # Load emulators
    # -------------------------------------------------------------------------
    emu_file = Path(EMULATOR_FOLDER) / f"gp_emulators_{corr}.pkl"
    if not emu_file.exists():
        ax.text(0.5, 0.5, f"No emulator for {corr}",
                transform=ax.transAxes,
                ha="center", va="center")
        continue

    with open(emu_file, "rb") as f:
        emulators = pickle.load(f)

    emulators = emulators[:N_PC_USE]
    
    # -------------------------------------------------------------------------
    # Predict PCs WITH uncertainties
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # PCA reconstruction
    # -------------------------------------------------------------------------
    info = pca_results[corr]
    inv_tf = info["inverse_tf_matrix"][:N_PC_USE, :]
    scaler = info["scaler"]
    labels = info["observable_labels"]

    #v0pt_all = reconstruct_from_pc_scores(pc_mean, inv_tf, scaler)
    
    pr_predictions = []
    orig_predictions = []
    post_predictions = []
    
    # Load the posterior samples
    posterior_file = f"posterior/mcmc_chain_{corr}.csv"
    samples_df = pd.read_csv(posterior_file)
    
    # Extracting the posterior samples values
    posterior_samples = samples_df.values  # shape (n_samples, 17)
    
    # Looping over all prior samples: supposing a uniform prior
    for params in np.random.uniform(param_min, param_max, (n_pr_samples, n_params)):        
        y_pred, cov_pred = predict_observables(params, emulators, inv_tf, scaler)
        pr_predictions.append(y_pred.flatten())
    pr_predictions = np.array(pr_predictions) # shape (n_pr_samples, n_obs)
    
    if corr == 'Grad':
        # Use the emulator surrogate to predict observables for each posterior sample
        for theta in orig_samples:
            # theta is expected to be a 17-D vector.
            if theta.shape[0] != 17:
                raise ValueError("Posterior sample does not have 17 elements after adjustment.")
            y_orig_pred, _ = predict_observables(theta, emulators, inv_tf, scaler)
            orig_predictions.append(y_orig_pred)
        orig_predictions = np.array(orig_predictions)  # shape (n_samples, n_obs)    
    
    # Use the emulator surrogate to predict observables for each posterior sample
    for theta in posterior_samples:
        # theta is expected to be a 17-D vector.
        if theta.shape[0] != 17:
            raise ValueError("Posterior sample does not have 17 elements after adjustment.")
        y_post_pred, _ = predict_observables(theta, emulators, inv_tf, scaler)
        post_predictions.append(y_post_pred)
    post_predictions = np.array(post_predictions)  # shape (n_samples, n_obs)
    
    # Reshape predictions into (n_pr_samples, n_cent_bins, n_pt_bins)
    pr_n_cent = 10
    pr_n_pt = 29    
    pr_predictions_reshaped = pr_predictions.reshape(n_pr_samples, pr_n_cent, pr_n_pt)
    
    if corr == 'Grad':
        # Reshape predictions into (n_samples, 10, 29)
        n_samples = orig_predictions.shape[0]
        n_cent = 10  
        n_pt = 29 
        orig_predictions_reshaped = orig_predictions.reshape(n_samples, n_cent, n_pt)
    
    # Reshape predictions into (n_samples, 10, 29)
    post_n_samples = post_predictions.shape[0]
    post_predictions_reshaped = post_predictions.reshape(post_n_samples, n_cent, n_pt)

    # -------------------------------------------------------------------------
    # Select centrality
    # -------------------------------------------------------------------------
    cent_idx = np.array([c for c, _ in labels])
    pt_vals  = np.array([p for _, p in labels])

    mask = cent_idx == CENTRALITY_INDEX
    if not np.any(mask):
        continue

    pt = pt_vals[mask]
    
    # Compute credible intervals: 5th, 50th, 95th percentiles
    pr_env = np.percentile(pr_predictions_reshaped[:, CENTRALITY_INDEX, :], [5, 50, 95], axis=0)
    if corr == 'Grad':
        orig_env = np.percentile(orig_predictions_reshaped[:, CENTRALITY_INDEX, :], [5, 50, 95], axis=0)
    post_env = np.percentile(post_predictions_reshaped[:, CENTRALITY_INDEX, :], [5, 50, 95], axis=0)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    
    ax.fill_between(pt, pr_env[0, :], pr_env[2, :],
                    hatch='///', edgecolor='black', facecolor='gray', alpha=0.4, lw=2, label="Prior 90% C.I.")
    if corr == 'Grad':
        ax.fill_between(pt, orig_env[0, :], orig_env[2, :],
                        color='orange', alpha=0.4, label="Original Posterior 90% C.I.")
    ax.fill_between(pt, post_env[0, :], post_env[2, :],
                    color=color_names[corr], alpha=0.4, label="Grad Posterior 90% C.I.")

    ax.set_xscale("log")
    ax.set_title(corr, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    
# =============================================================================
# FINAL COSMETICS
# =============================================================================

for ax in axes:
    ax.set_xlim(0.5, 3.927575)
    ax.set_ylim(-0.05, 0.35)
    ax.set_xlabel(r"$p_T$ [GeV/$c$]", fontsize=12)
    ax.set_ylabel(r"$v_0(p_T)$", fontsize=12)

axes[0].legend(framealpha=0.95, fontsize=10)

fig.suptitle(
    f"Emulator's Predictions for {exp_centrality_labels[CENTRALITY_INDEX]} centrality \n"
    "Pb–Pb 2.76 TeV",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'pred_v0pt_{exp_centrality_labels[CENTRALITY_INDEX]}.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ PRIOR + POSTERIORS")
print("✓ Saved figure as 'prior_v0pt_corrected.pdf'")
