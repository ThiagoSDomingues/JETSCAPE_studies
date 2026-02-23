# =============================================================================
# EMULATOR PRIOR PREDICTIONS FOR GLOBAL v0 — 90% CREDIBLE BANDS
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

N_PRIOR_SAMPLES  = 2000
N_PC_USE         = None  # None = use all available PCs (global v0 has fewer)
CONF_LEVEL       = 0.90

PCA_RESULTS_FILE = "pca_results/pca_results_global_v0_all_corrections.pkl"
EMULATOR_FOLDER  = "emulators_global_v0"
OUTPUT_FOLDER    = "predictions_global_v0"

VISCOUS_CORRECTIONS = ["Grad", "CE", "PTM", "PTB"]
color_names = {"Grad": 'blue', "CE": 'red', "PTM": 'magenta', "PTB": 'green'}

# Centrality labels (0-10%, 10-20%, ..., 80-90%)
CENTRALITY_LABELS = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", 
                     "50-60%", "60-70%", "70-80%", "80-90%"]

RANGE_FILE = (
    "design_pts_Pb_Pb_2760_production/"
    "design_ranges_main_PbPb-2760.dat"
)

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD DESIGN RANGES
# =============================================================================

print("\n" + "="*80)
print("EMULATOR PREDICTIONS FOR GLOBAL v0")
print("="*80)

print("\nLoading design parameter ranges...")
design_range = pd.read_csv(RANGE_FILE, index_col=0)
param_min = design_range["min"].values
param_max = design_range["max"].values
n_params  = len(param_min)
print(f"✓ Loaded {n_params} parameters")

# =============================================================================
# SAMPLE PRIORS
# =============================================================================

print(f"\nGenerating {N_PRIOR_SAMPLES} prior samples...")
np.random.seed(1)

# Load the original posterior samples (for comparison if needed)
print("Loading original posterior samples...")
orig_data_df = pd.read_csv("new_LHC_posterior_samples.csv")
orig_df = orig_data_df.iloc[:, :-1]
orig_samples = orig_df.values
print(f"✓ Loaded {len(orig_samples)} original posterior samples")

# =============================================================================
# LOAD PCA RESULTS
# =============================================================================

print("\nLoading PCA results for global v0...")
if not Path(PCA_RESULTS_FILE).exists():
    print(f"ERROR: PCA results file not found: {PCA_RESULTS_FILE}")
    exit(1)

with open(PCA_RESULTS_FILE, "rb") as f:
    pca_results = pickle.load(f)

print("✓ PCA results loaded for:")
for k, v in pca_results.items():
    print(f"  • {k}: {v['n_components']} PCs")

# =============================================================================
# PCA RECONSTRUCTION
# =============================================================================

def reconstruct_from_pc_scores(pc_scores, inverse_tf_matrix, scaler):
    """
    Reconstruct observables from PC scores.
    
    Parameters:
    -----------
    pc_scores : array (n_samples, n_components)
        Principal component scores
    inverse_tf_matrix : array (n_components, n_observables)
        Inverse transformation matrix from PCA
    scaler : StandardScaler
        Scaler used in PCA
        
    Returns:
    --------
    Y_reconstructed : array (n_samples, n_observables)
        Reconstructed observables
    """
    Y_scaled = pc_scores @ inverse_tf_matrix
    return Y_scaled * scaler.scale_ + scaler.mean_

# =============================================================================
# EMULATOR'S PREDICTION WITH UNCERTAINTY
# =============================================================================

def predict_observables_global_v0(model_parameters, emulators, inverse_tf_matrix, scaler):
    """
    Predict global v0 observables using trained emulators.
    
    Parameters:
    -----------
    model_parameters : array (17,)
        Model parameter values
    emulators : list
        List of trained GP emulators for each PC
    inverse_tf_matrix : array (n_components, n_observables)
        Inverse transformation matrix from PCA
    scaler : StandardScaler
        Scaler from PCA
        
    Returns:
    --------
    y_mean : array (n_observables,)
        Mean prediction
    y_cov : array (n_observables, n_observables)
        Covariance matrix of prediction
    """
    model_parameters = np.array(model_parameters).flatten()
    if model_parameters.shape[0] != 17:
        raise ValueError("Input model parameters must be a 17-dimensional array.")
    
    theta = model_parameters.reshape(1, -1)
    n_pc = len(emulators)
    
    # Predict PC scores with uncertainties
    pc_means = []
    pc_vars = []
    for emulator in emulators:
        mn, std = emulator.predict(theta, return_std=True)
        pc_means.append(mn.flatten()[0])
        pc_vars.append(std.flatten()[0]**2)
    
    pc_means = np.array(pc_means).reshape(1, -1)
    variance_matrix = np.diag(np.array(pc_vars))
    
    # Transform back to observable space
    inverse_transformed_mean = pc_means @ inverse_tf_matrix[:n_pc, :] + scaler.mean_.reshape(1, -1)
    
    A = inverse_tf_matrix[:n_pc, :]
    inverse_transformed_variance = np.einsum('ik,kl,lj->ij', A.T, variance_matrix, A)
    
    return inverse_transformed_mean.flatten(), inverse_transformed_variance

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_fig(fig, filename, folder):
    """Save figure to folder"""
    Path(folder).mkdir(parents=True, exist_ok=True)
    filepath = Path(folder) / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")

def extract_centrality_centers(observable_labels):
    """
    Extract centrality bin centers from observable labels.
    
    Parameters:
    -----------
    observable_labels : list of str
        Labels like "0-10%", "10-20%", etc.
        
    Returns:
    --------
    centrality_centers : array
        Center values of centrality bins
    """
    centrality_centers = []
    for label in observable_labels:
        # Extract numbers from label (e.g., "0-10%" -> [0, 10])
        parts = label.replace('%', '').split('-')
        cent_center = (float(parts[0]) + float(parts[1])) / 2
        centrality_centers.append(cent_center)
    return np.array(centrality_centers)

# =============================================================================
# GENERATE PREDICTIONS FOR ALL CORRECTIONS
# =============================================================================

all_predictions = {}

for corr in VISCOUS_CORRECTIONS:
    print("\n" + "="*80)
    print(f"Processing {corr} - Global v0")
    print("="*80)

    # -------------------------------------------------------------------------
    # Load emulators
    # -------------------------------------------------------------------------
    emu_file = Path(EMULATOR_FOLDER) / f"gp_emulators_global_v0_{corr}.pkl"
    if not emu_file.exists():
        print(f"  ✗ Emulator file not found: {emu_file}")
        print(f"  Skipping {corr}")
        continue

    with open(emu_file, "rb") as f:
        emulators = pickle.load(f)
    
    # Use specified number of PCs or all available
    if N_PC_USE is not None:
        emulators = emulators[:N_PC_USE]
        print(f"  Using {len(emulators)} PCs (out of {len(emulators)} available)")
    else:
        print(f"  Using all {len(emulators)} PCs")
    
    # -------------------------------------------------------------------------
    # Get PCA info
    # -------------------------------------------------------------------------
    info = pca_results[corr]
    n_pc_actual = len(emulators)
    inv_tf = info["inverse_tf_matrix"][:n_pc_actual, :]
    scaler = info["scaler"]
    observable_labels = info["observable_labels"]
    n_obs = len(observable_labels)
    
    print(f"  Observables: {n_obs} centrality bins")
    print(f"  Observable labels: {observable_labels}")
    
    # -------------------------------------------------------------------------
    # Generate prior predictions
    # -------------------------------------------------------------------------
    print(f"  Generating prior predictions...")
    pr_predictions = []
    
    for i, params in enumerate(np.random.uniform(param_min, param_max, (N_PRIOR_SAMPLES, n_params))):
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{N_PRIOR_SAMPLES} prior samples")
        
        y_pred, _ = predict_observables_global_v0(params, emulators, inv_tf, scaler)
        pr_predictions.append(y_pred.flatten())
    
    pr_predictions = np.array(pr_predictions)  # shape (N_PRIOR_SAMPLES, n_obs)
    print(f"  ✓ Prior predictions shape: {pr_predictions.shape}")
    
    # -------------------------------------------------------------------------
    # Generate original posterior predictions (only for Grad)
    # -------------------------------------------------------------------------
    orig_predictions = None
    if corr == 'Grad':
        print(f"  Generating original posterior predictions...")
        orig_predictions = []
        
        for i, theta in enumerate(orig_samples):
            if (i + 1) % 500 == 0:
                print(f"    Processed {i+1}/{len(orig_samples)} original samples")
            
            if theta.shape[0] != 17:
                raise ValueError("Original posterior sample does not have 17 elements.")
            
            y_orig_pred, _ = predict_observables_global_v0(theta, emulators, inv_tf, scaler)
            orig_predictions.append(y_orig_pred)
        
        orig_predictions = np.array(orig_predictions)
        print(f"  ✓ Original posterior predictions shape: {orig_predictions.shape}")
    
    # -------------------------------------------------------------------------
    # Generate correction-specific posterior predictions
    # -------------------------------------------------------------------------
    posterior_file = f"posterior/mcmc_chain_{corr}.csv"
    if Path(posterior_file).exists():
        print(f"  Loading posterior samples from {posterior_file}...")
        samples_df = pd.read_csv(posterior_file)
        posterior_samples = samples_df.values
        print(f"  ✓ Loaded {len(posterior_samples)} posterior samples")
        
        print(f"  Generating {corr} posterior predictions...")
        post_predictions = []
        
        for i, theta in enumerate(posterior_samples):
            if (i + 1) % 500 == 0:
                print(f"    Processed {i+1}/{len(posterior_samples)} posterior samples")
            
            if theta.shape[0] != 17:
                raise ValueError(f"{corr} posterior sample does not have 17 elements.")
            
            y_post_pred, _ = predict_observables_global_v0(theta, emulators, inv_tf, scaler)
            post_predictions.append(y_post_pred)
        
        post_predictions = np.array(post_predictions)
        print(f"  ✓ {corr} posterior predictions shape: {post_predictions.shape}")
    else:
        print(f"  ⚠️ Posterior file not found: {posterior_file}")
        post_predictions = None
    
    # -------------------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------------------
    all_predictions[corr] = {
        'prior': pr_predictions,
        'original_posterior': orig_predictions,
        'posterior': post_predictions,
        'observable_labels': observable_labels,
        'centrality_centers': extract_centrality_centers(observable_labels)
    }
    
    print(f"  ✓ Completed {corr}")

# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

print("\n" + "="*80)
print("SAVING PREDICTIONS...")
print("="*80)

predictions_file = Path(OUTPUT_FOLDER) / "all_predictions_global_v0.pkl"
with open(predictions_file, "wb") as f:
    pickle.dump(all_predictions, f)
print(f"✓ Saved predictions to: {predictions_file}")

# =============================================================================
# PLOTTING: 2x2 PANEL FIGURE
# =============================================================================

print("\n" + "="*80)
print("CREATING PREDICTION PLOTS...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, corr in zip(axes, VISCOUS_CORRECTIONS):
    
    if corr not in all_predictions:
        ax.text(0.5, 0.5, f"No predictions for {corr}",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
        continue
    
    pred_data = all_predictions[corr]
    cent_centers = pred_data['centrality_centers']
    
    # Compute credible intervals: 5th, 50th, 95th percentiles
    pr_env = np.percentile(pred_data['prior'], [5, 50, 95], axis=0)
    
    # Plot prior
    ax.fill_between(cent_centers, pr_env[0, :], pr_env[2, :],
                    hatch='///', edgecolor='black', facecolor='gray', 
                    alpha=0.4, lw=2, label="Prior 90% C.I.")
    ax.plot(cent_centers, pr_env[1, :], 'k--', lw=1.5, alpha=0.6, label="Prior median")
    
    # Plot original posterior (only for Grad)
    if corr == 'Grad' and pred_data['original_posterior'] is not None:
        orig_env = np.percentile(pred_data['original_posterior'], [5, 50, 95], axis=0)
        ax.fill_between(cent_centers, orig_env[0, :], orig_env[2, :],
                        color='orange', alpha=0.4, label="Original Posterior 90% C.I.")
        ax.plot(cent_centers, orig_env[1, :], 'orange', lw=2, alpha=0.8, 
               label="Original Posterior median")
    
    # Plot correction-specific posterior
    if pred_data['posterior'] is not None:
        post_env = np.percentile(pred_data['posterior'], [5, 50, 95], axis=0)
        ax.fill_between(cent_centers, post_env[0, :], post_env[2, :],
                        color=color_names[corr], alpha=0.4, 
                        label=f"{corr} Posterior 90% C.I.")
        ax.plot(cent_centers, post_env[1, :], color=color_names[corr], 
               lw=2, alpha=0.8, label=f"{corr} Posterior median")
    
    # Formatting
    ax.set_title(corr, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

# Final cosmetics
for ax in axes:
    ax.set_xlim(-5, 95)
    ax.set_ylim(-0.02, 0.15)
    ax.set_xlabel("Centrality [%]", fontsize=12)
    ax.set_ylabel(r"Global $v_0$", fontsize=12)

axes[0].legend(framealpha=0.95, fontsize=9, loc='best')

fig.suptitle(
    "Emulator's Predictions for Global $v_0$\n"
    "Pb–Pb 2.76 TeV",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, 'predictions_panel_global_v0.pdf', OUTPUT_FOLDER)

# =============================================================================
# INDIVIDUAL PLOTS FOR EACH CORRECTION
# =============================================================================

print("\nCreating individual plots for each correction...")

for corr in VISCOUS_CORRECTIONS:
    if corr not in all_predictions:
        continue
    
    pred_data = all_predictions[corr]
    cent_centers = pred_data['centrality_centers']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Compute credible intervals
    pr_env = np.percentile(pred_data['prior'], [5, 50, 95], axis=0)
    
    # Plot prior
    ax.fill_between(cent_centers, pr_env[0, :], pr_env[2, :],
                    hatch='///', edgecolor='black', facecolor='gray', 
                    alpha=0.4, lw=2, label="Prior 90% C.I.")
    ax.plot(cent_centers, pr_env[1, :], 'k--', lw=2, alpha=0.7, label="Prior median")
    
    # Plot original posterior (only for Grad)
    if corr == 'Grad' and pred_data['original_posterior'] is not None:
        orig_env = np.percentile(pred_data['original_posterior'], [5, 50, 95], axis=0)
        ax.fill_between(cent_centers, orig_env[0, :], orig_env[2, :],
                        color='orange', alpha=0.4, label="Original Posterior 90% C.I.")
        ax.plot(cent_centers, orig_env[1, :], 'orange', lw=2.5, alpha=0.9, 
               label="Original Posterior median")
    
    # Plot correction-specific posterior
    if pred_data['posterior'] is not None:
        post_env = np.percentile(pred_data['posterior'], [5, 50, 95], axis=0)
        ax.fill_between(cent_centers, post_env[0, :], post_env[2, :],
                        color=color_names[corr], alpha=0.5, 
                        label=f"{corr} Posterior 90% C.I.")
        ax.plot(cent_centers, post_env[1, :], color=color_names[corr], 
               lw=2.5, alpha=0.9, label=f"{corr} Posterior median")
    
    # Formatting
    ax.set_xlabel("Centrality [%]", fontsize=13)
    ax.set_ylabel(r"Global $v_0$", fontsize=13)
    ax.set_title(f"Emulator Predictions - {corr} - Global $v_0$\nPb–Pb 2.76 TeV", 
                fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.95, fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlim(-5, 95)
    ax.set_ylim(-0.02, 0.15)
    
    plt.tight_layout()
    save_fig(fig, f'predictions_global_v0_{corr}.pdf', OUTPUT_FOLDER)

# =============================================================================
# COMPARISON PLOT: ALL POSTERIORS TOGETHER
# =============================================================================

print("\nCreating comparison plot...")

fig, ax = plt.subplots(figsize=(12, 8))

for corr in VISCOUS_CORRECTIONS:
    if corr not in all_predictions:
        continue
    
    pred_data = all_predictions[corr]
    
    if pred_data['posterior'] is not None:
        cent_centers = pred_data['centrality_centers']
        post_env = np.percentile(pred_data['posterior'], [5, 50, 95], axis=0)
        
        ax.fill_between(cent_centers, post_env[0, :], post_env[2, :],
                        color=color_names[corr], alpha=0.3)
        ax.plot(cent_centers, post_env[1, :], color=color_names[corr], 
               lw=2.5, alpha=0.9, label=f"{corr} Posterior median", marker='o', markersize=6)

ax.set_xlabel("Centrality [%]", fontsize=13)
ax.set_ylabel(r"Global $v_0$", fontsize=13)
ax.set_title("Posterior Predictions Comparison - Global $v_0$\nPb–Pb 2.76 TeV", 
            fontsize=14, fontweight="bold")
ax.legend(framealpha=0.95, fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlim(-5, 95)
ax.set_ylim(-0.02, 0.15)

plt.tight_layout()
save_fig(fig, 'predictions_comparison_global_v0.pdf', OUTPUT_FOLDER)

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*80)
print("PREDICTION SUMMARY - GLOBAL v0")
print("="*80)

for corr in VISCOUS_CORRECTIONS:
    if corr not in all_predictions:
        continue
    
    print(f"\n{corr}:")
    pred_data = all_predictions[corr]
    
    print(f"  Prior predictions: {pred_data['prior'].shape}")
    if pred_data['original_posterior'] is not None:
        print(f"  Original posterior predictions: {pred_data['original_posterior'].shape}")
    if pred_data['posterior'] is not None:
        print(f"  {corr} posterior predictions: {pred_data['posterior'].shape}")
    
    print(f"  Centrality bins: {len(pred_data['observable_labels'])}")
    print(f"  Observable labels: {pred_data['observable_labels']}")
    
    # Print median values
    if pred_data['posterior'] is not None:
        post_median = np.median(pred_data['posterior'], axis=0)
        print(f"  Posterior median values:")
        for i, (label, val) in enumerate(zip(pred_data['observable_labels'], post_median)):
            print(f"    {label}: {val:.6f}")

print("\n" + "="*80)
print("✓ EMULATOR PREDICTIONS FOR GLOBAL v0 COMPLETE")
print(f"✓ All outputs saved to: {OUTPUT_FOLDER}/")
print("="*80 + "\n")
