#!/usr/bin/env python3
"""
Plot emulator predictions against ATLAS experimental data with model/data ratio panels.
Generates predictions on-the-fly from existing emulators.
Creates SEPARATE plots for each viscous correction for v0(pT), global v0, and ratio.
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PLOTTING STYLE (Gardim style)
# =============================================================================

plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["xtick.major.pad"] = 4
plt.rcParams["ytick.major.pad"] = 4
plt.rcParams["xtick.minor.pad"] = 4
plt.rcParams["ytick.minor.pad"] = 4
plt.rcParams["legend.handletextpad"] = 0.0
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["ytick.direction"] = 'in'
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Centrality to plot (0-9 for the 10 centrality bins)
CENTRALITY_INDEX = 7  # 60-70%

# Experimental data files
EXP_V0PT_FILE = f'experimental_data/differential_radial_flow/HEPData-ins2907010-v1-Figure_4a_cent{CENTRALITY_INDEX+1}.csv'
EXP_V0_FILE = 'experimental_data/differential_radial_flow/HEPData-ins2907010-v1-Figure_2a.csv'

# PCA and Emulator files
PCA_RESULTS_V0PT_FILE = "pca_results/pca_results_all_corrections.pkl"
PCA_RESULTS_V0_FILE = "pca_results/pca_results_global_v0_all_corrections.pkl"
EMULATOR_FOLDER_V0PT = "emulators_v0pt"
EMULATOR_FOLDER_V0 = "emulators_global_v0"

# Design ranges and posterior samples
RANGE_FILE = "design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat"
ORIGINAL_POSTERIOR_FILE = "new_LHC_posterior_samples.csv"

# Output folder
OUTPUT_FOLDER = "predictions_vs_experimental"

# Prediction settings
N_PRIOR_SAMPLES = 2000
N_PC_USE_V0PT = 5
N_PC_USE_V0 = None  # None = use all

# Viscous corrections
VISCOUS_CORRECTIONS = ["Grad", "CE", "PTM", "PTB"]
COLOR_NAMES = {"Grad": 'blue', "CE": 'red', "PTM": 'magenta', "PTB": 'green'}

# Centrality labels
EXP_CENTRALITY_LABELS = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", 
                         "40-50%", "50-60%", "60-70%", "70-80%", "80-90%"]

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def configure_axis(ax, fontsize):
    """Configure axis ticks"""
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(labelsize=fontsize)

def save_fig(fig, filename):
    """Save figure"""
    filepath = Path(OUTPUT_FOLDER) / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close(fig)

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_observables(model_parameters, emulators, inverse_tf_matrix, scaler):
    """Predict observables using trained emulators"""
    model_parameters = np.array(model_parameters).flatten()
    if model_parameters.shape[0] != 17:
        raise ValueError("Input model parameters must be a 17-dimensional array.")
    
    theta = model_parameters.reshape(1, -1)
    n_pc = len(emulators)
    
    # Predict PC scores
    pc_means = []
    for emulator in emulators:
        mn, _ = emulator.predict(theta, return_std=True)
        pc_means.append(mn.flatten()[0])
    
    pc_means = np.array(pc_means).reshape(1, -1)
    
    # Transform back to observable space
    inverse_transformed_mean = pc_means @ inverse_tf_matrix[:n_pc, :] + scaler.mean_.reshape(1, -1)
    
    return inverse_transformed_mean.flatten()

def generate_predictions(param_samples, emulators, pca_info, observable_type='v0pt'):
    """Generate predictions for a set of parameter samples"""
    n_samples = len(param_samples)
    inv_tf = pca_info["inverse_tf_matrix"][:len(emulators), :]
    scaler = pca_info["scaler"]
    
    predictions = []
    for i, theta in enumerate(param_samples):
        if (i + 1) % 500 == 0:
            print(f"      Processed {i+1}/{n_samples} samples")
        
        y_pred = predict_observables(theta, emulators, inv_tf, scaler)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    
    # Reshape based on observable type
    if observable_type == 'v0pt':
        predictions = predictions.reshape(n_samples, 10, 29)
    elif observable_type == 'v0':
        predictions = predictions.reshape(n_samples, -1)
    
    return predictions

# =============================================================================
# LOAD EXPERIMENTAL DATA
# =============================================================================

def load_experimental_v0pt(filepath, centrality_idx):
    """Load experimental v0(pT) data from CSV file"""
    if not Path(filepath).exists():
        print(f"Warning: Experimental data file '{filepath}' not found!")
        return None
    
    data = pd.read_csv(filepath, comment='#')
    data.columns = ['pT', 'v0', 'stat_plus', 'stat_minus', 'sys_plus', 'sys_minus']
    
    stat_err = np.abs(data['stat_plus'])
    sys_err = np.abs(data['sys_plus'])
    total_err = np.sqrt(stat_err**2 + sys_err**2)
    
    print(f"  ✓ Loaded v0(pT) for {EXP_CENTRALITY_LABELS[centrality_idx]}: {len(data)} points")
    
    return {
        'pT': data['pT'].values,
        'v0': data['v0'].values,
        'stat_err': stat_err.values,
        'sys_err': sys_err.values,
        'total_err': total_err.values
    }

def load_experimental_v0_global(filepath, centrality_idx):
    """Load experimental global v0 for a specific centrality"""
    if not Path(filepath).exists():
        print(f"Warning: Experimental data file '{filepath}' not found!")
        return None
    
    data = pd.read_csv(filepath, comment='#')
    
    # Find first section
    first_section_end = None
    for i, row in data.iterrows():
        if i > 0 and str(row.iloc[0]) == 'Centrality [%]':
            first_section_end = i
            break
    
    if first_section_end is not None:
        data = data.iloc[:first_section_end].copy()
    
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.columns = ['centrality', 'v0', 'stat_plus', 'stat_minus', 'sys_plus', 'sys_minus']
    
    stat_err = np.abs(data['stat_plus'].values)
    sys_err = np.abs(data['sys_plus'].values)
    total_err = np.sqrt(stat_err**2 + sys_err**2)
    
    # Map centrality index to center value
    centbins = np.array([[0,5], [5,10], [10,20], [20,30], [30,40], [40,50],
                         [50,60], [60,70], [70,80], [80,90]])
    cent_center = (centbins[centrality_idx][0] + centbins[centrality_idx][1]) / 2
    
    # Find closest experimental centrality
    idx = np.argmin(np.abs(data['centrality'].values - cent_center))
    
    print(f"  ✓ Loaded global v0 for {EXP_CENTRALITY_LABELS[centrality_idx]}: v0={data['v0'].values[idx]:.5f}")
    
    return {
        'v0': data['v0'].values[idx],
        'stat_err': stat_err[idx],
        'sys_err': sys_err[idx],
        'total_err': total_err[idx]
    }

# =============================================================================
# PLOTTING FUNCTIONS - SEPARATE PLOTS FOR EACH CORRECTION
# =============================================================================

def plot_v0pt_single_correction(corr, pred_data, exp_data, centrality_idx):
    """Plot v0(pT) for a single correction with model/data ratio panel"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), 
                             gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    
    ax_top = axes[0]
    ax_bot = axes[1]
    
    fontsize = 20
    
    if pred_data['posterior'] is None:
        print(f"  ✗ No posterior predictions for {corr}")
        return None
    
    post_pred = pred_data['posterior'][:, centrality_idx, :]
    pT = pred_data['pt_values']
    
    env = np.percentile(post_pred, [5, 50, 95], axis=0)
    
    # Top panel: v0(pT)
    ax_top.fill_between(pT, env[0, :], env[2, :],
                       color=COLOR_NAMES[corr], alpha=0.3, label=f'{corr} 90% C.I.')
    ax_top.plot(pT, env[1, :], color=COLOR_NAMES[corr], 
               lw=2.5, alpha=0.9, label=f'{corr} Posterior Median')
    
    # Plot experimental data
    if exp_data is not None:
        ax_top.errorbar(exp_data['pT'], exp_data['v0'],
                       yerr=exp_data['total_err'],
                       fmt='o', color='black', markersize=7, capsize=4,
                       linewidth=2, alpha=0.95, label='ATLAS Data', zorder=10)
    
    # Bottom panel: model/data ratio
    if exp_data is not None:
        # Interpolate to experimental pT bins
        env_interp = np.zeros((3, len(exp_data['pT'])))
        for i in range(3):
            env_interp[i, :] = np.interp(exp_data['pT'], pT, env[i, :])
        
        ratio_low = env_interp[0, :] / exp_data['v0']
        ratio_high = env_interp[2, :] / exp_data['v0']
        ratio_med = env_interp[1, :] / exp_data['v0']
        
        ax_bot.fill_between(exp_data['pT'], ratio_low, ratio_high,
                           color=COLOR_NAMES[corr], alpha=0.3)
        ax_bot.plot(exp_data['pT'], ratio_med, color=COLOR_NAMES[corr],
                   lw=2.5, alpha=0.9)
        
        # Experimental uncertainty band
        exp_rel_err = exp_data['total_err'] / exp_data['v0']
        ax_bot.errorbar(exp_data['pT'], np.ones_like(exp_data['pT']),
                       yerr=exp_rel_err, fmt='o', color='black',
                       markersize=7, capsize=4, linewidth=2, alpha=0.95, zorder=10)
    
    # Formatting
    ax_top.set_ylabel(r"$v_0(p_T)$", fontsize=fontsize+2)
    ax_top.legend(loc='best', frameon=False, fontsize=fontsize-2)
    ax_top.set_xscale('log')
    ax_top.grid(True, alpha=0.3, which='both')
    ax_top.set_ylim(-0.02, 0.35)
    configure_axis(ax_top, fontsize-2)
    
    ax_bot.axhline(1.0, color='k', lw=1.5, ls='--')
    ax_bot.set_xlabel(r"$p_T$ [GeV/c]", fontsize=fontsize+2)
    ax_bot.set_ylabel("Model/Data", fontsize=fontsize-2)
    ax_bot.set_xscale('log')
    ax_bot.grid(True, alpha=0.3, which='both')
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlim(0.2, 3.5)
    configure_axis(ax_bot, fontsize-2)
    
    fig.suptitle(rf"$v_0(p_T)$ — {corr} — {EXP_CENTRALITY_LABELS[centrality_idx]} — Pb-Pb 2.76 TeV",
                fontsize=fontsize+4, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.05)
    
    return fig

def plot_v0_global_single_correction(corr, pred_data, exp_data, centrality_idx):
    """Plot global v0 for a single correction with model/data comparison"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    fontsize = 20
    
    if pred_data['posterior'] is None:
        print(f"  ✗ No posterior predictions for {corr}")
        return None
    
    post_pred = pred_data['posterior'][:, centrality_idx]
    env = np.percentile(post_pred, [5, 50, 95])
    
    # Plot prediction bar
    x = [0]
    bars = ax.bar(x, [env[1]], yerr=[[env[1] - env[0]], [env[2] - env[1]]],
                  color=COLOR_NAMES[corr], alpha=0.7, capsize=5, width=0.5,
                  label=f'{corr} Posterior')
    
    # Plot experimental value
    if exp_data is not None:
        ax.axhline(exp_data['v0'], color='black', lw=2.5, 
                  label='ATLAS Data', zorder=10)
        ax.axhspan(exp_data['v0'] - exp_data['total_err'],
                  exp_data['v0'] + exp_data['total_err'],
                  alpha=0.2, color='gray', zorder=0)
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([corr], fontsize=fontsize)
    ax.set_ylabel(r"Global $v_0$", fontsize=fontsize+2)
    ax.set_title(rf"Global $v_0$ — {corr} — {EXP_CENTRALITY_LABELS[centrality_idx]} — Pb-Pb 2.76 TeV",
                fontsize=fontsize+2, fontweight='bold')
    ax.legend(fontsize=fontsize-2, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.08)
    ax.set_xlim(-0.6, 0.6)
    configure_axis(ax, fontsize-2)
    
    plt.tight_layout()
    
    return fig

def plot_ratio_single_correction(corr, pred_data, exp_data_v0pt, exp_data_v0, centrality_idx):
    """Plot v0(pT)/v0 ratio for a single correction with model/data ratio panel"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10),
                            gridspec_kw={'height_ratios': [3, 1]},
                            sharex=True)
    
    ax_top = axes[0]
    ax_bot = axes[1]
    
    fontsize = 20
    
    # Calculate experimental ratio
    exp_ratio = None
    if exp_data_v0pt is not None and exp_data_v0 is not None:
        exp_ratio = exp_data_v0pt['v0'] / exp_data_v0['v0']
        exp_ratio_err = exp_ratio * np.sqrt(
            (exp_data_v0pt['total_err'] / exp_data_v0pt['v0'])**2 +
            (exp_data_v0['total_err'] / exp_data_v0['v0'])**2
        )
    
    if pred_data['posterior'] is None:
        print(f"  ✗ No posterior predictions for {corr}")
        return None
    
    post_pred = pred_data['posterior'][:, centrality_idx, :]
    pT = pred_data['pt_values']
    
    env = np.percentile(post_pred, [5, 50, 95], axis=0)
    
    # Top panel: ratio
    ax_top.fill_between(pT, env[0, :], env[2, :],
                       color=COLOR_NAMES[corr], alpha=0.3, label=f'{corr} 90% C.I.')
    ax_top.plot(pT, env[1, :], color=COLOR_NAMES[corr],
               lw=2.5, alpha=0.9, label=f'{corr} Posterior Median')
    
    # Plot experimental ratio
    if exp_ratio is not None:
        ax_top.errorbar(exp_data_v0pt['pT'], exp_ratio,
                       yerr=exp_ratio_err,
                       fmt='o', color='black', markersize=7, capsize=4,
                       linewidth=2, alpha=0.95, label='ATLAS Data', zorder=10)
    
    # Bottom panel: model/data ratio
    if exp_ratio is not None:
        # Interpolate to experimental pT bins
        env_interp = np.zeros((3, len(exp_data_v0pt['pT'])))
        for i in range(3):
            env_interp[i, :] = np.interp(exp_data_v0pt['pT'], pT, env[i, :])
        
        ratio_low = env_interp[0, :] / exp_ratio
        ratio_high = env_interp[2, :] / exp_ratio
        ratio_med = env_interp[1, :] / exp_ratio
        
        ax_bot.fill_between(exp_data_v0pt['pT'], ratio_low, ratio_high,
                           color=COLOR_NAMES[corr], alpha=0.3)
        ax_bot.plot(exp_data_v0pt['pT'], ratio_med, color=COLOR_NAMES[corr],
                   lw=2.5, alpha=0.9)
        
        # Experimental uncertainty band
        exp_rel_err = exp_ratio_err / exp_ratio
        ax_bot.errorbar(exp_data_v0pt['pT'], np.ones_like(exp_data_v0pt['pT']),
                       yerr=exp_rel_err, fmt='o', color='black',
                       markersize=7, capsize=4, linewidth=2, alpha=0.95, zorder=10)
    
    # Formatting
    ax_top.set_ylabel(r"$v_0(p_T) / v_0^{\mathrm{global}}$", fontsize=fontsize+2)
    ax_top.legend(loc='best', frameon=False, fontsize=fontsize-2)
    ax_top.set_xscale('log')
    ax_top.grid(True, alpha=0.3, which='both')
    ax_top.set_ylim(0, 10)
    configure_axis(ax_top, fontsize-2)
    
    ax_bot.axhline(1.0, color='k', lw=1.5, ls='--')
    ax_bot.set_xlabel(r"$p_T$ [GeV/c]", fontsize=fontsize+2)
    ax_bot.set_ylabel("Model/Data", fontsize=fontsize-2)
    ax_bot.set_xscale('log')
    ax_bot.grid(True, alpha=0.3, which='both')
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlim(0.2, 3.5)
    configure_axis(ax_bot, fontsize-2)
    
    fig.suptitle(rf"$v_0(p_T) / v_0^{{\mathrm{{global}}}}$ — {corr} — {EXP_CENTRALITY_LABELS[centrality_idx]} — Pb-Pb 2.76 TeV",
                fontsize=fontsize+4, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.05)
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PLOTTING EMULATOR PREDICTIONS VS ATLAS EXPERIMENTAL DATA")
    print("="*80)
    
    print(f"\nCentrality: {EXP_CENTRALITY_LABELS[CENTRALITY_INDEX]}")
    
    # Load experimental data
    print("\nLoading experimental data...")
    exp_data_v0pt = load_experimental_v0pt(EXP_V0PT_FILE, CENTRALITY_INDEX)
    exp_data_v0 = load_experimental_v0_global(EXP_V0_FILE, CENTRALITY_INDEX)
    
    # Load PCA results
    print("\nLoading PCA results...")
    with open(PCA_RESULTS_V0PT_FILE, 'rb') as f:
        pca_results_v0pt = pickle.load(f)
    with open(PCA_RESULTS_V0_FILE, 'rb') as f:
        pca_results_v0 = pickle.load(f)
    print("  ✓ PCA results loaded")
    
    # Load design ranges and posterior samples
    print("\nLoading design ranges and posterior samples...")
    design_range = pd.read_csv(RANGE_FILE, index_col=0)
    param_min = design_range["min"].values
    param_max = design_range["max"].values
    orig_data_df = pd.read_csv(ORIGINAL_POSTERIOR_FILE)
    orig_samples = orig_data_df.iloc[:, :-1].values
    print(f"  ✓ Loaded {len(param_min)} parameters and {len(orig_samples)} original posterior samples")
    
    # Generate predictions for all corrections
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS FROM EMULATORS")
    print("="*80)
    
    all_predictions_v0pt = {}
    all_predictions_v0 = {}
    all_predictions_ratio = {}
    
    for corr in VISCOUS_CORRECTIONS:
        print(f"\n--- Processing {corr} ---")
        
        # Load emulators
        emu_file_v0pt = Path(EMULATOR_FOLDER_V0PT) / f"gp_emulators_{corr}.pkl"
        emu_file_v0 = Path(EMULATOR_FOLDER_V0) / f"gp_emulators_global_v0_{corr}.pkl"
        
        if not emu_file_v0pt.exists() or not emu_file_v0.exists():
            print(f"  ✗ Emulator files not found")
            continue
        
        with open(emu_file_v0pt, 'rb') as f:
            emulators_v0pt = pickle.load(f)[:N_PC_USE_V0PT]
        
        with open(emu_file_v0, 'rb') as f:
            emulators_v0 = pickle.load(f)
            if N_PC_USE_V0 is not None:
                emulators_v0 = emulators_v0[:N_PC_USE_V0]
        
        print(f"  ✓ Loaded emulators: {len(emulators_v0pt)} PCs (v0pt), {len(emulators_v0)} PCs (v0)")
        
        # Get PCA info
        pca_info_v0pt = pca_results_v0pt[corr]
        pca_info_v0 = pca_results_v0[corr]
        
        # Generate samples
        np.random.seed(1)
        prior_samples = np.random.uniform(param_min, param_max, (N_PRIOR_SAMPLES, len(param_min)))
        
        posterior_file = f"posterior/mcmc_chain_{corr}.csv"
        posterior_samples = None
        if Path(posterior_file).exists():
            samples_df = pd.read_csv(posterior_file)
            posterior_samples = samples_df.values
            print(f"  ✓ Loaded {len(posterior_samples)} {corr} posterior samples")
        else:
            print(f"  ⚠️  Posterior file not found: {posterior_file}")
        
        # Generate v0(pT) predictions
        print(f"  Generating v0(pT) predictions...")
        print(f"    Prior...")
        prior_v0pt = generate_predictions(prior_samples, emulators_v0pt, pca_info_v0pt, 'v0pt')
        
        orig_v0pt = None
        if corr == 'Grad':
            print(f"    Original posterior...")
            orig_v0pt = generate_predictions(orig_samples, emulators_v0pt, pca_info_v0pt, 'v0pt')
        
        post_v0pt = None
        if posterior_samples is not None:
            print(f"    {corr} posterior...")
            post_v0pt = generate_predictions(posterior_samples, emulators_v0pt, pca_info_v0pt, 'v0pt')
        
        pt_values = np.array([label[1] for label in pca_info_v0pt["observable_labels"] if label[0] == 0])
        
        all_predictions_v0pt[corr] = {
            'prior': prior_v0pt,
            'original_posterior': orig_v0pt,
            'posterior': post_v0pt,
            'pt_values': pt_values
        }
        
        # Generate global v0 predictions
        print(f"  Generating global v0 predictions...")
        print(f"    Prior...")
        prior_v0 = generate_predictions(prior_samples, emulators_v0, pca_info_v0, 'v0')
        
        orig_v0 = None
        if corr == 'Grad':
            print(f"    Original posterior...")
            orig_v0 = generate_predictions(orig_samples, emulators_v0, pca_info_v0, 'v0')
        
        post_v0 = None
        if posterior_samples is not None:
            print(f"    {corr} posterior...")
            post_v0 = generate_predictions(posterior_samples, emulators_v0, pca_info_v0, 'v0')
        
        all_predictions_v0[corr] = {
            'prior': prior_v0,
            'original_posterior': orig_v0,
            'posterior': post_v0
        }
        
        # Compute ratio
        print(f"  Computing ratio v0(pT)/v0...")
        ratio_prior = prior_v0pt / prior_v0[:, :, np.newaxis]
        ratio_orig = None if orig_v0pt is None else orig_v0pt / orig_v0[:, :, np.newaxis]
        ratio_post = None if post_v0pt is None else post_v0pt / post_v0[:, :, np.newaxis]
        
        all_predictions_ratio[corr] = {
            'prior': ratio_prior,
            'original_posterior': ratio_orig,
            'posterior': ratio_post,
            'pt_values': pt_values
        }
        
        print(f"  ✓ Completed {corr}")
    
    # Create plots - SEPARATE for each correction
    print("\n" + "="*80)
    print("CREATING PLOTS (SEPARATE FOR EACH CORRECTION)...")
    print("="*80)
    
    for corr in VISCOUS_CORRECTIONS:
        if corr not in all_predictions_v0pt:
            continue
        
        print(f"\n--- Creating plots for {corr} ---")
        
        # Plot 1: v0(pT)
        print(f"1. Plotting v0(pT) for {corr}...")
        if exp_data_v0pt is not None:
            fig = plot_v0pt_single_correction(corr, all_predictions_v0pt[corr], 
                                             exp_data_v0pt, CENTRALITY_INDEX)
            if fig is not None:
                save_fig(fig, f'v0pt_vs_exp_{corr}_{EXP_CENTRALITY_LABELS[CENTRALITY_INDEX].replace("-", "")}.pdf')
        
        # Plot 2: Global v0
        print(f"2. Plotting global v0 for {corr}...")
        if exp_data_v0 is not None:
            fig = plot_v0_global_single_correction(corr, all_predictions_v0[corr], 
                                                  exp_data_v0, CENTRALITY_INDEX)
            if fig is not None:
                save_fig(fig, f'v0_global_vs_exp_{corr}_{EXP_CENTRALITY_LABELS[CENTRALITY_INDEX].replace("-", "")}.pdf')
        
        # Plot 3: Ratio
        print(f"3. Plotting ratio for {corr}...")
        if exp_data_v0pt is not None and exp_data_v0 is not None:
            fig = plot_ratio_single_correction(corr, all_predictions_ratio[corr], 
                                              exp_data_v0pt, exp_data_v0, CENTRALITY_INDEX)
            if fig is not None:
                save_fig(fig, f'ratio_vs_exp_{corr}_{EXP_CENTRALITY_LABELS[CENTRALITY_INDEX].replace("-", "")}.pdf')
    
    print("\n" + "="*80)
    print("✓ PLOTTING COMPLETE")
    print(f"Output folder: {OUTPUT_FOLDER}/")
    print(f"Created plots for: {list(all_predictions_v0pt.keys())}")
    print("="*80 + "\n")
