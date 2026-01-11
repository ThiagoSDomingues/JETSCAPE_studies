### Author: OptimusThi

#!/usr/bin/env python3
"""
Script to plot observables
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files for different viscous corrections
VISCOUS_CORRECTIONS = {
    'Grad': 'design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_Grad.pkl',
    'CE':   'design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_CE.pkl',
    'PTM':  'design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_PTM.pkl',
    'PTB':  'design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_PTB.pkl'
}

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# SELECT WHICH VISCOUS CORRECTION TO PLOT
# Options: 'Grad', 'CE', 'PTM', 'PTB'
SELECTED_CORRECTION = 'Grad'
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Centrality to plot (index in centbins array)
CENTRALITY_INDEX = 6  # 50-60%
CENTRALITY_LABEL = '50-60%'

# Species to plot
SPECIES = 'charged'  # 'charged', 'pi', 'kaon', 'proton', 'Sigma', 'Xi'

# =============================================================================
# LOAD DATA: all data is stored as pickle files
# =============================================================================

def load_results(filename):
    """Load pickle file with v0(pT) results"""
    if not Path(filename).exists():
        print(f"Warning: File '{filename}' not found!")
        return None

    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results

# =============================================================================
# PLOTTING FUNCTIONS: plotting design points for all viscous corrections
# =============================================================================

def plot_viscous_corrections_comparison(all_data, cent_idx, species_name):
    """
    Create two subplots:
    1. v0(pT) vs pT (log scale)
    2. v0(pT)/v0 vs pT (log scale)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    colors = {
        'Grad': 'blue',
        'CE': 'red',
        'PTM': 'magenta',
        'PTB': 'green'
    }

    for correction_name, results in all_data.items():
        if len(results) == 0:
            continue

        color = colors.get(correction_name, 'gray')

        if cent_idx not in results[0]['centrality_data']:
            print(f"Warning: Centrality {CENTRALITY_LABEL} not found in {correction_name}")
            continue

        for result in results:
            cent_data = result['centrality_data'][cent_idx]
            if species_name not in cent_data:
                continue

            sp = cent_data[species_name]
            pT = np.array(sp['pt_centers'])
            v0pt = np.array(sp['v0pT'])
            v0_global = sp['v0_global']
            ratio = v0pt / v0_global if v0_global != 0 else np.zeros_like(v0pt)

            axes[0].plot(pT, v0pt, color=color, alpha=0.25, linewidth=0.8)
            axes[1].plot(pT, ratio, color=color, alpha=0.25, linewidth=0.8)

    for ax in axes:
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.tick_params(labelsize=11)
        ax.set_xlim(0.2, 4.0)

    axes[0].set_xlabel(r"$p_T$ [GeV/c]", fontsize=13)
    axes[0].set_ylabel(r"$v_0(p_T)$", fontsize=13)
    axes[0].set_title(
        f"{SELECTED_CORRECTION} viscous correction – {species_name} – {CENTRALITY_LABEL}",
        fontsize=13, fontweight='bold'
    )
    axes[0].set_ylim(-0.1, 0.35)

    axes[1].set_xlabel(r"$p_T$ [GeV/c]", fontsize=13)
    axes[1].set_ylabel(r"$v_0(p_T)/v_0$", fontsize=13)
    axes[1].set_title(
        f"Scaled observable – Pb–Pb 2.76 TeV – {CENTRALITY_LABEL}",
        fontsize=13, fontweight='bold'
    )
    axes[1].set_ylim(-4, 15)

    plt.tight_layout()
    return fig

# plot global v_0 against centrality
def plot_v0_vs_centrality_panel(all_data, species_name):
    """
    Create a 2x2 panel figure showing v0 vs centrality for all viscous corrections.
    Each panel shows all design points as faint lines and highlights one.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()
    
    # Map correction names to subplot positions
    correction_order = ['Grad', 'CE', 'PTM', 'PTB']
    
    for idx, corr_key in enumerate(correction_order):
        if corr_key not in all_data or all_data[corr_key] is None:
            axes[idx].text(0.5, 0.5, f'No data for {corr_key}', 
                          ha='center', va='center', transform=axes[idx].transAxes,
                          fontsize=12, color='red')
            continue
        
        corr_info = VISCOUS_CORRECTIONS[corr_key]
        data = all_data[corr_key]
        color = corr_info['color']
        label = corr_info['label']
        
        cent_bins = data['centrality_bins']
        v0_data_list = data['v0_data']
        
        ax = axes[idx]
        
        # Plot all design points as faint lines
        for dp_data in v0_data_list:
            ax.plot(cent_bins, dp_data['v0_global'], 
                   color=color, alpha=0.2, linewidth=1)
        
        # Highlight selected design point with error bars
        if DESIGN_POINT_INDEX is not None and DESIGN_POINT_INDEX < len(v0_data_list):
            main_dp = v0_data_list[DESIGN_POINT_INDEX]
            ax.errorbar(cent_bins, main_dp['v0_global'], 
                       yerr=main_dp['v0_global_err'],
                       fmt='o-', color=color, markersize=6, capsize=4,
                       linewidth=2, alpha=0.9, 
                       label=f'DP {DESIGN_POINT_INDEX}')
        
        # Add label for all design points
        ax.plot([], [], color=color, alpha=0.2, linewidth=1,
               label=f'All DPs (N={len(v0_data_list)})')
        
        # Axis settings
        ax.set_xlabel('Centrality [%]', fontsize=12)
        ax.set_ylabel(r'$v_0$ (global)', fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, framealpha=0.9, loc='best')
        ax.tick_params(labelsize=10)
        
        # Set x-axis limits (0-100% centrality)
        ax.set_xlim(-5, 105)
        
        # Set consistent y-axis limits across all panels
        ax.set_ylim(-0.02, 0.12)
    
    # Overall title
    fig.suptitle(f'Global $v_0$ vs Centrality – {species_name} – Pb-Pb 2.76 TeV', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    print(f"\nLoading viscous correction: {SELECTED_CORRECTION}")
    print("-" * 80)

    filepath = VISCOUS_CORRECTIONS[SELECTED_CORRECTION]
    results = load_results(filepath)

    if results is None:
        raise RuntimeError("Failed to load selected viscous correction file.")

    all_data = {SELECTED_CORRECTION: results}

    print(f"✓ Loaded {len(results)} design points")

    print("\nGenerating plot...")
    fig = plot_viscous_corrections_comparison(
        all_data,
        CENTRALITY_INDEX,
        SPECIES
    )

    plt.show()
    print("\nDone!")
