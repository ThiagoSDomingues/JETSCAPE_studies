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
