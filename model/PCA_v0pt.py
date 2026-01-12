#!/usr/bin/env python3
"""
Perform PCA on v0(pT) data for all viscous corrections.
Prepares data for emulator training by reducing dimensionality.
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files for different viscous corrections
VISCOUS_CORRECTIONS = {
    'Grad': 'design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_Grad.pkl',
    'CE': 'design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_CE.pkl',
    'PTM': 'design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_PTM.pkl',
    'PTB': 'design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_PTB.pkl'
}
# Note: we must change it to more generic way to load design points for each viscous correction!

# PCA settings
N_COMPONENTS = None  # If None, determined by variance_threshold
VARIANCE_THRESHOLD = 0.99  # Capture 99% of variance

# Species to analyze
SPECIES = 'charged'

# Centralities to include (None = all available)
CENTRALITIES_TO_INCLUDE = None  # Or specify list like [5, 6] for 50-60%, 60-70%.
# Note: we can remove some centralities to improve emulation performance.

# Output folder
OUTPUT_FOLDER = 'pca_results'


