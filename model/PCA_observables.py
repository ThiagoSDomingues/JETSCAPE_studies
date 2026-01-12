### Author: OptimusThi
#!/usr/bin/env python3
"""
Perform PCA on model calculations data for all viscous corrections.
Prepares data for emulator training by reducing dimensionality.
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import math
from load_calculations import VISCOUS_CORRECTIONS, load_results

# PCA settings
N_COMPONENTS = None  # If None, determined by variance_threshold
VARIANCE_THRESHOLD = 0.99  # Capture 99% of variance

# Species to analyze 
SPECIES = 'charged'

# Centralities to include (None = all available)
CENTRALITIES_TO_INCLUDE = None  # Or specify list like [5, 6] for 50-60%, 60-70%
# Note: we can remove some centralities to improve emulation performance.

# Output folder
OUTPUT_FOLDER = 'pca_results'

def input_matrix(results, species, centralities=None):
    """
    Extract observables data as a matrix for PCA.
    
    Returns:
    --------
    Y_data : array (n_design_points, n_observables)
        Flattened observables values across all centralities and pT bins for differential observables
    design_point_ids : array
        Design point identifiers
    observable_labels : list
        Labels for each observable (cent_idx, pT_bin)
    """
    if results is None or len(results) == 0:
        return None, None, None
    
    # Get centrality indices
    first_result = results[0]
    if centralities is None:
        cent_indices = sorted(first_result['centrality_data'].keys())
    else:
        cent_indices = centralities
    
    # Build data matrix
    Y_data = []
    design_point_ids = []
    observable_labels = []
    
    for result in results:
        # Collect differential pt observables values for this design point
        obspt_values = []
        
        for cent_idx in cent_indices:
            if cent_idx not in result['centrality_data']:
                continue
            cent_data = result['centrality_data'][cent_idx]
            
            if species not in cent_data:
                continue
            
            sp = cent_data[species]
            obspt = sp['obspT']
            
            # Add to flattened vector
            obspt_values.extend(obspt)
            
            # Create labels for first design point
            if len(observable_labels) == 0 or len(observable_labels) < len(obspt_values):
                pt_centers = sp['pt_centers']
                for pt in pt_centers:
                    observable_labels.append((cent_idx, pt))
        
        if len(obspt_values) > 0:
            Y_data.append(obspt_values)
            design_point_ids.append(result['design_point'])
    
    Y_data = np.array(Y_data)
    design_point_ids = np.array(design_point_ids)
    
    print(f"  Data matrix shape: {Y_data.shape}")
    print(f"  Design points: {len(design_point_ids)}")
    print(f"  Observables: {Y_data.shape[1]}")
    
    return Y_data, design_point_ids, observable_labels

# =============================================================================
# PCA FUNCTIONS
# =============================================================================
def perform_pca_on_observables(Y_data, n_components=10, variance_threshold=0.99):
    """
    Perform PCA on input data matrix.
    
    Parameters:
    -----------
    Y_data : array (n_design_points, n_observables)
    n_components : int or None
        If None, determined by variance_threshold
    variance_threshold : float
        Cumulative variance to capture if n_components is None
    
    Returns:
    --------
    scaler : StandardScaler object
    pc_scores : array (n_design_points, n_components)
    inverse_tf_matrix : array (n_components, n_observables)
    explained_variance : array
    """
    # Standardize data
    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y_data)
    
    # SVD decomposition
    u, s, vh = np.linalg.svd(Y_scaled, full_matrices=False)
    
    # Determine number of components
    if n_components is None:
        explained_var_ratio = (s**2) / np.sum(s**2)
        cumulative_var = np.cumsum(explained_var_ratio)
        n_components = np.searchsorted(cumulative_var, variance_threshold) + 1
        print(f"  Selected {n_components} PCs to capture {variance_threshold*100:.1f}% variance")
    
    # PC scores (whitened)
    pc_scores = u[:, :n_components] * math.sqrt(u.shape[0] - 1)
    
    # Inverse transformation matrix
    inverse_tf_matrix = (np.diag(s[:n_components]) @ vh[:n_components, :] * 
                        scaler.scale_.reshape(1, -1) / math.sqrt(u.shape[0] - 1))
    
    # Explained variance
    explained_variance = (s[:n_components]**2) / np.sum(s**2)
    
    print(f"  PC scores shape: {pc_scores.shape}")
    print(f"  Explained variance by first {n_components} PCs: {np.sum(explained_variance):.4f}")
    
    return scaler, pc_scores, inverse_tf_matrix, explained_variance

def reconstruct_from_pca(pc_scores, inverse_tf_matrix, scaler):
    """
    Reconstruct original data from PC scores.
    
    Parameters:
    -----------
    pc_scores : array (n_samples, n_components)
    inverse_tf_matrix : array (n_components, n_observables)
    scaler : StandardScaler object
    
    Returns:
    --------
    Y_reconstructed : array (n_samples, n_observables)
    """
    Y_scaled = pc_scores @ inverse_tf_matrix / math.sqrt(pc_scores.shape[0] - 1)
    Y_reconstructed = scaler.inverse_transform(Y_scaled)
    return Y_reconstructed
