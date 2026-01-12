### Author: OptimustThi
import pickle
from pathlib import Path

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

# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(filename):
    """Load pickle file with v0(pT) results"""
    if not Path(filename).exists():
        print(f"Warning: File '{filename}' not found!")
        return None
    
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results
