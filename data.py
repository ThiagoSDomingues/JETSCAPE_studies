import numpy as np
import os

centrality_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", 
                             "40-50%", "50-60%", "60-70%", "70-80%", "80-90%"
                    ] # 10 simulations centrality bins

# Define experimental centrality labels
exp_centrality_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", 
                             "40-50%", "50-60%"
                            ] # 7 experimental centrality bins                           

exp_markers = ['o', 's', '^', 'D', 'v', '<', '>']
                            
idf_label = {
            0 : 'Grad',
            1 : 'Chapman-Enskog R.T.A',
            2 : 'Pratt-Torrieri-McNelis',
            3 : 'Pratt-Torrieri-Bernhard'
            } # labels for different delta_f models

# Base directory where all the data folders are located
data_base_dir = 'universal_jetscape'           

# pt cuts for spectrum calculation 
ptcuts = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 
                   1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 
                   1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2., 2.1, 2.2, 2.3, 
                   2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.2, 3.4, 3.6, 3.8, 4., 10.])

# Calculate midpoints for pt bins
ptlist = (ptcuts[1:] + ptcuts[:-1]) / 2 
            
# Function to load the experimetal data
def load_experimental():
    
    # File path for the experimental data
    exp_file_path = "ALICE_PbPb2p76.dat"
    num_centrality_bins = len(exp_centrality_labels) # 7 
    num_xt_bins = 41 # 41 bins

    data = np.loadtxt(exp_file_path, comments='#')
    
    # Initialize storage for xT and U(x_T) values for each centrality bin
    xt_values = np.zeros((num_centrality_bins, num_xt_bins))
    u_xt_values = np.zeros((num_centrality_bins, num_xt_bins))
    u_xt_error = np.zeros((num_centrality_bins, num_xt_bins))
    
    # Fill arrays with x_T and U(x_T) values
    for i in range(num_centrality_bins):
        xt_values[i, :] = data[i, 0::3]  # indices for pT values
        u_xt_values[i, :] = data[i, 1::3]  # indices for U(x_T) values
        u_xt_error[i, :] = data[i, 2::3] # indices for U(x_T) errors
    
    return xt_values, u_xt_values, u_xt_error
    
def remove_design_pts(idf):
    
    # Problematic design points for Pb-Pb 2760 and Au-Au 200 with 500 design points
    nan_sets_by_deltaf = {
        0: set([334, 341, 377, 429, 447, 483]),
        1: set([285, 334, 341, 447, 483, 495]),
        2: set([209, 280, 322, 334, 341, 412, 421, 424, 429, 432, 446, 447, 453, 468, 483, 495]),
        3: set([60, 232, 280, 285, 322, 324, 341, 377, 432, 447, 464, 468, 482, 483, 485, 495])
    }

    # Other sets of problematic design points
    unfinished_events_design_pts_set = set([289, 324, 326, 459, 462, 242, 406, 440, 123])
    strange_features_design_pts_set = set([289, 324, 440, 459, 462])
    
    # Union of all problematic design points for a specific delta_f
    nan_design_pts_set = nan_sets_by_deltaf[idf]
    delete_design_pts_set = nan_design_pts_set.union(unfinished_events_design_pts_set.union(strange_features_design_pts_set))
    
    return delete_design_pts_set
    # Load design points data, removing problematic points
#    file_name = 'design_pts_Pb_Pb_2760_production/design_points_main_PbPb-2760.dat'
#    design = pd.read_csv(file_name, index_col=0)
#    design = design.drop(index=delete_design_pts_set)  # Remove rows with problematic design points
    
#    return design

def load_spectra_data(design_idx, idf):
    """
    Load the Universal spectra data file for a given design point.

    Parameters:
    -----------
    design_point_folder : str
        Path to the folder containing the design point data.

    Returns:
    --------
    meanpt : np.array
        Mean pT for each centrality bin.
    u_xt_data : np.array
        Scaled spectra U(x_T) for each centrality bin.
    """
    design_point_folder = f'universal_jetscape/{design_idx}'
    # Path to data files
    u_moment_file = os.path.join(design_point_folder, f'u_moment_{idf}.dat')
    u_xt_file = os.path.join(design_point_folder, f'universal_{idf}.dat')
    
    # Check if files exist
    if not os.path.exists(u_moment_file) or not os.path.exists(u_xt_file):
        return None, None
    
    # Read data from files
    u_moment_data = np.loadtxt(u_moment_file, comments='#')
    meanpt = u_moment_data[:, 1] # mean pT for each centrality bin
    u_xt_data = np.loadtxt(u_xt_file, comments='#')

    return meanpt, u_xt_data    


def load_moment_data(num_design_points, num_corrections=4):
    """
    Load moment data for all design points and corrections.

    Parameters:
    -----------
    base_folder : str
        Path to the base folder containing design point subfolders.
    num_design_points : int
        Total number of design points.
    num_corrections : int
        Number of viscous corrections (default: 4).

    Returns:
    --------
    data : dict
        Nested dictionary where keys are corrections, and values are pandas DataFrames
        for each correction containing moments for all design points and centralities.
    """
    data = {i: [] for i in range(num_corrections)}
    for correction in range(num_corrections):
        for dp in range(num_design_points):
            file_path = os.path.join(data_base_dir, str(dp), f"u_moment_{correction}.dat")
            if os.path.exists(file_path):
                # Load data and append it
                dp_data = np.loadtxt(file_path, comments='#')
                data[correction].append(dp_data)
        # Concatenate data for all design points for this correction
        data[correction] = np.vstack(data[correction]) if data[correction] else None
    return data
    
def calculate_kl_divergence(u_xt_data):
    """
    Calculate the KL divergence for each centrality curve with respect to the average curve.

    Parameters:
    -----------
    u_xt_data : np.array
        Scaled spectra U(x_T) for each centrality bin.

    Returns:
    --------
    kl_divergences : list
        KL divergence for each centrality curve.
    """
    mean_curve = np.mean(u_xt_data, axis=0)
    kl_divergences = []

    for i, curve in enumerate(u_xt_data):
        kl_div = np.sum(curve * np.log(curve / mean_curve))
        kl_divergences.append(kl_div)

    return kl_divergences

# Example usage
#kl_divergences = calculate_kl_divergence(u_xt_data)
#for i, kl in enumerate(kl_divergences):
#    print(f"Centrality {i}: KL Divergence = {kl:.4f}")

from scipy.interpolate import CubicSpline

def interpolate_spectra(x_T, u_xt_data):
    """
    Interpolates U(x_T) using cubic splines.

    Parameters:
    -----------
    x_T : np.array
        Array of x_T values.
    u_xt_data : np.array
        Scaled spectra U(x_T) for each centrality bin.

    Returns:
    --------
    interpolators : list
        A list of cubic spline interpolators, one for each centrality.
    """
    interpolators = []
    for curve in u_xt_data:
        interpolators.append(CubicSpline(x_T, curve))
    return interpolators

# Example usage
#x_T = ptlist / meanpt.mean()  # Assuming ptlist and meanpt are defined
#interpolators = interpolate_spectra(x_T, u_xt_data)

# Example: Evaluate the interpolated function at a fine grid
#x_T_fine = np.linspace(x_T.min(), x_T.max(), 500)
#interpolated_curve = interpolators[0](x_T_fine)

    

#def load_spectra_design_pts(idf, remove):

#    for design_idx in range(num_design_points):
#        meanpt, u_xt_data = load_spectra_data(design_idx, idf)    
    
#def load_moment_data(base_dir, idf, exp_data):
    
#    for design_idx in range(num_design_points):
    
        # Construct the path for the current design point    
#        design_point_folder = os.path.join(design_points_path, str(design_idx))
    
#        u_moment_file = os.path.join(design_point_folder, 'u_moment_0.dat')
#        u_xt_file = os.path.join(design_point_folder, 'universal_0.dat')
        
        # Check if files exist
#        if not os.path.exists(u_moment_file) or not os.path.exists(u_xt_file):
#            print(f"Data files not found for design point {design_idx}. Skipping...")
#            continue
        
        # Read data from files
#        u_moment_data = np.loadtxt(u_moment_file, comments='#')
#        u_xt_data = np.loadtxt(u_xt_file, comments='#')

        # Extract mean pT for each centrality bin
#        meanpt = u_moment_data[:, 1]  # mean pT for each centrality bin
   
#    return                
