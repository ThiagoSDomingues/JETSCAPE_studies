import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from data import * 

def load_data(design_point_folder):
    """
    Load data files for a given design point.

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
    u_moment_file = os.path.join(design_point_folder, 'u_moment_0.dat')
    u_xt_file = os.path.join(design_point_folder, 'universal_0.dat')

    if not os.path.exists(u_moment_file) or not os.path.exists(u_xt_file):
        return None, None

    u_moment_data = np.loadtxt(u_moment_file, comments='#')
    meanpt = u_moment_data[:, 1]
    u_xt_data = np.loadtxt(u_xt_file, comments='#')

    return meanpt, u_xt_data

def plot_spectra(design_point, meanpt, u_xt_data, x_log_scale, overlay_experimental, exp_data=None):
    """
    Plot the spectra for a single design point.

    Parameters:
    -----------
    design_point : int
        The design point number.
    meanpt : np.array
        Mean pT for each centrality bin.
    u_xt_data : np.array
        Scaled spectra U(x_T) for each centrality bin.
    x_log_scale : bool
        Whether to use a log scale for the x-axis.
    overlay_experimental : bool
        Whether to overlay experimental data.
    exp_data : tuple of np.array
        Experimental data points (x, y).
    """
    ptcuts = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 
                       1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 
                       1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2., 2.1, 2.2, 2.3, 
                       2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.2, 3.4, 3.6, 3.8, 4., 10.])
    ptlist = (ptcuts[1:] + ptcuts[:-1]) / 2

    plt.figure(figsize=(10, 6), dpi=300)
    centrality_labels = [
        r'$U_{0-5 \%}$', r'$U_{5-10 \%}$', r'$U_{10-20 \%}$',
        r'$U_{20-30 \%}$', r'$U_{30-40 \%}$', r'$U_{40-50 \%}$',
        r'$U_{50-60 \%}$', r'$U_{60-70 \%}$', r'$U_{70-80 \%}$', r'$U_{80-90 \%}$'
    ]
    
    num_centralities = meanpt.shape[0]
    if overlay_experimental:
        num_centralities = len(exp_centrality_labels)
        centrality_labels = spectra_centrality_labels[0:num_centralities]
        
    for i in range(num_centralities):
        plt.plot(
            ptlist / meanpt[i],
            u_xt_data[i, :],
            label=centrality_labels[i],
            alpha=0.8
        )
        
    if overlay_experimental and exp_data:
        plt.scatter(exp_data[0], exp_data[1], color='black', label='Experimental Data')

    plt.xlabel(r'$x_T = p_T / \langle p_T \rangle$', fontsize=14)
    plt.ylabel(r'$U(x_T)$', fontsize=14)
    plt.title(f'Scaled Spectra $U(x_T)$ for Design Point {design_point}', fontsize=16)
    if x_log_scale:
        plt.xscale('log')
    plt.grid(True)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()

    st.pyplot(plt)

# Streamlit App
st.title("Interactive Scaled Spectra Plot")
st.sidebar.header("Options")

# Sidebar options
viscous_correction = st.sidebar.selectbox(
    "Choose Viscous Correction",
    ["Grad", "Chapman-Enskog R.T.A", "Pratt-Torrieri-McNelis", "Pratt-Torrieri-Bernhard"]
)

design_point = st.sidebar.slider("Select Design Point", 0, 499, 0)

overlay_experimental = st.sidebar.checkbox("Overlay Experimental Data")
x_log_scale = st.sidebar.checkbox("Log Scale for x-axis")
show_parameters = st.sidebar.checkbox("Show Design Point Parameters")

# Example paths
base_folder = f'universal_jetscape/{design_point}'
meanpt, u_xt_data = load_data(base_folder)

# Load experimental data (if available)
experimental_data = None
if overlay_experimental:
    # Replace with actual path to experimental data
    exp_data_path = "path_to_experimental_data.dat"
    if os.path.exists(exp_data_path):
        experimental_data = np.loadtxt(exp_data_path, comments='#').T

if meanpt is not None and u_xt_data is not None:
    plot_spectra(design_point, meanpt, u_xt_data, x_log_scale, overlay_experimental, experimental_data)

    if show_parameters:
        st.sidebar.subheader("Design Point Parameters")
        st.sidebar.write(f"Mean pT: {meanpt}")
else:
    st.error(f"Data files not found for design point {design_point}. Please check the path.")

