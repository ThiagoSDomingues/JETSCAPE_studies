import os
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
from data import * 

spectra_centrality_labels = [
        r'$U_{0-5 \%}$', r'$U_{5-10 \%}$', r'$U_{10-20 \%}$',
        r'$U_{20-30 \%}$', r'$U_{30-40 \%}$', r'$U_{40-50 \%}$',
        r'$U_{50-60 \%}$', r'$U_{60-70 \%}$', r'$U_{70-80 \%}$', r'$U_{80-90 \%}$' 
    ]

idf_color = {0 : 'blue', 1 : 'red', 2 : 'magenta', 3 : 'green'} # colors for different df models
idf_names = ['Grad', 'Chapman-Enskog R.T.A', 'Pratt-Torrieri-McNelis', 'Pratt-Torrieri-Bernhard'] # names for different df models
idf_label_short = {
            0 : 'Grad',
            1 : 'CE',
            2 : 'PTM',
            3 : 'PTB'
            } # short labels for different df models 
            
            
### Design points: create plots for all desing points, in all centralities and different viscous corrections ### 



def plot_scaled_spectra(design_idx, idf, overlay_experimental, x_log_scale):
    """
    Plots the scaled spectra U(x_T) for a single design point.

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
    
    # Load spectra data
    meanpt, u_xt_data = load_spectra_data(design_idx, idf)    # mean pT, universal spectra for each centrality bin
        
    # Plot scaled spectra for all centrality bins
    plt.figure(figsize=(10, 6), dpi=300)
    
    num_centralities = meanpt.shape[0]
    centrality_labels = spectra_centrality_labels
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
    
    if overlay_experimental:
        # Load experimental data
        exp_xT, exp_u_xT, err_u_xT = load_experimental()
        for i in range(num_centralities):
            plt.errorbar(exp_xT[i], exp_u_xT[i], yerr=err_u_xT[i], fmt=exp_markers[i], label=f"Centrality {exp_centrality_labels[i]}", markersize=6, capsize=4, linestyle='none')
            
    # Add labels, title, and legend
    plt.xlabel(r'$x_T = p_T / \langle p_T \rangle$', fontsize=14)
    plt.ylabel(r'$U(x_T)$', fontsize=14)
    plt.title(f'Scaled Spectra $U(x_T)$ for Design Point {design_idx} {idf_label_short[idf]} Pb-Pb 2.76 TeV', fontsize=16)
    if x_log_scale:
        plt.xscale('log')
    plt.grid(True)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()

    # Show plot
    plt.show()        
       

def plot_all_design_pts(idf):
    """
    Plots the scaled spectra U(x_T) for all design points.

    Parameters:
    -----------
    design_points_path : str
        The parent directory containing folders for each design point.
        Example: 'universal_jetscape/'.
    num_design_points : int
        The total number of design points to plot.
    """
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Number of design points (folders) and centrality bins
    num_design_points = 500
    delete_design_pts_set = remove_design_pts(idf)
    for design_idx in range(num_design_points):
        if design_idx in delete_design_pts_set:
            continue
            
        meanpt, u_xt_data = load_spectra_data(design_idx, idf)
        num_centralities = len(exp_centrality_labels)
        for i in range(num_centralities):
            plt.plot(
                ptlist / meanpt[i],
                u_xt_data[i, :],
                alpha=0.3,  # Use transparency to visualize overlaps
                color=idf_color[idf]  # Use a single color for all design points
            )
    
    exp_xT, exp_u_xT, err_u_xT = load_experimental()        
    # Add experimental data to the plot
    for i in range(len(exp_centrality_labels)):
        plt.errorbar(
            exp_xT[i],
            exp_u_xT[i],
            yerr=err_u_xT[i],
            fmt=exp_markers[i],
            label=f"Centrality {exp_centrality_labels[i]}",
            markersize=6,
            capsize=4,
            linestyle='none'
        )        
    # Customize plot aesthetics
    plt.xlabel(r'$x_T = p_T / \langle p_T \rangle$', fontsize=14)
    plt.ylabel(r'$U(x_T)$', fontsize=14)
    plt.title(f'Scaled Spectra $U(x_T)$ for Filtered Design Points {idf_label_short[idf]} Pb-Pb 2.76 TeV', fontsize=16)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10, loc='upper right')
    plt.xscale("log")
#    plt.yscale("log")
    plt.tight_layout()

    # Show plot
    plt.show()        

### Sensitivity analysis: create a grouped bar chart for all the viscous corrections ### 

#@plot
#def spectra_sobol_sensitivity(idf):

#   """
#    Plots Sobol Sensitivity index of the scaled spectra observables to model parameters grouped bar chart for all the viscous corrections.
#    (See https://journals.plos.org/plosone/article/file?type=supplementary&id=info:doi/10.1371/journal.pone.0095610.s003)

#    """  
     
# labels = [''] model parameters labels

#x = np.arange(len(labels)) # the label locations
#width = 0.35 # the width of the bars

#fig, ax = plt.subplots()
#grad = ax.bar(x - width/4, grad, width, label=df_names[0], color=idf_color[0]) 
#ce = ax.bar(x - width/2, ce, width, label=df_names[1], color=idf_color[1]) 
#ptb = ax.bar(x + width/2, ptb, width, label=df_names[2], color=idf_color[2]) 
#ptm =ax.bar(x + width/4, ptm, width, label=df_names[3], color=idf_color[3])

# Add text for labels, title, and custom x-axis tick labels, etc...
#ax.set_ylabel('Sobol index')
#ax.set_title('Grouped Sobol sensitivity analysis')
#ax.set_ticks(x)
#ax.set_xtickklabels(labels)
#ax.legend()

#fig.tight_layout()
#plt.show()
