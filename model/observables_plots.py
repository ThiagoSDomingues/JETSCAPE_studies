### Author: OptimusThi
"""
Script to plot observables.
"""

import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_fig(fig, filename, folder):
    """Save figure to folder"""
    Path(folder).mkdir(parents=True, exist_ok=True)
    filepath = Path(folder) / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")

# =============================================================================
# PLOTTING FUNCTIONS: principal component analysis
# =============================================================================

def plot_pca_variance(explained_variance, correction_name, save_folder):
    """Plot explained variance by principal components"""
    n_pc = len(explained_variance)
    cumulative = np.cumsum(explained_variance)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Individual variance
    ax1.bar(np.arange(1, n_pc+1), explained_variance, color='steelblue', alpha=0.7)
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title(f"Variance Explained by Each PC - {correction_name}", fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.bar(np.arange(1, n_pc+1), cumulative, color='forestgreen', alpha=0.7)
    ax2.axhline(0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
    ax2.axhline(0.99, color='orange', linestyle='--', linewidth=2, label='99% threshold')
    ax2.set_xlabel("Principal Component", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax2.set_title(f"Cumulative Variance - {correction_name}", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    save_fig(fig, f"pca_variance_{correction_name}.pdf", save_folder)
    return fig

def plot_pc_loadings(inverse_tf_matrix, observable_labels, correction_name, 
                     save_folder, n_pc_to_plot=4):
    """Plot PC loadings to understand what each PC represents"""
    n_pc = min(n_pc_to_plot, inverse_tf_matrix.shape[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i in range(n_pc):
        ax = axes[i]
        loadings = inverse_tf_matrix[i, :]
        
        # Group by centrality
        centralities = sorted(set([label[0] for label in observable_labels]))
        colors = plt.cm.viridis(np.linspace(0, 1, len(centralities)))
        
        idx = 0
        for cent_idx, color in zip(centralities, colors):
            # Find indices for this centrality
            cent_mask = [label[0] == cent_idx for label in observable_labels]
            pt_values = [label[1] for label, mask in zip(observable_labels, cent_mask) if mask]
            cent_loadings = loadings[cent_mask]
            
            ax.plot(pt_values, cent_loadings, 'o-', color=color, 
                   label=f'Cent {cent_idx}', alpha=0.7, markersize=4)
        
        ax.set_xlabel(r'$p_T$ [GeV/c]', fontsize=11)
        ax.set_ylabel(f'PC{i+1} Loading', fontsize=11)
        ax.set_title(f'PC{i+1} Loadings', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        ax.set_xscale('log')
    
    fig.suptitle(f'Principal Component Loadings - {correction_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, f"pc_loadings_{correction_name}.pdf", save_folder)
    return fig

def plot_reconstruction_comparison(Y_data, Y_reconstructed, design_point_idx, 
                                   observable_labels, correction_name, save_folder):
    """Compare original vs reconstructed data for a sample design point"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get data for one design point
    original = Y_data[design_point_idx, :]
    reconstructed = Y_reconstructed[design_point_idx, :]
    
    # Group by centrality and plot
    centralities = sorted(set([label[0] for label in observable_labels]))
    colors = plt.cm.viridis(np.linspace(0, 1, len(centralities)))
    
    for cent_idx, color in zip(centralities, colors):
        # Find indices for this centrality
        cent_mask = [label[0] == cent_idx for label in observable_labels]
        pt_values = [label[1] for label, mask in zip(observable_labels, cent_mask) if mask]
        orig_vals = original[cent_mask]
        recon_vals = reconstructed[cent_mask]
        
        ax.plot(pt_values, orig_vals, 'o-', color=color, 
               label=f'Original Cent {cent_idx}', alpha=0.7, markersize=6)
        ax.plot(pt_values, recon_vals, 's--', color=color, 
               label=f'Reconstructed Cent {cent_idx}', alpha=0.7, markersize=5)
    
    ax.set_xlabel(r'$p_T$ [GeV/c]', fontsize=12)
    ax.set_ylabel(r'$v_0(p_T)$', fontsize=12)
    ax.set_title(f'PCA Reconstruction - Design Point {design_point_idx} - {correction_name}', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xscale('log')
    
    plt.tight_layout()
    save_fig(fig, f"reconstruction_comparison_{correction_name}.pdf", save_folder)
    return fig

def compare_pca_across_corrections(pca_results_dict, save_folder):
    """Compare PCA results across different viscous corrections"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Number of PCs needed
    corrections = list(pca_results_dict.keys())
    n_pcs = [pca_results_dict[corr]['n_components'] for corr in corrections]
    
    ax1 = axes[0]
    ax1.bar(corrections, n_pcs, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Number of PCs', fontsize=12)
    ax1.set_title(f'PCs Required for {VARIANCE_THRESHOLD*100:.0f}% Variance', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: First 3 PC variances
    ax2 = axes[1]
    for i, corr in enumerate(corrections):
        exp_var = pca_results_dict[corr]['explained_variance'][:3]
        x = np.arange(1, len(exp_var)+1) + i*0.2
        ax2.bar(x, exp_var, width=0.2, label=corr, alpha=0.7)
    
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax2.set_title('First 3 PCs Variance', fontsize=13)
    ax2.set_xticks([1.3, 2.3, 3.3])
    ax2.set_xticklabels(['PC1', 'PC2', 'PC3'])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_fig(fig, "pca_comparison_all_corrections.pdf", save_folder)
    return fig
