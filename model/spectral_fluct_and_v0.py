#!/usr/bin/env python3
"""
Spectral Fluctuations and v_0(p_T) Analysis
==========================================

Implements the analysis of arXiv:2004.00690:
- Spectral fluctuations at fixed multiplicity
- Normalized correlation function C(p_T^a, p_T^b)
- Extraction of v_0(p_T) via factorization
- Comparison to simple model (Eq. 21)

Author: Thiago Siqueira Domingues
Date: February 2026
"""

# =============================================================================
# CORE PHYSICS
# =============================================================================

def calculate_spectrum_fluctuations_fixed_N(dNdpt_events, pt_bins):
    N = np.trapz(dNdpt_events, pt_bins, axis=1)
    N_mean = np.mean(N)
    sigma_N_sq = np.var(N, ddof=1)

    mean_spectrum = np.mean(dNdpt_events, axis=0)

    delta_N = N - N_mean
    delta_Np = dNdpt_events - mean_spectrum[np.newaxis, :]

    corr_standard = np.einsum('ei,ej->ij', delta_Np, delta_Np) / len(N)
    cov_Np_N = np.mean(delta_Np * delta_N[:, np.newaxis], axis=0)

    corr_fixed_N = corr_standard - np.outer(cov_Np_N, cov_Np_N) / sigma_N_sq

    return corr_fixed_N, mean_spectrum, N_mean, sigma_N_sq


def calculate_correlation_function_C(corr_fixed_N, mean_spectrum):
    norm = np.outer(mean_spectrum, mean_spectrum)
    C = np.zeros_like(corr_fixed_N)
    mask = norm > 1e-15
    C[mask] = corr_fixed_N[mask] / norm[mask]
    return C


def extract_v0_from_factorization(C_matrix, pt_bins, pt_range=(0.5, 3.0)):
    mask = (pt_bins >= pt_range[0]) & (pt_bins <= pt_range[1])
    C_sub = C_matrix[np.ix_(mask, mask)]

    eigvals, eigvecs = np.linalg.eigh(C_sub)
    idx = np.argmax(np.abs(eigvals))

    v0_sub = np.sqrt(np.abs(eigvals[idx])) * eigvecs[:, idx]

    if np.mean(v0_sub[:len(v0_sub)//2]) > 0:
        v0_sub *= -1

    v0 = np.zeros(len(pt_bins))
    v0[mask] = v0_sub
    return v0, eigvals[idx]


def calculate_simple_model_C(mean_spectrum, pt_bins, dNdpt_events):
    N_evt = np.trapz(dNdpt_events, pt_bins, axis=1)
    pT_evt = np.trapz(pt_bins * dNdpt_events, pt_bins, axis=1) / N_evt

    pT_mean = np.mean(pT_evt)

    delta_pT = pT_evt - pT_mean
    delta_N = N_evt - np.mean(N_evt)

    sigma_N_sq = np.var(N_evt, ddof=1)
    cov_pT_N = np.mean(delta_pT * delta_N)

    sigma_n_pT_sq = np.var(pT_evt, ddof=1) - cov_pT_N**2 / sigma_N_sq
    sigma_n_pT = np.sqrt(max(0.0, sigma_n_pT_sq))

    pref = (sigma_n_pT / pT_mean)**2
    f = 2.0 * pt_bins / pT_mean - 2.0
    return pref * np.outer(f, f), pT_mean, sigma_n_pT

# =============================================================================
# PLOTTING
# =============================================================================

def plot_C_slices_vs_factorization(C_matrix, v0_pt, pt_bins,
                                   C_model, pT_mean,
                                   centrality_label):

    pt_b_values = np.arange(0.6, 3.5, 0.4)
    colors = plt.cm.viridis(np.linspace(0, 1, len(pt_b_values)))

    C_fact = np.outer(v0_pt, v0_pt)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # (a) Factorization
    ax = axes[0]
    for i, ptb in enumerate(pt_b_values):
        ib = np.argmin(np.abs(pt_bins - ptb))
        ax.plot(pt_bins, C_matrix[:, ib], color=colors[i], lw=2.5)
        ax.plot(pt_bins, C_fact[:, ib], '--', color=colors[i], lw=2)

    ax.set_xlabel(r'$p_T^a$ (GeV/$c$)', fontsize=18)
    ax.set_ylabel(r'$C(p_T^a, p_T^b)$', fontsize=18)
    ax.set_title(f'(a) Data vs Factorization ({centrality_label})', fontsize=16)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlim(0, 3)
    ax.set_ylim(-.02, .06)
    ax.grid(alpha=0.3)

    # (b) Simple model
    ax = axes[1]
    for i, ptb in enumerate(pt_b_values):
        ib = np.argmin(np.abs(pt_bins - ptb))
        ax.plot(pt_bins, C_matrix[:, ib], color=colors[i], lw=2.5)
        ax.plot(pt_bins, C_model[:, ib], '--', color=colors[i], lw=2)

    ax.set_xlabel(r'$p_T^a$ (GeV/$c$)', fontsize=18)
    ax.set_ylabel(r'$C(p_T^a, p_T^b)$', fontsize=18)
    ax.set_title(f'(b) Data vs Model ({centrality_label})', fontsize=16)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlim(0, 3)
    ax.set_ylim(-.02, .06)
    ax.grid(alpha=0.3)

    ax.text(
        0.97, 0.05,
        rf'$\langle p_T \rangle = {pT_mean:.3f}$ GeV/$c$',
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    return fig

def plot_fixed_N_covariance(corr_fixed_N, pt_bins, centrality_label):
    """
    Plot the fixed-multiplicity covariance
    << δ_n N(p^a) δ_n N(p^b) >>
    """

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)

    im = ax.imshow(
        corr_fixed_N,
        origin='lower',
        extent=[pt_bins[0], pt_bins[-1], pt_bins[0], pt_bins[-1]],
        aspect='auto'
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        r'$\langle \hat\delta \mathcal{N}(p^a)\hat\delta \mathcal{N}(p^b)\rangle$',
        fontsize=14
    )

    ax.set_xlabel(r'$p_T^a$ (GeV/$c$)', fontsize=16)
    ax.set_ylabel(r'$p_T^b$ (GeV/$c$)', fontsize=16)

    ax.set_title(
        rf'Fixed-$N$ covariance, {centrality_label}',
        fontsize=15
    )

    ax.tick_params(labelsize=13)
    plt.tight_layout()

    return fig
