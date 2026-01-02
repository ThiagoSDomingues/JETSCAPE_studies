# Author: OptimusThi
#!/usr/bin/env python3

"""
Functions to calculate integrated and pt-differential observables:
- pt-integrated observables:
  - Mean pT
  - Global radial flow v_0 = sigma_[pT] / <[pT]>
  - dN/dy for identified particles
  - dN_ch/deta for charged hadrons
  - dE_T/deta for transverse energy
- pt-differential observables:  
"""
import numpy as np

def compute_mean_pt_lowlevel(arg):
    """Compute mean pT."""
    Q0_diff = arg[:, :Npt].astype(float)  # [events, pT]
    weights = arg[:, Npt].astype(float)
    
    # Mean pT = sum(N_i * pT_i) / sum(N_i)
    event_mean_pt = np.sum(Q0_diff * ptlist, axis=1) / np.sum(Q0_diff, axis=1)
    mean_pt = np.average(event_mean_pt, weights=weights)
    return mean_pt

def compute_dNdy_lowlevel(arg):
    """Compute dN/dy (rapidity density)."""
    N_total = arg[:, 0].astype(float)
    weights = arg[:, 1].astype(float)
    
    # dN/dy = N / (Delta_y), where Delta_y = 2.0 (full rapidity range)
    dNdy = np.average(N_total, weights=weights) / 2.0
    return dNdy

def compute_dNdeta_lowlevel(arg):
    """Compute dN/deta (pseudorapidity density)."""
    N_total = arg[:, 0].astype(float)
    weights = arg[:, 1].astype(float)
    
    # dN/deta = N / Delta_eta
    delta_eta = etarange[1] - etarange[0]
    dNdeta = np.average(N_total, weights=weights) / delta_eta
    return dNdeta

def compute_dETdeta_lowlevel(arg):
    """Compute dE_T/deta (transverse energy density)."""
    Q0_diff = arg[:, :Npt].astype(float)  # [events, pT]
    mass = arg[:, Npt].astype(float)
    weights = arg[:, Npt+1].astype(float)
    
    # E_T = sqrt(m^2 + pT^2) for each particle
    # Sum over pT bins
    ET_per_event = np.zeros(len(Q0_diff))
    for i, pt in enumerate(ptlist):
        mT = np.sqrt(mass**2 + pt**2)
        ET_per_event += Q0_diff[:, i] * mT
    
    delta_eta = etarange[1] - etarange[0]
    dETdeta = np.average(ET_per_event, weights=weights) / delta_eta
    return dETdeta

def compute_v0_lowlevel(arg):
    """
    Compute global radial flow v_0 = sigma_[pT] / <[pT]>
    where [pT] is the event-wise mean pT.
    """
    Q0_diff = arg[:, :Npt].astype(float)
    weights = arg[:, Npt].astype(float)
    
    # Compute event-wise mean pT
    event_mean_pt = np.sum(Q0_diff * ptlist, axis=1) / np.sum(Q0_diff, axis=1)
    
    # Compute <[pT]> and sigma_[pT]
    mean_of_mean_pt = np.average(event_mean_pt, weights=weights)
    sigma_mean_pt = np.sqrt(np.average((event_mean_pt - mean_of_mean_pt)**2, weights=weights))
    
    # v_0 = sigma_[pT] / <[pT]>
    v0 = sigma_mean_pt / mean_of_mean_pt
    
    return v0

def compute_mean_pt(Qn_normalized, etarange, ptrange, weights, pid=None):
    """Compute mean pT with jackknife errors."""
    Qn_temp = cutQn(Qn_normalized, etarange)
    
    if pid is None:  # Charged hadrons
        Q0_diff = 2 * np.sum(Qn_temp[:, :3, :], axis=1)
    else:  # Identified particle
        Q0_diff = 2 * Qn_temp[:, pid, :]
    
    arg = np.c_[Q0_diff, weights]
    return jackknifeerror(compute_mean_pt_lowlevel, arg)

def compute_dNdy(Qn_normalized, weights, pid):
    """Compute dN/dy for identified particle with jackknife errors."""
    # Use rapidity range [-1, 1] (original data)
    N_total = 2 * np.sum(Qn_normalized[:, pid, :], axis=1)
    
    arg = np.c_[N_total, weights]
    return jackknifeerror(compute_dNdy_lowlevel, arg)

def compute_dNdeta(Qn_normalized, etarange, weights):
    """Compute dN_ch/deta for charged hadrons with jackknife errors."""
    Qn_temp = cutQn(Qn_normalized, etarange)
    N_ch = 2 * np.sum(Qn_temp[:, :3, :], axis=(1, 2))
    
    arg = np.c_[N_ch, weights]
    return jackknifeerror(compute_dNdeta_lowlevel, arg)

def compute_dETdeta(Qn_normalized, etarange, weights):
    """Compute dE_T/deta for charged hadrons with jackknife errors."""
    Qn_temp = cutQn(Qn_normalized, etarange)
    
    # Sum over charged hadrons (pi, K, p)
    Q0_diff_ch = 2 * np.sum(Qn_temp[:, :3, :], axis=1)
    
    # Use average mass for charged hadrons (dominated by pions)
    avg_mass = np.average(masslist[:3], weights=[1.0, 0.15, 0.05])  # Approximate ratios
    
    arg = np.c_[Q0_diff_ch, np.full(len(Q0_diff_ch), avg_mass), weights]
    return jackknifeerror(compute_dETdeta_lowlevel, arg)

def compute_v0(Qn_normalized, etarange, ptrange, weights, pid=None):
    """Compute global radial flow v_0 with jackknife errors."""
    Qn_temp = cutQn(Qn_normalized, etarange)
    
    if pid is None:  # Charged hadrons
        Q0_diff = 2 * np.sum(Qn_temp[:, :3, :], axis=1)
    else:  # Identified particle
        Q0_diff = 2 * Qn_temp[:, pid, :]
    
    arg = np.c_[Q0_diff, weights]
    return jackknifeerror(compute_v0_lowlevel, arg)
