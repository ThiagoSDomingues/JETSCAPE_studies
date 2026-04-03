#!/usr/bin/env python3
"""
Calculate differential anisotropic flow v_n{2}(pT) and v_n{4}(pT) 
using scalar product method for all design points.

Based on formalism from arXiv:2005.14682 Section on Flow Analysis.
Uses Q-vectors from Qn data structure.
"""
import numpy as np
import os
import pickle
import multiprocessing as mp
from functools import partial

# --- Configuration ---
base_directory = '/data/js-sims-bayes/src/Qns/'

particle_names = np.array(['pi', 'K', 'p', 'Sigma', 'Xi'])
masslist = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])

# Original fine pT cuts (do not change; used for input data)
fine_ptcuts = np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.,10.])
Nfine_pt = len(fine_ptcuts) - 1

# New wider pT cuts (adjust as needed; wider at high pT to reduce noise)
new_ptcuts = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0])
ptlist = (new_ptcuts[1:] + new_ptcuts[:-1]) / 2
Npt = len(ptlist)

# Precompute bin indices for rebinnig: for each new bin, list of fine ipt indices it covers
bin_indices = []
for i in range(len(new_ptcuts) - 1):
    start_pt, end_pt = new_ptcuts[i], new_ptcuts[i+1]
    start_idx = np.searchsorted(fine_ptcuts, start_pt, side='left') - 1  # Inclusive start
    end_idx = np.searchsorted(fine_ptcuts, end_pt, side='left')  # Exclusive end
    bin_indices.append(list(range(max(0, start_idx), end_idx)))

delta_f_models = 4
max_n_events = 2500  # Increase this (e.g., to 5000) if you have more events available for better stats
Ndp = 500

centbins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90]])
ncentbins = len(centbins)

SPECIES_FLOW = {"pi": 0, "kaon": 1, "proton": 2}
HARMONICS = [2, 3, 4]

# --- Helper functions ---
def read_design_point(base_directory, dp):
    qn_file = os.path.join(base_directory, f'Qns_{dp}.npy')
    nsamples_file = os.path.join(base_directory, f'Nsamples_{dp}.npy')
    Qn = np.load(qn_file)
    Nsamples = np.load(nsamples_file)
    return Qn, Nsamples

def sort_by_Nch(Qn_norm):
    """Sort events by charged multiplicity"""
    if Qn_norm.ndim == 4:
        charged_mult = 2.0 * np.sum(Qn_norm[:, :3, 0, :], axis=(1,2))
    else:
        charged_mult = 2.0 * np.sum(Qn_norm[:, :3, :], axis=(1,2))
    return np.argsort(charged_mult)[::-1]

def select_centrality(eventlist, centbins):
    nev = len(eventlist)
    groups = []
    for lo, hi in centbins:
        i0 = int(np.floor(lo/100.0 * nev))
        i1 = int(np.floor(hi/100.0 * nev))
        groups.append(eventlist[i0:i1])
    return groups

# --- Flow calculation functions following arXiv:2005.14682 ---

def compute_C_n_2(Q_n, N):
    """
    Compute 2-particle cumulant C_n{2}.
    
    C_n{2} = Re{<Q_n Q_n^* - N>} / <N(N-1)>
    
    Parameters:
    -----------
    Q_n : complex array (n_events,)
        Flow vectors for harmonic n
    N : array (n_events,)
        Multiplicities
    
    Returns:
    --------
    C_n_2 : float
        2-particle cumulant
    """
    # Compute per-event correlation
    corr = np.real(Q_n * np.conj(Q_n) - N)
    weights = N * (N - 1)
    
    # Avoid division by zero
    mask = weights > 0
    if not np.any(mask):
        return 0.0
    
    C_n_2 = np.sum(corr[mask]) / np.sum(weights[mask])
    return C_n_2

def compute_C_n_4(Q_n, Q_2n, N, C_n_2):
    """
    Compute 4-particle cumulant C_n{4}.
    
    C_n{4} = <4> / <N(N-1)(N-2)(N-3)> - 2(C_n{2})^2
    
    where <4> includes self-correlation subtractions
    
    Parameters:
    -----------
    Q_n : complex array (n_events,)
        Flow vectors for harmonic n
    Q_2n : complex array (n_events,)
        Flow vectors for harmonic 2n
    N : array (n_events,)
        Multiplicities
    C_n_2 : float
        Pre-computed 2-particle cumulant
    
    Returns:
    --------
    C_n_4 : float
        4-particle cumulant
    """
    # Compute <4> with self-correlations subtracted
    term1 = (Q_n * np.conj(Q_n))**2
    term2 = -2.0 * np.real(Q_2n * np.conj(Q_n) * np.conj(Q_n))
    term3 = -4.0 * (N - 2) * (Q_n * np.conj(Q_n))
    term4 = Q_2n * np.conj(Q_2n)
    term5 = 2.0 * N * (N - 3)
    
    four_corr = np.real(term1) + term2 + np.real(term3) + np.real(term4) + term5
    weights = N * (N - 1) * (N - 2) * (N - 3)
    
    # Avoid division by zero
    mask = weights > 0
    if not np.any(mask):
        return 0.0
    
    avg_four = np.sum(four_corr[mask]) / np.sum(weights[mask])
    C_n_4 = avg_four - 2.0 * C_n_2**2
    
    return C_n_4

def compute_vn_2_pt(Q_n_poi_pt, Q_n_ref, Q0_poi_pt, N_ref, C_n_2_ref):
    """
    Compute differential flow v_n{2}(pT) using scalar product method.
    
    v_n{2}(pT) = Re{<Q_n^POI(pT) (Q_n^ref)^*>} / (<Q0^POI(pT) N^ref> sqrt(C_n^ref{2}))
    
    Parameters:
    -----------
    Q_n_poi_pt : complex array (n_events,)
        POI flow vector for this pT bin
    Q_n_ref : complex array (n_events,)
        Reference flow vector
    Q0_poi_pt : array (n_events,)
        POI multiplicity in this pT bin
    N_ref : array (n_events,)
        Reference multiplicity
    C_n_2_ref : float
        Reference 2-particle cumulant
    
    Returns:
    --------
    vn_2_pt : float
        Differential flow v_n{2}(pT)
    """
    # Scalar product
    numerator = np.real(Q_n_poi_pt * np.conj(Q_n_ref))
    denominator = Q0_poi_pt * N_ref
    
    # Avoid division by zero
    mask = denominator > 0
    if not np.any(mask):
        return 0.0
    
    avg_num = np.mean(numerator[mask])
    avg_denom = np.mean(denominator[mask])
    
    if C_n_2_ref <= 0 or avg_denom == 0:
        return 0.0
    
    vn_2_pt = avg_num / (avg_denom * np.sqrt(C_n_2_ref))
    
    return vn_2_pt

def compute_vn_4_pt(Q_n_poi_pt, Q_n_ref, Q_2n_ref, Q0_poi_pt, N_ref, C_n_2_ref, C_n_4_ref, vn_2_pt):
    """
    Compute differential flow v_n{4}(pT).
    
    v_n{4}(pT) = -d_n{4}(pT) / (-C_n^ref{4})^(3/4)
    
    where d_n{4}(pT) = <4>(pT) / <Q0^POI(pT) N^ref (N^ref-1) (N^ref-2)> 
                       - 2 v_n{2}(pT) (C_n^ref{2})^(3/2)
    
    Parameters:
    -----------
    Q_n_poi_pt : complex array (n_events,)
        POI flow vector for this pT bin
    Q_n_ref : complex array (n_events,)
        Reference flow vector
    Q_2n_ref : complex array (n_events,)
        Reference flow vector for 2n harmonic
    Q0_poi_pt : array (n_events,)
        POI multiplicity in this pT bin
    N_ref : array (n_events,)
        Reference multiplicity
    C_n_2_ref : float
        Reference 2-particle cumulant
    C_n_4_ref : float
        Reference 4-particle cumulant
    vn_2_pt : float
        Pre-computed v_n{2}(pT)
    
    Returns:
    --------
    vn_4_pt : float
        Differential flow v_n{4}(pT)
    """
    if C_n_4_ref >= 0:  # Need negative cumulant for real v_n{4}
        return 0.0
    
    # Compute <4>(pT) with self-correlations subtracted
    term1 = np.real(Q_n_poi_pt * Q_n_ref * np.conj(Q_n_ref) * np.conj(Q_n_ref))
    term2 = -2.0 * (N_ref - 1) * np.real(Q_n_poi_pt * np.conj(Q_n_ref))
    term3 = -np.real(Q_n_poi_pt * Q_n_ref * np.conj(Q_2n_ref))
    
    four_corr_pt = term1 + term2 + term3
    weights = Q0_poi_pt * N_ref * (N_ref - 1) * (N_ref - 2)
    
    # Avoid division by zero
    mask = weights > 0
    if not np.any(mask):
        return 0.0
    
    avg_four_pt = np.sum(four_corr_pt[mask]) / np.sum(weights[mask])
    
    # Compute d_n{4}(pT)
    d_n_4_pt = avg_four_pt - 2.0 * vn_2_pt * (C_n_2_ref)**(3.0/2.0)
    
    # Compute v_n{4}(pT)
    vn_4_pt = -d_n_4_pt / ((-C_n_4_ref)**(3.0/4.0))
    
    return vn_4_pt

# --- Main computation function ---
def compute_flow_design_point(idp, design_point, delta_f=0, harmonics=HARMONICS):
    """
    Compute differential flow v_n{2}(pT) and v_n{4}(pT) for a single design point.
    """
    print(f'Processing flow for design point {design_point}, which is {idp}')
    
    try:
        Qn_all, Nsamples_all = read_design_point(base_directory, design_point)
        
        nsamples = Nsamples_all[delta_f]
        nsamples = nsamples.astype(float)
        nsamples[nsamples == 0] = np.inf
        
        # Normalize Q-vectors
        # Qn_all shape: (delta_f, events, particles, harmonics, fine_pT)
        Qn_norm = Qn_all[delta_f] / nsamples[:, None, None, None]
        
        # Sort by charged multiplicity
        eventlist = sort_by_Nch(Qn_norm)
        centrality_events = select_centrality(eventlist, centbins)
        
        results = {
            'design_point': design_point,
            'delta_f': delta_f,
            'centrality_data': {}
        }
        
        # Loop over centrality bins
        for centrality, events_in_bin in enumerate(centrality_events):
            if len(events_in_bin) == 0:
                continue
            
            Qn_cent = Qn_norm[events_in_bin]
            cent_results = {}
            
            # Loop over harmonics
            for harmonic in harmonics:
                if harmonic >= Qn_cent.shape[2]:  # Check harmonic availability
                    continue
                
                # Build reference flow vectors (all charged particles, all pT) - unchanged, uses all fine bins
                Q_n_ref = np.zeros(len(events_in_bin), dtype=complex)
                Q_2n_ref = np.zeros(len(events_in_bin), dtype=complex)
                N_ref = np.zeros(len(events_in_bin))
                
                for iev in range(len(events_in_bin)):
                    # Sum charged particles (pi, K, p) * 2 for +/-, all pT bins
                    for pid in [0, 1, 2]:
                        Q_n_ref[iev] += 2.0 * np.sum(Qn_cent[iev, pid, harmonic, :])
                        N_ref[iev] += 2.0 * np.sum(np.abs(Qn_cent[iev, pid, 0, :]))
                        
                        # For 2n harmonic
                        if 2*harmonic < Qn_cent.shape[2]:
                            Q_2n_ref[iev] += 2.0 * np.sum(Qn_cent[iev, pid, 2*harmonic, :])
                
                # Compute reference cumulants
                C_n_2_ref = compute_C_n_2(Q_n_ref, N_ref)
                C_n_4_ref = compute_C_n_4(Q_n_ref, Q_2n_ref, N_ref, C_n_2_ref)
                
                # Reference flow
                v_n_2_ref = np.sqrt(C_n_2_ref) if C_n_2_ref > 0 else 0.0
                v_n_4_ref = np.power(-C_n_4_ref, 0.25) if C_n_4_ref < 0 else 0.0
                
                cent_results[f'v{harmonic}_ref'] = {
                    'v_n_2': v_n_2_ref,
                    'v_n_4': v_n_4_ref,
                    'C_n_2': C_n_2_ref,
                    'C_n_4': C_n_4_ref
                }
                
                # Compute differential flow for each species
                for species_name, pid in SPECIES_FLOW.items():
                    vn_2_pt_array = np.zeros(Npt)
                    vn_4_pt_array = np.zeros(Npt)
                    
                    # Loop over NEW pT bins
                    for ipt_new in range(Npt):
                        fine_ipts = bin_indices[ipt_new]
                        if not fine_ipts:
                            continue
                        
                        # Sum over fine bins for this new bin (factor of 2 for +/-)
                        Q_n_poi_pt = 2.0 * np.sum(Qn_cent[:, pid, harmonic, fine_ipts], axis=1)
                        Q0_poi_pt = 2.0 * np.sum(np.abs(Qn_cent[:, pid, 0, fine_ipts]), axis=1)
                        
                        # Compute v_n{2}(pT)
                        vn_2_pt = compute_vn_2_pt(Q_n_poi_pt, Q_n_ref, Q0_poi_pt, N_ref, C_n_2_ref)
                        vn_2_pt_array[ipt_new] = vn_2_pt
                        
                        # Compute v_n{4}(pT)
                        vn_4_pt = compute_vn_4_pt(Q_n_poi_pt, Q_n_ref, Q_2n_ref, 
                                                 Q0_poi_pt, N_ref, C_n_2_ref, C_n_4_ref, vn_2_pt)
                        vn_4_pt_array[ipt_new] = vn_4_pt
                    
                    cent_results[f'v{harmonic}_{species_name}'] = {
                        'vn_2_pt': vn_2_pt_array,
                        'vn_4_pt': vn_4_pt_array,
                        'pt': ptlist.copy()
                    }
            
            results['centrality_data'][centrality] = cent_results
        
        return results
        
    except Exception as e:
        print(f"Error processing design point {design_point}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_flow_wrapper(args, delta_f=0, harmonics=HARMONICS):
    idp, design_point = args
    return compute_flow_design_point(idp, design_point, delta_f, harmonics)

def compute_all_design_points_flow(design_points_list, delta_f=0, harmonics=HARMONICS, njobs=None):
    """
    Compute flow for all design points in parallel.
    """
    if njobs is None:
        njobs = min(48, mp.cpu_count() * 2, len(design_points_list))
    
    print(f'Computing flow for {len(design_points_list)} design points with {njobs} processes')
    
    compute_func = partial(compute_flow_wrapper, delta_f=delta_f, harmonics=harmonics)
    
    if njobs == 1:
        results = []
        for idp, design_point in enumerate(design_points_list):
            result = compute_func((idp, design_point))
            results.append(result)
    else:
        with mp.Pool(processes=njobs) as pool:
            args = [(idp, design_point) for idp, design_point in enumerate(design_points_list)]
            results = pool.map(compute_func, args)
    
    results = [r for r in results if r is not None]
    
    print(f'Successfully computed flow for {len(results)} design points')
    
    return results

# --- Usage ---
if __name__ == "__main__":
    # Test with first 10 design points
    design_points = range(10)
    
    # Compute flow for all design points
    all_results = compute_all_design_points_flow(
        design_points,
        delta_f=0,  # Grad model
        harmonics=[2, 3, 4],
        njobs=48
    )
    
    # Save results
    output_file = 'flow_vn_pt_design_points_results_wider_bins.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    if len(all_results) > 0:
        sample = all_results[0]
        print(f"\nProcessed {len(all_results)} design points")
        if len(sample['centrality_data']) > 0:
            first_cent = list(sample['centrality_data'].values())[0]
            print(f"Observables computed: {list(first_cent.keys())}")
