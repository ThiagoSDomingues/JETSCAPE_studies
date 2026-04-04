#!/usr/bin/env python3
"""
Flexible calculation of double-scaling v0(pT) observable for all design points from Qn data.
Allows user-defined pT bins and kinematic ranges for both differential and integrated observables.
"""
import numpy as np
import os
import pickle
import multiprocessing as mp
from functools import partial

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# =============================================================================

# --- System selection ---
SYSTEM_CONFIG = {
    'PbPb_2760': {
        'base_directory': '/data/js-sims-bayes/src/Qns/',
        'max_n_events': 2500,
        'Ndp': 500,
        'label': 'Pb_Pb_2760'
    },
    'AuAu_200': {
        'base_directory': '/data/js-sims-bayes/src/QnsAu',
        'max_n_events': 2500,
        'Ndp': 500,
        'label': 'Au_Au_200'
    },
    'XeXe_5440': {
        'base_directory': '/data/js-sims-bayes/src/QnsXe',
        'max_n_events': 1600,
        'Ndp': 1000,
        'label': 'Xe_Xe_5440'
    }
}

# Select system
SYSTEM = 'PbPb_2760'
config = SYSTEM_CONFIG[SYSTEM]
base_directory = config['base_directory']
max_n_events = config['max_n_events']
Ndp = config['Ndp']
system_label = config['label']

# --- Viscous correction model ---
DELTA_F_MODELS = {
    'Grad': 0,
    'CE': 1,
    'PTM': 2,
    'PTB': 3
}
DELTA_F_MODEL = 'PTB'  # Choose: 'Grad', 'CE', 'PTM', 'PTB'
delta_f_index = DELTA_F_MODELS[DELTA_F_MODEL]

# --- Kinematic ranges ---
# Eta range for integrated observables (reference particles)
eta_range = [0.5, 2.4]  # Will use positive eta only (symmetry)

# pT range for integrated observables (global v0 calculation)
pt_range_integrated = [0.5, 2.0]

# pT bins for differential observables v0(pT)
# Example: CMS-like binning
#pt_bins_differential = np.array([
#    [0.5, 0.7],
#    [0.7, 0.9],
#    [0.9, 1.0],
#    [1.0, 1.5],
#    [1.5, 2.0],
#    [2.0, 3.0],
#    [3.0, 4.0],
#    [4.0, 5.0],
#    [5.0, 6.0],
#    [6.0, 7.0],
#    [7.0, 8.0],
#    [8.0, 9.0],
#    [9.0, 10.0]
#])

centers = np.array([
    0.55231, 0.64791, 0.74831, 0.84819, 0.94830,
    1.09310, 1.29330, 1.49340, 1.69350, 1.89360,
    2.09380, 2.29390, 2.49410, 2.69420, 2.89450,
    3.09460, 3.29490, 3.49510, 3.69530, 3.89540,
    4.22420, 4.72680, 5.23030, 5.73200, 6.23280,
    6.83950, 7.59340, 8.46080, 9.46140
])

# Compute bin edges
edges = np.zeros(len(centers) + 1)

# First edge (extrapolate)
edges[0] = centers[0] - 0.5*(centers[1] - centers[0])

# Middle edges (midpoints)
for i in range(1, len(centers)):
    edges[i] = 0.5*(centers[i-1] + centers[i])

# Last edge (extrapolate)
edges[-1] = centers[-1] + 0.5*(centers[-1] - centers[-2])

# Build final ATLAS bins: [low, high]
pt_bins_differential = np.column_stack([edges[:-1], edges[1:]])


# --- Particle species ---
# All available particles
particle_names = np.array(['pi', 'K', 'p', 'Sigma', 'Xi'])
masslist = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])

# Species to compute v0 for (use indices or names)
SPECIES_TO_COMPUTE = {
    'charged': None,  # Will be computed as 2*(pi + K + p)
    'pi': 0,
    'kaon': 1,
    'proton': 2,
    'Sigma': 3,
    'Xi': 4
}

# --- Simulation parameters ---
ptcuts = np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,
                   0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,
                   1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,
                   1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,
                   3.8,4.,10.])
ptlist = (ptcuts[1:] + ptcuts[:-1]) / 2
ptbinwidth = ptcuts[1:] - ptcuts[:-1]
Npt = len(ptlist)

delta_f_models = 4
n_harmonics = 6

# Centrality bins
centbins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],
                     [50,60],[60,70],[70,80],[80,90]])
ncentbins = len(centbins)

# --- Parallel processing ---
N_JOBS = 48  # Number of parallel processes

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_pt_mask(pt_range):
    """Get boolean mask for pT bins within specified range"""
    mask = (ptlist >= pt_range[0]) & (ptlist <= pt_range[1])
    return mask

def map_to_pt_bins(pt_bins, ptlist_source, counts):
    """
    Map counts from fine ptlist_source bins to coarser user-defined pt_bins.
    
    Parameters:
    -----------
    pt_bins : ndarray, shape (N_bins, 2)
        User-defined pT bins [[pt_low, pt_high], ...]
    ptlist_source : ndarray
        Center values of source pT bins
    counts : ndarray, shape (..., Npt_source)
        Counts in source pT bins (last dimension is pT)
    
    Returns:
    --------
    mapped_counts : ndarray, shape (..., N_bins)
        Counts summed into user-defined bins
    pt_centers : ndarray, shape (N_bins,)
        Center values of user-defined bins
    """
    N_bins = len(pt_bins)
    output_shape = counts.shape[:-1] + (N_bins,)
    mapped_counts = np.zeros(output_shape, dtype=counts.dtype)
    pt_centers = np.zeros(N_bins)
    
    for i, (pt_low, pt_high) in enumerate(pt_bins):
        # Find source bins that overlap with this target bin
        mask = (ptlist_source >= pt_low) & (ptlist_source < pt_high)
        mapped_counts[..., i] = np.sum(counts[..., mask], axis=-1)
        pt_centers[i] = (pt_low + pt_high) / 2
    
    return mapped_counts, pt_centers

def read_design_point(base_directory, dp):
    """Read Qn and Nsamples files for given design point"""
    qn_file = os.path.join(base_directory, f'Qns_{dp}.npy')
    nsamples_file = os.path.join(base_directory, f'Nsamples_{dp}.npy')
    
    Qn = np.load(qn_file)
    Nsamples = np.load(nsamples_file)
    
    return Qn, Nsamples

def sort_by_Nch(Qn_norm, pt_mask):
    """Sort events by charged multiplicity (descending) in specified pT range"""
    # Sum over pi, K, p (indices 0,1,2) for charged particles, multiply by 2 for +/-
    charged_mult = 2.0 * np.sum(Qn_norm[:, :3, 0, :][:, :, pt_mask], axis=(1,2))
    eventlist = np.argsort(charged_mult)[::-1]  # Descending order
    return eventlist

def select_centrality(eventlist, centbins):
    """Group events by centrality percentiles"""
    nev = len(eventlist)
    groups = []
    for lo, hi in centbins:
        i0 = int(np.floor(lo/100.0 * nev))
        i1 = int(np.floor(hi/100.0 * nev))
        groups.append(eventlist[i0:i1])
    return groups

def jackknife_error_vector(function, arg):
    """Compute jackknife error for vector-valued functions"""
    val = function(arg)
    nev = arg.shape[0]
    if nev <= 1:
        return val, np.zeros_like(val)
    
    jk = np.array([function(np.delete(arg, i, axis=0)) for i in range(nev)])
    err = np.sqrt(nev - 1) * np.std(jk, axis=0)
    corr = nev * val - (nev - 1) * np.mean(jk, axis=0)
    return corr, err

def jackknife_error_scalar(function, arg):
    """Compute jackknife error for scalar-valued functions"""
    val = function(arg)
    nev = arg.shape[0]
    if nev <= 1:
        return val, 0.0
    
    jk = np.array([function(np.delete(arg, i, axis=0)) for i in range(nev)])
    err = np.sqrt(nev - 1) * np.std(jk)
    corr = nev * val - (nev - 1) * np.mean(jk)
    return corr, err

def _v0pT_lowlevel(arg):
    """Core v0(pT) calculation"""
    Npt_local = arg.shape[1] - 3
    Q0poi = arg[:, :Npt_local].astype(float)          # (nev, Npt)
    Nref = arg[:, Npt_local].astype(float)            # (nev,)
    meanPTref = arg[:, Npt_local+1].astype(float)     # (nev,)
    w = arg[:, Npt_local+2].astype(float)             # (nev,)

    deltaNref = Nref - np.average(Nref, weights=w)
    deltaPT = meanPTref - np.average(meanPTref, weights=w)
    delta_poi = Q0poi - np.average(Q0poi, weights=w, axis=0)

    sigmaNsq = np.average(deltaNref**2, weights=w)
    if sigmaNsq == 0:
        return np.zeros(Npt_local)

    delta_hat_PT = deltaPT - (np.average(deltaPT * deltaNref, weights=w) / sigmaNsq) * deltaNref
    deltaN_col = deltaNref[:, None]
    deltahat_poi_SP = delta_poi - (np.average(delta_poi * deltaN_col, weights=w, axis=0) / sigmaNsq) * deltaN_col

    sigmahatPT = np.sqrt(np.average(delta_hat_PT**2, weights=w))
    if sigmahatPT == 0:
        return np.zeros(Npt_local)

    cov = np.average(deltahat_poi_SP * delta_hat_PT[:, None], weights=w, axis=0)
    mean_ref = np.average(Q0poi, weights=w, axis=0)
    out = np.zeros_like(mean_ref)
    nz = (mean_ref != 0)
    out[nz] = cov[nz] / (sigmahatPT * mean_ref[nz])
    return out

def compute_v0pT_and_err(diff_counts, ref_counts, pt_centers, weights=None):
    """Compute v0(pT) with jackknife errors"""
    nev, Npt_local = diff_counts.shape
    if weights is None:
        w = np.ones(nev, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    
    # Integrated reference count per event
    Nref = np.sum(ref_counts, axis=1)
    
    # Mean pT of reference per event
    meanPTref = np.zeros_like(Nref)
    mask = (Nref > 0)
    if np.any(mask):
        meanPTref[mask] = np.sum(ref_counts[mask] * pt_centers[None, :], axis=1) / Nref[mask]
    
    # Build argument matrix
    arg = np.empty((nev, Npt_local + 3), dtype=float)
    arg[:, :Npt_local] = diff_counts
    arg[:, Npt_local] = Nref
    arg[:, Npt_local+1] = meanPTref
    arg[:, Npt_local+2] = w
    
    return jackknife_error_vector(_v0pT_lowlevel, arg)

def event_mean_pt_from_counts(counts, pt_centers):
    """Compute per-event mean pT from counts"""
    totals = np.sum(counts, axis=1)
    mean_pt = np.zeros_like(totals, dtype=float)
    mask = totals > 0
    if np.any(mask):
        mean_pt[mask] = np.sum(counts[mask] * pt_centers[None, :], axis=1) / totals[mask]
    return mean_pt, totals

def _v0_global_lowlevel(mean_pt_events):
    """Core global v0 calculation for jackknife"""
    mean_ensemble = np.mean(mean_pt_events)
    sigma = np.std(mean_pt_events, ddof=0)
    if mean_ensemble == 0:
        return 0.0
    return sigma / mean_ensemble

def v0_global_from_mean_pts(mean_pt_events, weights=None):
    """Compute global v0 = sigma([pT]) / <[pT]> with jackknife error"""
    if weights is not None:
        w = np.asarray(weights)
        mean_ensemble = np.average(mean_pt_events, weights=w)
        var = np.average((mean_pt_events - mean_ensemble)**2, weights=w)
        sigma = np.sqrt(var)
    else:
        mean_ensemble = np.mean(mean_pt_events)
        sigma = np.std(mean_pt_events, ddof=0)
    
    if mean_ensemble == 0:
        return 0.0, 0.0, mean_ensemble, sigma
    
    v0_val = sigma / mean_ensemble
    
    # Jackknife error
    v0_corr, v0_err = jackknife_error_scalar(_v0_global_lowlevel, mean_pt_events)
    
    return v0_val, v0_err, mean_ensemble, sigma

# =============================================================================
# MAIN COMPUTATION FUNCTION
# =============================================================================

def compute_v0_design_point(idp, design_point, delta_f, pt_mask_integrated, 
                           pt_bins_diff, pt_range_integ):
    """
    Compute v0(pT) double-scaling observable for a single design point.
    """
    print(f'Processing design point {design_point}, which is {idp} of {Ndp-1}')
    
    try:
        # Read data
        Qn_all, Nsamples_all = read_design_point(base_directory, design_point)
        
        # Extract dimensions and validate
        (Ndelta_f, Nevents, Nparticles, Nharmonics, Npt_data) = Qn_all.shape
        if (Ndelta_f != delta_f_models or Nevents > max_n_events or 
            Nparticles != len(masslist) or Nharmonics != n_harmonics or Npt_data != Npt):
            raise ValueError(f'Unexpected dimensions: {Qn_all.shape}')
        
        nsamples = Nsamples_all[delta_f]
        
        # Normalize by number of samples
        Qn_norm = Qn_all[delta_f] / nsamples[:, None, None, None]
        
        # Sort by charged multiplicity in the integrated pT range
        eventlist = sort_by_Nch(Qn_norm, pt_mask_integrated)
        
        # Group by centrality
        centrality_events = select_centrality(eventlist, centbins)
        
        # Initialize results dictionary
        results = {
            'design_point': design_point,
            'delta_f': delta_f,
            'pt_bins_differential': pt_bins_diff,
            'pt_range_integrated': pt_range_integ,
            'centrality_data': {}
        }
        
        # Compute for each centrality bin
        for centrality, events_in_bin in enumerate(centrality_events):
            if len(events_in_bin) == 0:
                continue
                
            cent_label = f"{centbins[centrality,0]:.0f}-{centbins[centrality,1]:.0f}%"
            
            # Extract normalized counts for this centrality
            Qn_cent = Qn_norm[events_in_bin]  # (nev_cent, Nparticles, Nharmonics, Npt)
            
            # Get Q0 (multiplicity, harmonic=0) for all particles in fine bins
            Q0_fine = Qn_cent[:, :, 0, :]  # (nev_cent, Nparticles, Npt)
            
            # Map to differential bins
            Q0_diff, pt_centers_diff = map_to_pt_bins(pt_bins_diff, ptlist, Q0_fine)
            
            # Extract for integrated range
            Q0_integ = Q0_fine[:, :, pt_mask_integrated]  # (nev_cent, Nparticles, Npt_integ)
            pt_centers_integ = ptlist[pt_mask_integrated]
            
            # Build charged reference (factor 2 for +/- particles)
            q0_charged_diff = 2.0 * np.sum(Q0_diff[:, :3, :], axis=1)  # (nev, Npt_diff)
            q0_charged_integ = 2.0 * np.sum(Q0_integ[:, :3, :], axis=1)  # (nev, Npt_integ)
            
            # Compute v0(pT) for each species
            cent_results = {}
            
            for species_name, pid in SPECIES_TO_COMPUTE.items():
                if species_name == 'charged':
                    # Charged hadrons
                    diff_counts = q0_charged_diff
                    integ_counts = q0_charged_integ
                else:
                    # Individual species (factor 2 for +/-)
                    diff_counts = 2.0 * Q0_diff[:, pid, :]
                    integ_counts = 2.0 * Q0_integ[:, pid, :]
                
                ref_counts_diff = q0_charged_diff
                ref_counts_integ = q0_charged_integ
                
                # Compute v0(pT) and errors in differential bins
                v0pT_mean, v0pT_err = compute_v0pT_and_err(
                    diff_counts, ref_counts_diff, pt_centers_diff
                )
                
                # Compute per-event mean pT in integrated range
                meanpt_events, totals = event_mean_pt_from_counts(
                    integ_counts, pt_centers_integ
                )
                
                # Global v0 and ensemble mean pT with errors
                v0_global_val, v0_global_err, meanpt_ensemble, sigma_pt = \
                    v0_global_from_mean_pts(meanpt_events)
                
                # Double scaling variables
                x = pt_centers_diff / (meanpt_ensemble if meanpt_ensemble != 0 else 1.0)
                y = v0pT_mean / (v0_global_val if v0_global_val != 0 else 1.0)
                yerr = v0pT_err / (v0_global_val if v0_global_val != 0 else 1.0)
                
                cent_results[species_name] = {
                    'v0pT': v0pT_mean,
                    'v0pT_err': v0pT_err,
                    'v0_global': v0_global_val,
                    'v0_global_err': v0_global_err,
                    'mean_pt_ensemble': meanpt_ensemble,
                    'x_scaled': x,
                    'y_scaled': y,
                    'y_scaled_err': yerr,
                    'pt_centers': pt_centers_diff,
                    'n_events': len(events_in_bin)
                }
            
            results['centrality_data'][centrality] = cent_results
        
        return results
        
    except Exception as e:
        print(f"Error processing design point {design_point}: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# PARALLEL EXECUTION
# =============================================================================

def compute_all_design_points_v0(design_points_list, delta_f, pt_mask_integ, 
                                 pt_bins_diff, pt_range_integ, njobs=None):
    """
    Compute v0(pT) for all design points in parallel.
    """
    if njobs is None:
        njobs = min(N_JOBS, mp.cpu_count(), len(design_points_list))
    
    print(f'Computing v0(pT) for {len(design_points_list)} design points')
    print(f'Using {njobs} parallel processes')
    print(f'Integrated pT range: {pt_range_integ}')
    print(f'Differential pT bins: {len(pt_bins_diff)} bins')
    
    # Create partial function
    compute_func = partial(
        compute_v0_design_point_wrapper,
        delta_f=delta_f,
        pt_mask_integrated=pt_mask_integ,
        pt_bins_diff=pt_bins_diff,
        pt_range_integ=pt_range_integ
    )
    
    # Parallel computation
    if njobs == 1:
        results = []
        for idp, design_point in enumerate(design_points_list):
            result = compute_func((idp, design_point))
            results.append(result)
    else:
        with mp.Pool(processes=njobs) as pool:
            args = [(idp, dp) for idp, dp in enumerate(design_points_list)]
            results = pool.map(compute_func, args)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    print(f'Successfully computed v0(pT) for {len(results)} design points')
    
    return results

def compute_v0_design_point_wrapper(args, delta_f, pt_mask_integrated, 
                                   pt_bins_diff, pt_range_integ):
    """Wrapper for multiprocessing"""
    idp, design_point = args
    return compute_v0_design_point(
        idp, design_point, delta_f, pt_mask_integrated, 
        pt_bins_diff, pt_range_integ
    )

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print(f"System: {SYSTEM}")
    print(f"Viscous correction: {DELTA_F_MODEL}")
    print(f"Number of design points: {Ndp}")
    print(f"pT range for integrated observables: {pt_range_integrated}")
    print(f"Number of differential pT bins: {len(pt_bins_differential)}")
    print(f"Species to compute: {list(SPECIES_TO_COMPUTE.keys())}")
    print("="*80)
    
    # Get pT mask for integrated observables
    pt_mask_integrated = get_pt_mask(pt_range_integrated)
    
    # Choose design points to process
    design_points = range(Ndp)  # All design points
    # design_points = range(10)  # For testing
    
    # Compute v0(pT) for all design points
    all_results = compute_all_design_points_v0(
        design_points,
        delta_f=delta_f_index,
        pt_mask_integ=pt_mask_integrated,
        pt_bins_diff=pt_bins_differential,
        pt_range_integ=pt_range_integrated,
        njobs=N_JOBS
    )
    
    # Generate output filename
    output_file = f'v0pt_design_points_results_{system_label}_{DELTA_F_MODEL}.pkl'
    
    # Save results
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Processed {len(all_results)} design points")
    if len(all_results) > 0:
        sample_result = all_results[0]
        print(f"  Centralities computed: {len(sample_result['centrality_data'])}")
        if len(sample_result['centrality_data']) > 0:
            first_cent = list(sample_result['centrality_data'].values())[0]
            print(f"  Species: {list(first_cent.keys())}")
            first_species = list(first_cent.values())[0]
            print(f"  pT bins: {len(first_species['pt_centers'])}")
            print(f"  Example v0_global: {first_species['v0_global']:.4f} ± {first_species['v0_global_err']:.4f}")
