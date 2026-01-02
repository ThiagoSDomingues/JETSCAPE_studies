# Author: OptimusThi
#!/usr/bin/env python3
"""
Calculate integrated observables vs centrality for CMS predictions:
- Mean pT
- Global radial flow v_0 = sigma_[pT] / <[pT]>
- dN/dy for identified particles
- dN_ch/deta for charged hadrons
- dE_T/deta for transverse energy

For charged hadrons and all identified particles (pi, K, p, Sigma, Xi).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from calculations_file_format_single_event import return_result_dtype

# ================== Constants ==================
masslist = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])
particle_names = np.array(['pi', 'K', 'p', 'Sigma', 'Xi'])
particle_labels = [r'$\pi^{\pm}$', r'$K^{\pm}$', r'$p+\bar{p}$', r'$\Sigma$', r'$\Xi$']

ptcuts = np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.,10.])
ptlist = (ptcuts[1:]+ptcuts[:-1])/2
ptbinwidth = ptcuts[1:]-ptcuts[:-1]
Npt = len(ptlist)

# CMS kinematic cuts
etarange = [0.5, 2.4]
ptrange_integrated = [0.5, 2.0]
