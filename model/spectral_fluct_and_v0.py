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

import numpy as np

### Correlation function of deviations from the event averaged <dN/dpT> at fixed multiplicity

deltaNpT = dNdpT - np.average(dNdpT)
deltaN = N - np.average(N)
sigmaN = np.average(deltaN^2)

### calculate for deltaNpT a and b.

corr = np.average(deltaNpT * deltaNpT) - (np.average(deltaNpT * deltaN) * np.average(deltaNpT * deltaN))/ sigmaN^2

# two particle correlation function

# two_part_corr =  
