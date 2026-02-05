### Author: OptimusThi
import numpy as np

### Correlation function of deviations from the event averaged <dN/dpT> at fixed multiplicity

deltaNpT = dNdpT - np.average(dNdpT)
deltaN = N - np.average(N)
sigmaN = np.average(deltaN^2)

### calculate for deltaNpT a and b.

corr = np.average(deltaNpT * deltaNpT) - (np.average(deltaNpT * deltaN) * np.average(deltaNpT * deltaN))/ sigmaN^2

# two particle correlation function

# two_part_corr =  
