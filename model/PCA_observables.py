### Author: OptimusThi
#!/usr/bin/env python3
"""
Perform PCA on model calculations data for all viscous corrections.
Prepares data for emulator training by reducing dimensionality.
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import math
from load_calculations import VISCOUS_CORRECTIONS, load_results

# PCA settings
N_COMPONENTS = None  # If None, determined by variance_threshold
VARIANCE_THRESHOLD = 0.99  # Capture 99% of variance

# Species to analyze 
SPECIES = 'charged'

# Centralities to include (None = all available)
CENTRALITIES_TO_INCLUDE = None  # Or specify list like [5, 6] for 50-60%, 60-70%
# Note: we can remove some centralities to improve emulation performance.

# Output folder
OUTPUT_FOLDER = 'pca_results'

