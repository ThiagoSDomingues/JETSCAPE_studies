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
