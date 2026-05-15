#!/usr/bin/env python3
"""
High-quality plotting utilities for v0(pT) analyses
===================================================

Features
--------
✓ Publication-quality plots
✓ Flexible API for Jupyter notebooks
✓ Multiple viscous correction comparisons
✓ Plot either:
    - v0(pT)
    - v0(pT)/v0
    - both
✓ Flexible centrality selection
✓ Species selection
✓ Design point filtering
✓ Mean curve overlay
✓ Standard deviation bands
✓ Custom color maps
✓ Log or linear axes
✓ Easy notebook usage

Example usage
-------------
from v0pt_plotter import V0PTPlotter

plotter = V0PTPlotter()

plotter.load_viscous_correction(
    "Grad",
    "design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_Grad.pkl"
)

fig, ax = plotter.plot(
    correction="Grad",
    centrality_index=6,
    species="charged",
    observable="ratio",
)

plt.show()
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler

# =============================================================================
# GLOBAL MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({

    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 500,
    "figure.figsize": (10, 7),

    # Fonts
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,

    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,

    # Axes
    "axes.linewidth": 1.8,
    "axes.grid": False,

    # Lines
    "lines.linewidth": 2.5,

    # Legend
    "legend.frameon": False,

    # Math
    "mathtext.fontset": "cm",

    # Better spacing
    "figure.autolayout": False,
})

# =============================================================================
# DEFAULT COLORS
# =============================================================================

DEFAULT_COLORS = {
    "Grad": "#1f77b4",
    "CE": "#d62728",
    "PTM": "#9467bd",
    "PTB": "#2ca02c",
}

# =============================================================================
# MAIN CLASS
# =============================================================================

class V0PTPlotter:

    def __init__(self):

        self.data = {}

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    def load_viscous_correction(self, name, filepath):
        """
        Load a viscous correction pickle file.

        Parameters
        ----------
        name : str
            Name of correction ('Grad', 'CE', etc.)

        filepath : str
            Path to pickle file
        """

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        with open(filepath, "rb") as f:
            self.data[name] = pickle.load(f)

        print(f"✓ Loaded {name}: {len(self.data[name])} design points")

    # =========================================================================
    # EXTRACT CURVES
    # =========================================================================

    def _extract_observable(
        self,
        correction,
        centrality_index,
        species,
        observable="v0pt",
    ):

        results = self.data[correction]

        curves = []

        for result in results:

            if centrality_index not in result["centrality_data"]:
                continue

            cent_data = result["centrality_data"][centrality_index]

            if species not in cent_data:
                continue

            sp = cent_data[species]

            pT = np.array(sp["pt_centers"])
            v0pt = np.array(sp["v0pT"])
            v0_global = sp["v0_global"]

            if observable == "v0pt":
                y = v0pt

            elif observable == "ratio":

                if v0_global != 0:
                    y = v0pt / v0_global
                else:
                    y = np.zeros_like(v0pt)

            else:
                raise ValueError(
                    "observable must be 'v0pt' or 'ratio'"
                )

            curves.append((pT, y))

        return curves

    # =========================================================================
    # MAIN PLOTTING FUNCTION
    # =========================================================================

    def plot(
        self,
        correction,
        centrality_index,
        species="charged",
        observable="v0pt",
        show_mean=True,
        show_std=True,
        show_all_design_points=True,
        alpha_design=0.15,
        figsize=(10, 7),
        xscale="log",
        xlim=(0.2, 4.0),
        ylim=None,
        title=None,
        xlabel=r"$p_T$ [GeV/$c$]",
        savepath=None,
        color=None,
    ):
        """
        Main plotting routine.

        Parameters
        ----------
        correction : str

        centrality_index : int

        species : str

        observable : str
            'v0pt' or 'ratio'

        show_mean : bool

        show_std : bool

        show_all_design_points : bool

        alpha_design : float

        figsize : tuple

        xscale : str
            'log' or 'linear'

        savepath : str or None
        """

        curves = self._extract_observable(
            correction,
            centrality_index,
            species,
            observable,
        )

        if len(curves) == 0:
            raise RuntimeError("No curves found.")

        if color is None:
            color = DEFAULT_COLORS.get(correction, "black")

        fig, ax = plt.subplots(figsize=figsize)

        # ==============================================================
        # PLOT ALL DESIGN POINTS
        # ==============================================================

        if show_all_design_points:

            for pT, y in curves:

                ax.plot(
                    pT,
                    y,
                    color=color,
                    alpha=alpha_design,
                    linewidth=1.0,
                )

        # ==============================================================
        # COMPUTE MEAN/STANDARD DEVIATION
        # ==============================================================

        all_y = np.array([curve[1] for curve in curves])

        mean_y = np.mean(all_y, axis=0)
        std_y = np.std(all_y, axis=0)

        pT = curves[0][0]

        # ==============================================================
        # STANDARD DEVIATION BAND
        # ==============================================================

        if show_std:

            ax.fill_between(
                pT,
                mean_y - std_y,
                mean_y + std_y,
                color=color,
                alpha=0.25,
                label=r"$1\sigma$",
            )

        # ==============================================================
        # MEAN CURVE
        # ==============================================================

        if show_mean:

            ax.plot(
                pT,
                mean_y,
                color=color,
                linewidth=4,
                label=f"{correction} mean",
            )

        # ==============================================================
        # AXES CONFIGURATION
        # ==============================================================

        ax.set_xscale(xscale)

        ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xlabel(xlabel)

        if observable == "v0pt":
            ax.set_ylabel(r"$v_0(p_T)$")

        elif observable == "ratio":
            ax.set_ylabel(r"$v_0(p_T)/v_0$")

        # ==============================================================
        # GRID
        # ==============================================================

        ax.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.7,
            alpha=0.4,
        )

        ax.grid(
            True,
            which="minor",
            linestyle=":",
            linewidth=0.5,
            alpha=0.25,
        )

        # ==============================================================
        # MINOR TICKS
        # ==============================================================

        ax.minorticks_on()

        # ==============================================================
        # TITLE
        # ==============================================================

        if title is None:

            title = (
                f"{correction} viscous correction\n"
                f"{species} | centrality index = {centrality_index}"
            )

        ax.set_title(title, pad=18)

        # ==============================================================
        # LEGEND
        # ==============================================================

        ax.legend()

        # ==============================================================
        # TIGHT LAYOUT
        # ==============================================================

        plt.tight_layout()

        # ==============================================================
        # SAVE
        # ==============================================================

        if savepath is not None:

            plt.savefig(
                savepath,
                bbox_inches="tight",
            )

            print(f"✓ Saved: {savepath}")

        return fig, ax

    # =========================================================================
    # COMPARISON PLOT
    # =========================================================================

    def compare_corrections(
        self,
        corrections,
        centrality_index,
        species="charged",
        observable="v0pt",
        figsize=(10, 7),
        xscale="log",
        show_std=False,
        savepath=None,
    ):

        fig, ax = plt.subplots(figsize=figsize)

        for correction in corrections:

            curves = self._extract_observable(
                correction,
                centrality_index,
                species,
                observable,
            )

            all_y = np.array([curve[1] for curve in curves])

            mean_y = np.mean(all_y, axis=0)
            std_y = np.std(all_y, axis=0)

            pT = curves[0][0]

            color = DEFAULT_COLORS.get(correction, None)

            ax.plot(
                pT,
                mean_y,
                linewidth=4,
                label=correction,
                color=color,
            )

            if show_std:

                ax.fill_between(
                    pT,
                    mean_y - std_y,
                    mean_y + std_y,
                    alpha=0.2,
                    color=color,
                )

        ax.set_xscale(xscale)

        ax.set_xlim(0.2, 4.0)

        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)

        ax.minorticks_on()

        ax.set_xlabel(r"$p_T$ [GeV/$c$]")

        if observable == "v0pt":
            ax.set_ylabel(r"$v_0(p_T)$")

        elif observable == "ratio":
            ax.set_ylabel(r"$v_0(p_T)/v_0$")

        ax.set_title(
            f"Comparison of viscous corrections\n"
            f"{species} | centrality index = {centrality_index}",
            pad=18,
        )

        ax.legend()

        plt.tight_layout()

        if savepath is not None:

            plt.savefig(
                savepath,
                bbox_inches="tight",
            )

            print(f"✓ Saved: {savepath}")

        return fig, ax

# =============================================================================
# EXAMPLE NOTEBOOK USAGE
# =============================================================================

if __name__ == "__main__":

    plotter = V0PTPlotter()

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    plotter.load_viscous_correction(
        "Grad",
        "design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_Grad.pkl"
    )

    plotter.load_viscous_correction(
        "CE",
        "design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_CE.pkl"
    )

    # ==========================================================
    # SINGLE PLOT
    # ==========================================================

    fig, ax = plotter.plot(
        correction="Grad",
        centrality_index=6,
        species="charged",
        observable="ratio",
        show_std=True,
        ylim=(-4, 14),
        savepath="high_quality_ratio_plot.png",
    )

    plt.show()

    # ==========================================================
    # COMPARISON PLOT
    # ==========================================================

    fig, ax = plotter.compare_corrections(
        corrections=["Grad", "CE"],
        centrality_index=6,
        species="charged",
        observable="ratio",
        show_std=True,
        savepath="comparison_plot.png",
    )

    plt.show()
