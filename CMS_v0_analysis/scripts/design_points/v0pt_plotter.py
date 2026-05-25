#!/usr/bin/env python3
"""
High-quality plotting utilities for v0(pT) analyses
===================================================

NEW FEATURES
------------
✓ Overlay CMS and ATLAS experimental data
✓ Supports:
    - v0(pT)
    - v0(pT)/v0
    - global v0
✓ Easy loading of simulation + experimental data
✓ Cleaner API for viscous corrections
✓ Experimental uncertainty bands
✓ Multiple experiment overlays
✓ Centrality-aware global-v0 plotting
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# GLOBAL STYLE
# =============================================================================

plt.rcParams.update({

    "figure.dpi": 150,
    "savefig.dpi": 500,
    "figure.figsize": (10, 7),

    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    "axes.linewidth": 1.8,

    "lines.linewidth": 2.5,

    "legend.frameon": False,

    "mathtext.fontset": "cm",
})

# =============================================================================
# COLORS
# =============================================================================

DEFAULT_COLORS = {
    "Grad": "#1f77b4",
    "CE": "#d62728",
    "PTM": "#9467bd",
    "PTB": "#2ca02c",
}

EXPERIMENT_COLORS = {
    "CMS": "black",
    "ATLAS": "darkorange",
}

# =============================================================================
# MAIN CLASS
# =============================================================================

class V0PTPlotter:

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self):

        self.sim_data = {}
        self.exp_data = {}

    # =========================================================================
    # LOAD SIMULATION DATA
    # =========================================================================

    def load_simulation(
        self,
        correction,
        filepath,
    ):

        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            self.sim_data[correction] = pickle.load(f)

        print(f"✓ Loaded simulation: {correction}")

    # =========================================================================
    # LOAD EXPERIMENTAL DATA
    # =========================================================================

    def load_experiment(
        self,
        experiment_name,
        filepath,
    ):

        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            self.exp_data[experiment_name] = pickle.load(f)

        print(f"✓ Loaded experimental data: {experiment_name}")

    # =========================================================================
    # EXTRACT SIM CURVES
    # =========================================================================

    def _extract_simulation_curves(
        self,
        correction,
        centrality_index,
        species,
        observable,
    ):

        results = self.sim_data[correction]

        curves = []

        for result in results:

            if centrality_index not in result["centrality_data"]:
                continue

            cent = result["centrality_data"][centrality_index]

            if species not in cent:
                continue

            sp = cent[species]

            pT = np.array(sp["pt_centers"])
            v0pt = np.array(sp["v0pT"])
            v0 = sp["v0_global"]

            if observable == "v0pt":
                y = v0pt

            elif observable == "ratio":

                if v0 != 0:
                    y = v0pt / v0
                else:
                    y = np.zeros_like(v0pt)

            elif observable == "v0":
                y = np.full_like(pT, v0)

            else:
                raise ValueError("Unknown observable")

            curves.append((pT, y, v0))

        return curves

    # =========================================================================
    # OVERLAY EXPERIMENTAL DATA
    # =========================================================================

    def _plot_experimental_overlay(
        self,
        ax,
        experiment,
        observable,
        centrality_index,
        species,
    ):

        if experiment not in self.exp_data:
            return

        data = self.exp_data[experiment]

        if centrality_index not in data["centrality_data"]:
            return

        cent = data["centrality_data"][centrality_index]

        if species not in cent:
            return

        sp = cent[species]

        color = EXPERIMENT_COLORS.get(experiment, "black")

        # -------------------------------------------------------------
        # v0(pT)
        # -------------------------------------------------------------

        if observable == "v0pt":

            ax.errorbar(
                sp["pt_centers"],
                sp["v0pT"],
                yerr=sp.get("v0pT_err", None),
                fmt="o",
                color=color,
                label=f"{experiment} data",
                markersize=6,
                capsize=3,
            )

        # -------------------------------------------------------------
        # ratio
        # -------------------------------------------------------------

        elif observable == "ratio":

            ratio = np.array(sp["v0pT"]) / sp["v0_global"]

            ratio_err = None

            if "v0pT_err" in sp:

                ratio_err = (
                    np.array(sp["v0pT_err"])
                    / sp["v0_global"]
                )

            ax.errorbar(
                sp["pt_centers"],
                ratio,
                yerr=ratio_err,
                fmt="s",
                color=color,
                label=f"{experiment} data",
                markersize=6,
                capsize=3,
            )

    # =========================================================================
    # MAIN PLOT
    # =========================================================================

    def plot(
        self,
        correction,
        centrality_index,
        species="charged",
        observable="ratio",
        overlay_experiments=None,
        show_mean=True,
        show_std=True,
        show_design_points=True,
        alpha_design=0.10,
        figsize=(10, 7),
        xscale="log",
        xlim=(0.2, 4.0),
        ylim=None,
        color=None,
        title=None,
        savepath=None,
    ):

        curves = self._extract_simulation_curves(
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

        # ==========================================================
        # ALL DESIGN POINTS
        # ==========================================================

        if show_design_points:

            for pT, y, _ in curves:

                ax.plot(
                    pT,
                    y,
                    color=color,
                    alpha=alpha_design,
                    linewidth=1.0,
                )

        # ==========================================================
        # MEAN + STD
        # ==========================================================

        all_y = np.array([c[1] for c in curves])

        mean_y = np.mean(all_y, axis=0)
        std_y = np.std(all_y, axis=0)

        pT = curves[0][0]

        if show_std:

            ax.fill_between(
                pT,
                mean_y - std_y,
                mean_y + std_y,
                color=color,
                alpha=0.25,
                label=r"Simulation $1\sigma$",
            )

        if show_mean:

            ax.plot(
                pT,
                mean_y,
                color=color,
                linewidth=4,
                label=f"{correction}",
            )

        # ==========================================================
        # EXPERIMENTAL OVERLAY
        # ==========================================================

        if overlay_experiments is not None:

            for exp in overlay_experiments:

                self._plot_experimental_overlay(
                    ax=ax,
                    experiment=exp,
                    observable=observable,
                    centrality_index=centrality_index,
                    species=species,
                )

        # ==========================================================
        # LABELS
        # ==========================================================

        ax.set_xscale(xscale)
        ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xlabel(r"$p_T$ [GeV/$c$]")

        if observable == "v0pt":
            ax.set_ylabel(r"$v_0(p_T)$")

        elif observable == "ratio":
            ax.set_ylabel(r"$v_0(p_T)/v_0$")

        elif observable == "v0":
            ax.set_ylabel(r"$v_0$")

        # ==========================================================
        # GRID
        # ==========================================================

        ax.grid(
            True,
            which="major",
            linestyle="--",
            alpha=0.4,
        )

        ax.grid(
            True,
            which="minor",
            linestyle=":",
            alpha=0.2,
        )

        ax.minorticks_on()

        # ==========================================================
        # TITLE
        # ==========================================================

        if title is None:

            title = (
                f"{correction} | "
                f"{observable} | "
                f"cent={centrality_index}"
            )

        ax.set_title(title)

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
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":

    plotter = V0PTPlotter()

    # ==========================================================
    # LOAD SIMULATION DATA
    # ==========================================================

    plotter.load_simulation(
        "Grad",
        "design_points_data/differential_radial_flow/v0pt_design_points_results_Pb_Pb_2760_Grad.pkl"
    )

    # ==========================================================
    # LOAD EXPERIMENTAL DATA
    # ==========================================================

    plotter.load_experiment(
        "CMS",
        "experimental_data/CMS_v0pt.pkl"
    )

    plotter.load_experiment(
        "ATLAS",
        "experimental_data/ATLAS_v0pt.pkl"
    )

    # ==========================================================
    # PLOT
    # ==========================================================

    fig, ax = plotter.plot(
        correction="Grad",
        centrality_index=6,
        species="charged",
        observable="ratio",
        overlay_experiments=["CMS", "ATLAS"],
        show_std=True,
        ylim=(-4, 14),
        savepath="ratio_overlay.pdf",
    )

    plt.show()
