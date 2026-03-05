#!/usr/bin/env python3
"""
Plot emulator predictions against ATLAS experimental data with model/data ratio panels.
Loads ALL prediction data from pre-saved pickle files (no re-computation).

Expected input files (produced by the prediction script):
    predictions_vs_experimental/all_predictions_v0pt.pkl
    predictions_vs_experimental/all_predictions_v0.pkl
    predictions_vs_experimental/all_predictions_ratio.pkl

To generate those files, add the following block at the end of the
prediction script (just before the "CREATING PLOTS" section):

    import pickle
    with open(Path(OUTPUT_FOLDER) / "all_predictions_v0pt.pkl",   "wb") as f: pickle.dump(all_predictions_v0pt,   f)
    with open(Path(OUTPUT_FOLDER) / "all_predictions_v0.pkl",     "wb") as f: pickle.dump(all_predictions_v0,     f)
    with open(Path(OUTPUT_FOLDER) / "all_predictions_ratio.pkl",  "wb") as f: pickle.dump(all_predictions_ratio,  f)
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PLOTTING STYLE (Gardim style)
# =============================================================================

plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["xtick.major.pad"] = 4
plt.rcParams["ytick.major.pad"] = 4
plt.rcParams["xtick.minor.pad"] = 4
plt.rcParams["ytick.minor.pad"] = 4
plt.rcParams["legend.handletextpad"] = 0.0
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["ytick.direction"] = 'in'
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Centrality to plot (0-based index)
CENTRALITY_INDEX = 7  # 60-70%

# Experimental data files
EXP_V0PT_FILE = (
    f'experimental_data/differential_radial_flow/'
    f'HEPData-ins2907010-v1-Figure_4a_cent{CENTRALITY_INDEX + 1}.csv'
)
EXP_V0_FILE = 'experimental_data/differential_radial_flow/HEPData-ins2907010-v1-Figure_2a.csv'

# Pre-saved prediction files
PREDICTIONS_FOLDER = "predictions_vs_experimental"
PRED_V0PT_FILE  = f"{PREDICTIONS_FOLDER}/all_predictions_v0pt.pkl"
PRED_V0_FILE    = f"{PREDICTIONS_FOLDER}/all_predictions_v0.pkl"
PRED_RATIO_FILE = f"{PREDICTIONS_FOLDER}/all_predictions_ratio.pkl"

# Output folder (can be the same as PREDICTIONS_FOLDER or separate)
OUTPUT_FOLDER = "predictions_vs_experimental"

# Viscous corrections
VISCOUS_CORRECTIONS = ["Grad", "CE", "PTM", "PTB"]
COLOR_NAMES = {"Grad": 'blue', "CE": 'red', "PTM": 'magenta', "PTB": 'green'}

EXP_CENTRALITY_LABELS = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%",
                          "40-50%", "50-60%", "60-70%", "70-80%", "80-90%"]

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# =============================================================================
# UTILITY
# =============================================================================

def configure_axis(ax, fontsize):
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(labelsize=fontsize)


def save_fig(fig, filename):
    filepath = Path(OUTPUT_FOLDER) / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close(fig)


def _ratio_band(ax, x, pred_arr, exp_y, color, alpha, lw):
    """Interpolate pred_arr onto exp x-points, compute ratio to exp_y, plot band + median."""
    env = np.percentile(pred_arr, [5, 50, 95], axis=0)
    env_interp = np.zeros((3, len(x)))
    for i in range(3):
        env_interp[i] = np.interp(x, pred_arr._x if hasattr(pred_arr, '_x') else x, env[i])
    ratio_lo = env_interp[0] / exp_y
    ratio_med = env_interp[1] / exp_y
    ratio_hi = env_interp[2] / exp_y
    ax.fill_between(x, ratio_lo, ratio_hi, color=color, alpha=alpha)
    ax.plot(x, ratio_med, color=color, lw=lw, alpha=0.9)


def _ratio_band_interp(ax, x_pred, pred_arr, x_exp, exp_y, color, alpha, lw):
    """Same but with explicit x_pred != x_exp grids."""
    env = np.percentile(pred_arr, [5, 50, 95], axis=0)
    env_interp = np.zeros((3, len(x_exp)))
    for i in range(3):
        env_interp[i] = np.interp(x_exp, x_pred, env[i])
    ratio_lo  = env_interp[0] / exp_y
    ratio_med = env_interp[1] / exp_y
    ratio_hi  = env_interp[2] / exp_y
    ax.fill_between(x_exp, ratio_lo, ratio_hi, color=color, alpha=alpha)
    ax.plot(x_exp, ratio_med, color=color, lw=lw, alpha=0.9)

# =============================================================================
# LOAD EXPERIMENTAL DATA
# =============================================================================

def load_experimental_v0pt(filepath):
    if not Path(filepath).exists():
        print(f"  Warning: {filepath} not found!")
        return None
    data = pd.read_csv(filepath, comment='#')
    data.columns = ['pT', 'v0', 'stat_plus', 'stat_minus', 'sys_plus', 'sys_minus']
    stat_err  = np.abs(data['stat_plus'])
    sys_err   = np.abs(data['sys_plus'])
    total_err = np.sqrt(stat_err**2 + sys_err**2)
    print(f"  ✓ Loaded v0(pT): {len(data)} points")
    return {
        'pT': data['pT'].values,
        'v0': data['v0'].values,
        'stat_err': stat_err.values,
        'sys_err':  sys_err.values,
        'total_err': total_err.values
    }


def load_experimental_v0_global(filepath, centrality_idx):
    if not Path(filepath).exists():
        print(f"  Warning: {filepath} not found!")
        return None
    data = pd.read_csv(filepath, comment='#')

    # Keep only the first section (pT^ref = 0.5–2 GeV)
    for i, row in data.iterrows():
        if i > 0 and str(row.iloc[0]) == 'Centrality [%]':
            data = data.iloc[:i].copy()
            break

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.columns = ['centrality', 'v0', 'stat_plus', 'stat_minus', 'sys_plus', 'sys_minus']

    stat_err  = np.abs(data['stat_plus'].values)
    sys_err   = np.abs(data['sys_plus'].values)
    total_err = np.sqrt(stat_err**2 + sys_err**2)

    centbins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],
                          [40,50],[50,60],[60,70],[70,80],[80,90]])
    cent_center = (centbins[centrality_idx][0] + centbins[centrality_idx][1]) / 2.0
    idx = np.argmin(np.abs(data['centrality'].values - cent_center))

    print(f"  ✓ Loaded global v0 for {EXP_CENTRALITY_LABELS[centrality_idx]}: "
          f"v0={data['v0'].values[idx]:.5f}")
    return {
        'v0':        data['v0'].values[idx],
        'stat_err':  stat_err[idx],
        'sys_err':   sys_err[idx],
        'total_err': total_err[idx]
    }

# =============================================================================
# LOAD PRE-SAVED PREDICTIONS
# =============================================================================

def load_pkl(filepath):
    if not Path(filepath).exists():
        print(f"  ERROR: {filepath} not found!")
        print("  Run the prediction script first and save predictions to disk.")
        return None
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"  ✓ Loaded: {filepath}  keys={list(data.keys())}")
    return data

# =============================================================================
# PLOTTING — v0(pT)
# =============================================================================

def plot_v0pt_single_correction(corr, pred_data, exp_data, centrality_idx):
    """
    v0(pT) plot for one correction.
    Layers: Prior (gray hatched) → Original JETSCAPE (orange) → Posterior (colored) → Data.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10),
                             gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    ax_top, ax_bot = axes
    fontsize = 20

    pT = pred_data['pt_values']

    # ---------- Layer 1: Prior ----------
    if pred_data.get('prior') is not None:
        pr = pred_data['prior'][:, centrality_idx, :]
        pr_env = np.percentile(pr, [5, 50, 95], axis=0)
        ax_top.fill_between(pT, pr_env[0], pr_env[2],
                            hatch='///', edgecolor='gray', facecolor='gray',
                            alpha=0.35, lw=1.5, label='Prior 90% C.I.')
        ax_top.plot(pT, pr_env[1], color='gray', lw=1.5, ls='--', alpha=0.8)
        if exp_data is not None:
            _ratio_band_interp(ax_bot, pT, pr, exp_data['pT'], exp_data['v0'],
                               color='gray', alpha=0.25, lw=1.5)

    # ---------- Layer 2: Original JETSCAPE posterior (Grad only, orange) ----------
    if pred_data.get('original_posterior') is not None:
        orig = pred_data['original_posterior'][:, centrality_idx, :]
        orig_env = np.percentile(orig, [5, 50, 95], axis=0)
        ax_top.fill_between(pT, orig_env[0], orig_env[2],
                            color='orange', alpha=0.45,
                            label='Original JETSCAPE Posterior 90% C.I.')
        ax_top.plot(pT, orig_env[1], color='orange', lw=2.5, alpha=0.9)
        if exp_data is not None:
            _ratio_band_interp(ax_bot, pT, orig, exp_data['pT'], exp_data['v0'],
                               color='orange', alpha=0.35, lw=2.5)

    # ---------- Layer 3: Correction-specific posterior ----------
    if pred_data.get('posterior') is not None:
        post = pred_data['posterior'][:, centrality_idx, :]
        post_env = np.percentile(post, [5, 50, 95], axis=0)
        ax_top.fill_between(pT, post_env[0], post_env[2],
                            color=COLOR_NAMES[corr], alpha=0.4,
                            label=f'{corr} Posterior 90% C.I.')
        ax_top.plot(pT, post_env[1], color=COLOR_NAMES[corr], lw=2.5, alpha=0.9)
        if exp_data is not None:
            _ratio_band_interp(ax_bot, pT, post, exp_data['pT'], exp_data['v0'],
                               color=COLOR_NAMES[corr], alpha=0.35, lw=2.5)
    else:
        print(f"  ✗ No posterior for {corr}")

    # ---------- Layer 4: Experimental data ----------
    if exp_data is not None:
        ax_top.errorbar(exp_data['pT'], exp_data['v0'],
                        yerr=exp_data['total_err'],
                        fmt='o', color='black', markersize=7, capsize=4,
                        linewidth=2, alpha=0.95, label='ATLAS PbPb 5.02 TeV', zorder=10)
        exp_rel_err = exp_data['total_err'] / exp_data['v0']
        ax_bot.errorbar(exp_data['pT'], np.ones_like(exp_data['pT']),
                        yerr=exp_rel_err, fmt='o', color='black',
                        markersize=7, capsize=4, linewidth=2, alpha=0.95, zorder=10)

    # Formatting
    ax_top.set_ylabel(r"$v_0(p_T)$", fontsize=fontsize + 6)
    ax_top.legend(loc='best', frameon=False, fontsize=fontsize - 4)
    ax_top.set_xscale('log')
    ax_top.set_ylim(-0.5, 0.35)
    configure_axis(ax_top, fontsize)

    ax_bot.axhline(1.0, color='k', lw=1.5, ls='--')
    ax_bot.set_xlabel(r"$p_T$ [GeV/c]", fontsize=fontsize + 6)
    ax_bot.set_ylabel("Model/Data", fontsize=fontsize)
    ax_bot.set_xscale('log')
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlim(0.5, 3.5)
    configure_axis(ax_bot, fontsize - 2)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.05)
    return fig

# =============================================================================
# PLOTTING — Global v0 (bar chart for single centrality)
# =============================================================================

def plot_v0_global_single_correction(corr, pred_data, exp_data, centrality_idx):
    """
    Bar chart comparing posterior prediction vs data for global v0
    at a single centrality bin.
    Layers: Prior (gray) → Original JETSCAPE (orange) → Posterior (colored) → Data line.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    fontsize = 20

    x_positions = {'prior': -0.55, 'original_posterior': 0.0, 'posterior': 0.55}
    width = 0.45

    # ---------- Prior ----------
    if pred_data.get('prior') is not None:
        pr = pred_data['prior'][:, centrality_idx]
        pr_env = np.percentile(pr, [5, 50, 95])
        ax.bar(x_positions['prior'], pr_env[1],
               yerr=[[pr_env[1] - pr_env[0]], [pr_env[2] - pr_env[1]]],
               color='gray', alpha=0.5, capsize=5, width=width,
               label='Prior 90% C.I.')

    # ---------- Original JETSCAPE posterior ----------
    if pred_data.get('original_posterior') is not None:
        orig = pred_data['original_posterior'][:, centrality_idx]
        orig_env = np.percentile(orig, [5, 50, 95])
        ax.bar(x_positions['original_posterior'], orig_env[1],
               yerr=[[orig_env[1] - orig_env[0]], [orig_env[2] - orig_env[1]]],
               color='orange', alpha=0.6, capsize=5, width=width,
               label='Original JETSCAPE Posterior 90% C.I.')

    # ---------- Correction-specific posterior ----------
    if pred_data.get('posterior') is not None:
        post = pred_data['posterior'][:, centrality_idx]
        post_env = np.percentile(post, [5, 50, 95])
        ax.bar(x_positions['posterior'], post_env[1],
               yerr=[[post_env[1] - post_env[0]], [post_env[2] - post_env[1]]],
               color=COLOR_NAMES[corr], alpha=0.7, capsize=5, width=width,
               label=f'{corr} Posterior 90% C.I.')

    # ---------- Experimental data ----------
    if exp_data is not None:
        ax.axhline(exp_data['v0'], color='black', lw=2.5,
                   label='ATLAS PbPb 5.02 TeV', zorder=10)
        ax.axhspan(exp_data['v0'] - exp_data['total_err'],
                   exp_data['v0'] + exp_data['total_err'],
                   alpha=0.2, color='gray', zorder=0)

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(['Prior', 'Orig.\nJETSCAPE', corr], fontsize=fontsize - 2)
    ax.set_ylabel(r"Global $v_0$", fontsize=fontsize + 2)
    ax.legend(fontsize=fontsize - 4, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.08)
    ax.set_xlim(-0.9, 0.9)
    configure_axis(ax, fontsize - 2)

    plt.tight_layout()
    return fig

# =============================================================================
# PLOTTING — v0(pT)/v0 ratio
# =============================================================================

def plot_ratio_single_correction(corr, pred_data, exp_data_v0pt, exp_data_v0,
                                 centrality_idx):
    """
    v0(pT)/v0 ratio plot for one correction.
    Layers: Prior (gray hatched) → Original JETSCAPE (orange) → Posterior (colored) → Data.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10),
                             gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    ax_top, ax_bot = axes
    fontsize = 20

    pT = pred_data['pt_values']

    # Experimental ratio
    exp_ratio = exp_ratio_err = None
    if exp_data_v0pt is not None and exp_data_v0 is not None:
        exp_ratio = exp_data_v0pt['v0'] / exp_data_v0['v0']
        exp_ratio_err = exp_ratio * np.sqrt(
            (exp_data_v0pt['total_err'] / exp_data_v0pt['v0'])**2 +
            (exp_data_v0['total_err']   / exp_data_v0['v0'])**2
        )

    # ---------- Layer 1: Prior ----------
    if pred_data.get('prior') is not None:
        pr = pred_data['prior'][:, centrality_idx, :]
        pr_env = np.percentile(pr, [5, 50, 95], axis=0)
        ax_top.fill_between(pT, pr_env[0], pr_env[2],
                            hatch='///', edgecolor='gray', facecolor='gray',
                            alpha=0.35, lw=1.5, label='Prior 90% C.I.')
        ax_top.plot(pT, pr_env[1], color='gray', lw=1.5, ls='--', alpha=0.8)
        if exp_ratio is not None:
            _ratio_band_interp(ax_bot, pT, pr, exp_data_v0pt['pT'], exp_ratio,
                               color='gray', alpha=0.25, lw=1.5)

    # ---------- Layer 2: Original JETSCAPE posterior ----------
    if pred_data.get('original_posterior') is not None:
        orig = pred_data['original_posterior'][:, centrality_idx, :]
        orig_env = np.percentile(orig, [5, 50, 95], axis=0)
        ax_top.fill_between(pT, orig_env[0], orig_env[2],
                            color='orange', alpha=0.45,
                            label='Original JETSCAPE Posterior 90% C.I.')
        ax_top.plot(pT, orig_env[1], color='orange', lw=2.5, alpha=0.9)
        if exp_ratio is not None:
            _ratio_band_interp(ax_bot, pT, orig, exp_data_v0pt['pT'], exp_ratio,
                               color='orange', alpha=0.35, lw=2.5)

    # ---------- Layer 3: Correction-specific posterior ----------
    if pred_data.get('posterior') is not None:
        post = pred_data['posterior'][:, centrality_idx, :]
        post_env = np.percentile(post, [5, 50, 95], axis=0)
        ax_top.fill_between(pT, post_env[0], post_env[2],
                            color=COLOR_NAMES[corr], alpha=0.4,
                            label=f'{corr} Posterior 90% C.I.')
        ax_top.plot(pT, post_env[1], color=COLOR_NAMES[corr], lw=2.5, alpha=0.9)
        if exp_ratio is not None:
            _ratio_band_interp(ax_bot, pT, post, exp_data_v0pt['pT'], exp_ratio,
                               color=COLOR_NAMES[corr], alpha=0.35, lw=2.5)
    else:
        print(f"  ✗ No posterior for {corr}")

    # ---------- Layer 4: Experimental data ----------
    if exp_ratio is not None:
        ax_top.errorbar(exp_data_v0pt['pT'], exp_ratio,
                        yerr=exp_ratio_err,
                        fmt='o', color='black', markersize=7, capsize=4,
                        linewidth=2, alpha=0.95, label='ATLAS PbPb 5.02 TeV', zorder=10)
        exp_rel_err = exp_ratio_err / exp_ratio
        ax_bot.errorbar(exp_data_v0pt['pT'], np.ones_like(exp_data_v0pt['pT']),
                        yerr=exp_rel_err, fmt='o', color='black',
                        markersize=7, capsize=4, linewidth=2, alpha=0.95, zorder=10)

    # Formatting
    ax_top.set_ylabel(r"$v_0(p_T) / v_0$", fontsize=fontsize + 6)
    ax_top.legend(loc='best', frameon=False, fontsize=fontsize - 4)
    ax_top.set_xscale('log')
    ax_top.set_ylim(-0.5, 15)
    configure_axis(ax_top, fontsize)

    ax_bot.axhline(1.0, color='k', lw=1.5, ls='--')
    ax_bot.set_xlabel(r"$p_T$ [GeV/c]", fontsize=fontsize + 6)
    ax_bot.set_ylabel("Model/Data", fontsize=fontsize)
    ax_bot.set_xscale('log')
    ax_bot.grid(True, alpha=0.3, which='both')
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlim(0.5, 4)
    configure_axis(ax_bot, fontsize)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.05)
    return fig

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PLOTTING PREDICTIONS VS ATLAS DATA (LOAD-ONLY MODE)")
    print("=" * 80)
    print(f"\nCentrality: {EXP_CENTRALITY_LABELS[CENTRALITY_INDEX]}")

    # ------------------------------------------------------------------
    # Load experimental data
    # ------------------------------------------------------------------
    print("\nLoading experimental data...")
    exp_data_v0pt = load_experimental_v0pt(EXP_V0PT_FILE)
    exp_data_v0   = load_experimental_v0_global(EXP_V0_FILE, CENTRALITY_INDEX)

    # ------------------------------------------------------------------
    # Load pre-saved predictions
    # ------------------------------------------------------------------
    print("\nLoading pre-saved predictions...")
    all_predictions_v0pt  = load_pkl(PRED_V0PT_FILE)
    all_predictions_v0    = load_pkl(PRED_V0_FILE)
    all_predictions_ratio = load_pkl(PRED_RATIO_FILE)

    missing = [p for p in [PRED_V0PT_FILE, PRED_V0_FILE, PRED_RATIO_FILE]
               if not Path(p).exists()]
    if missing:
        print("\nERROR: The following prediction files are missing:")
        for p in missing:
            print(f"    {p}")
        print("\nAdd these lines to your prediction script (before the plotting section):")
        print()
        print("    import pickle")
        print("    from pathlib import Path")
        print("    with open(Path(OUTPUT_FOLDER) / 'all_predictions_v0pt.pkl',  'wb') as f:")
        print("        pickle.dump(all_predictions_v0pt, f)")
        print("    with open(Path(OUTPUT_FOLDER) / 'all_predictions_v0.pkl',    'wb') as f:")
        print("        pickle.dump(all_predictions_v0, f)")
        print("    with open(Path(OUTPUT_FOLDER) / 'all_predictions_ratio.pkl', 'wb') as f:")
        print("        pickle.dump(all_predictions_ratio, f)")
    else:
        # ------------------------------------------------------------------
        # Create plots — one set per correction
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("CREATING PLOTS...")
        print("=" * 80)

        cent_label = EXP_CENTRALITY_LABELS[CENTRALITY_INDEX].replace("-", "")

        for corr in VISCOUS_CORRECTIONS:
            if corr not in all_predictions_v0pt:
                print(f"\n  Skipping {corr} — not found in saved predictions")
                continue

            print(f"\n--- {corr} ---")

            # 1. v0(pT)
            if exp_data_v0pt is not None:
                fig = plot_v0pt_single_correction(
                    corr, all_predictions_v0pt[corr], exp_data_v0pt, CENTRALITY_INDEX
                )
                if fig is not None:
                    save_fig(fig, f'v0pt_vs_exp_{corr}_{cent_label}.pdf')

            # 2. Global v0 (bar chart)
            if corr in all_predictions_v0 and exp_data_v0 is not None:
                fig = plot_v0_global_single_correction(
                    corr, all_predictions_v0[corr], exp_data_v0, CENTRALITY_INDEX
                )
                if fig is not None:
                    save_fig(fig, f'v0_global_vs_exp_{corr}_{cent_label}.pdf')

            # 3. v0(pT)/v0 ratio
            if corr in all_predictions_ratio and exp_data_v0pt is not None and exp_data_v0 is not None:
                fig = plot_ratio_single_correction(
                    corr, all_predictions_ratio[corr], exp_data_v0pt, exp_data_v0, CENTRALITY_INDEX
                )
                if fig is not None:
                    save_fig(fig, f'ratio_vs_exp_{corr}_{cent_label}.pdf')

        print("\n" + "=" * 80)
        print("✓ PLOTTING COMPLETE")
        print(f"Output folder: {OUTPUT_FOLDER}/")
        print("=" * 80 + "\n")
