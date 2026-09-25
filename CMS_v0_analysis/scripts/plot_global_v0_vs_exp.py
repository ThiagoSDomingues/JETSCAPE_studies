#!/usr/bin/env python3
"""
Plot global v0 emulator predictions against ATLAS 5.02 TeV experimental data,
with CMS Pb–Pb 2.76 TeV integrated v0 overlaid in all panels.

CMS data is read directly from main.root and SystUncs.root using a pure-Python
ROOT parser (no ROOT or uproot required).  Only ptref-1 (0.5–2.0 GeV/c) is
used, matching the ATLAS reference window, for the two available centrality
bins: 50–60% (x=55) and 60–70% (x=65).

CMS points are shown as square markers (colour-coded per ptref window used)
and appear in both the main panel and the model/data ratio panel.
"""
import struct
import zlib
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# =============================================================================
# PLOTTING STYLE (Gardim style)
# =============================================================================

plt.rcParams["xtick.major.size"]     = 6
plt.rcParams["ytick.major.size"]     = 6
plt.rcParams["xtick.minor.size"]     = 3
plt.rcParams["ytick.minor.size"]     = 3
plt.rcParams["xtick.major.width"]    = 1.2
plt.rcParams["ytick.major.width"]    = 1.2
plt.rcParams["xtick.minor.width"]    = 1
plt.rcParams["ytick.minor.width"]    = 1
plt.rcParams["axes.linewidth"]       = 1
plt.rcParams["lines.linewidth"]      = 1
plt.rcParams["xtick.major.pad"]      = 4
plt.rcParams["ytick.major.pad"]      = 4
plt.rcParams["xtick.minor.pad"]      = 4
plt.rcParams["ytick.minor.pad"]      = 4
plt.rcParams["legend.handletextpad"] = 0.0
plt.rcParams["xtick.direction"]      = 'in'
plt.rcParams["ytick.direction"]      = 'in'
plt.rcParams["xtick.minor.visible"]  = True
plt.rcParams["ytick.minor.visible"]  = True
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# =============================================================================
# CONFIGURATION
# =============================================================================

EXP_V0_FILE     = 'experimental_data/differential_radial_flow/HEPData-ins2907010-v1-Figure_2a.csv'
PREDICTIONS_FILE = "predictions_global_v0/all_predictions_global_v0.pkl"
OUTPUT_FOLDER   = "predictions_global_v0_vs_exp"

# Path to CMS ROOT files
CMS_MAIN_ROOT = "main.root"
CMS_SYST_ROOT = "SystUncs.root"

VISCOUS_CORRECTIONS = ["Grad", "CE", "PTM", "PTB"]
COLOR_NAMES = {"Grad": 'blue', "CE": 'red', "PTM": 'magenta', "PTB": 'green'}

CENTRALITY_EDGES = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80]
CENTRALITY_CENTERS_CORRECT = np.array(
    [(CENTRALITY_EDGES[i] + CENTRALITY_EDGES[i + 1]) / 2.0
     for i in range(len(CENTRALITY_EDGES) - 1)]
)

CENTRALITY_LABELS = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%",
                     "40-50%", "50-60%", "60-70%", "70-80%"]

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


# =============================================================================
# CMS ROOT FILE READER  (pure Python — no ROOT / uproot needed)
# =============================================================================
#
# ROOT stores each TGraph object in its own zlib-compressed block.
# Uncompressed single-point objects (TGraphErrors, N=1) are stored inline.
# The binary layout (big-endian IEEE-754) is:
#
#   TGraphErrors (N=1, uncompressed):
#     … header … [x: 8B double] [1B sep] [y: 8B double] [TList ~57B + ex]
#     … [stat_ey: last 8B in block]
#
#   TGraphErrors (N=2, uncompressed) — gr_v0_total in SystUncs:
#     … [x0,x1: 16B] [1B sep] [y0,y1: 16B] [TList 57B] [ex0,ex1: 16B] [1B sep]
#     [ey0,ey1: 16B]

def _read_pascal(raw: bytes, pos: int):
    n = raw[pos]
    return raw[pos + 1: pos + 1 + n].decode("latin-1", errors="replace"), pos + 1 + n


def _parse_root_keys(raw: bytes) -> dict:
    """Walk the ROOT TKey records and return {name: {data_start, data_len, classname}}."""
    keys = {}
    pos  = 100  # ROOT file header is 100 bytes
    while pos < len(raw) - 26:
        nbytes = struct.unpack(">i", raw[pos: pos + 4])[0]
        if nbytes == 0:
            break
        if nbytes < 0:
            pos += -nbytes
            continue
        keylen = struct.unpack(">H", raw[pos + 14: pos + 16])[0]
        str_pos = pos + 26
        try:
            classname, str_pos = _read_pascal(raw, str_pos)
            name,      str_pos = _read_pascal(raw, str_pos)
            _title,    str_pos = _read_pascal(raw, str_pos)
        except Exception:
            pos += nbytes
            continue
        keys[name] = {
            "data_start": pos + keylen,
            "data_len":   nbytes - keylen,
            "classname":  classname,
        }
        pos += nbytes
    return keys


def _rd(data: bytes, off: int) -> float:
    """Read one big-endian double at `off`."""
    return struct.unpack(">d", data[off: off + 8])[0]


def _parse_v0_scalar(raw: bytes, key: dict) -> Optional[dict]:
    """
    Parse a single-point (N=1) uncompressed TGraphErrors.
#    x ∈ {55.0, 65.0} encodes the centrality centre (%).
#    The statistical error is the last 8 bytes of the object data.
#    """
    data = raw[key["data_start"]: key["data_start"] + key["data_len"]]
    for x_off in range(70, 110):
        if x_off + 8 > len(data):
            break
        x = _rd(data, x_off)
        if abs(x - 55) < 0.1 or abs(x - 65) < 0.1:
            y    = _rd(data, x_off + 8 + 1)      # +8 for x, +1 separator byte
            stat = abs(_rd(data, len(data) - 8))  # last double = stat error
            return {"x": x, "y": y, "stat": stat}
    return None


def _parse_v0_syst_graph(raw_s: bytes, keys_s: dict,
                          name: str = "gr_v0_total") -> Optional[dict]:
    """
    Parse the uncompressed N=2 TGraphErrors that holds systematic errors
    for both centralities combined (x = [55, 65]).
    Layout after header:
      x[2]  (16B) | 1B sep | y[2] (16B) | TList 57B | ex[2] (16B) | 1B sep | ey[2] (16B)
    Returns dict with keys: x, y, ex, ey  (all length-2 arrays).
    """
    if name not in keys_s:
        return None
    k    = keys_s[name]
    data = raw_s[k["data_start"]: k["data_start"] + k["data_len"]]
    for x_off in range(70, 110):
        if x_off + 16 > len(data):
            break
        x0, x1 = _rd(data, x_off), _rd(data, x_off + 8)
        if abs(x0 - 55) < 0.1 and abs(x1 - 65) < 0.1:
            y_off  = x_off + 16 + 1         # +16 x-array, +1 sep
            ex_off = y_off  + 16 + 57       # +16 y-array, +57 TList metadata
            ey_off = ex_off + 16 + 1        # +16 ex-array, +1 sep
            return {
                "x":  np.array([_rd(data, x_off),  _rd(data, x_off  + 8)]),
                "y":  np.array([_rd(data, y_off),   _rd(data, y_off  + 8)]),
                "ex": np.array([_rd(data, ex_off),  _rd(data, ex_off + 8)]),
                "ey": np.array([_rd(data, ey_off),  _rd(data, ey_off + 8)]),
            }
    return None


def load_cms_v0_data(main_root_path: str, syst_root_path: str) -> Optional[dict]:
    """
    Extract CMS Pb–Pb 2.76 TeV integrated v0 (ptref-1, 0.5–2.0 GeV/c) for
    both available centrality bins: 50–60% and 60–70%.

    Returns
    -------
    dict with:
        centrality  : np.array [55.0, 65.0]   (bin centres, %)
        v0          : np.array [v0_5060, v0_6070]
        stat_err    : np.array
        syst_err    : np.array
        total_err   : np.array  sqrt(stat² + syst²)
    or None if files are missing.
    """
    main_p = Path(main_root_path)
    syst_p = Path(syst_root_path)
    if not main_p.exists() or not syst_p.exists():
        print(f"  WARNING: CMS ROOT files not found at '{main_root_path}' / '{syst_root_path}'")
        return None

    raw   = main_p.read_bytes()
    raw_s = syst_p.read_bytes()
    keys_m = _parse_root_keys(raw)
    keys_s = _parse_root_keys(raw_s)

    # Statistical data from main.root (ptref-1 = window _1)
    r55 = _parse_v0_scalar(raw, keys_m["v0_55_1"])
    r65 = _parse_v0_scalar(raw, keys_m["v0_65_1"])
    if r55 is None or r65 is None:
        print("  WARNING: Could not parse CMS v0 scalar objects.")
        return None

    # Systematic errors from SystUncs.root (gr_v0_total, ptref-1)
    syst_graph = _parse_v0_syst_graph(raw_s, keys_s, "gr_v0_total")
    if syst_graph is not None:
        # syst_graph['ey'] = [syst_for_55, syst_for_65]
        syst_arr = syst_graph["ey"]
    else:
        print("  WARNING: Could not parse CMS syst graph; syst set to 0.")
        syst_arr = np.array([0.0, 0.0])

    centrality = np.array([r55["x"], r65["x"]])   # [55.0, 65.0]
    v0         = np.array([r55["y"], r65["y"]])
    stat_err   = np.array([r55["stat"], r65["stat"]])
    syst_err   = syst_arr
    total_err  = np.sqrt(stat_err**2 + syst_err**2)

    print(f"  ✓ CMS 2.76 TeV  v0 = {v0}  stat = {stat_err}  syst = {syst_err}")
    return {
        "centrality": centrality,
        "v0":         v0,
        "stat_err":   stat_err,
        "syst_err":   syst_err,
        "total_err":  total_err,
    }


# =============================================================================
# UTILITY FUNCTIONS  (unchanged from original)
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


# =============================================================================
# CENTRALITY BIN ALIGNMENT  (unchanged from original)
# =============================================================================

def remap_predictions_to_correct_centrality(pred_data):
    stored_centers = np.asarray(pred_data['centrality_centers'])
    n_pred    = len(stored_centers)
    n_correct = len(CENTRALITY_CENTERS_CORRECT)

    if n_pred == n_correct:
        print(f"    Remapping {n_pred} prediction bins:")
        print(f"      Stored centers : {stored_centers}")
        print(f"      Correct centers: {CENTRALITY_CENTERS_CORRECT}")
        remapped = pred_data.copy()
        remapped['centrality_centers'] = CENTRALITY_CENTERS_CORRECT.copy()
        return remapped
    else:
        print(f"    WARNING: {n_pred} prediction bins vs {n_correct} correct bins.")
        remapped = pred_data.copy()
        for key in ('prior', 'original_posterior', 'posterior'):
            arr = pred_data.get(key)
            if arr is not None:
                arr_new = np.zeros((arr.shape[0], n_correct))
                for s in range(arr.shape[0]):
                    arr_new[s, :] = np.interp(
                        CENTRALITY_CENTERS_CORRECT, stored_centers, arr[s, :]
                    )
                remapped[key] = arr_new
        remapped['centrality_centers'] = CENTRALITY_CENTERS_CORRECT.copy()
        return remapped


def clip_predictions_to_exp_range(pred_data, exp_data):
    cent    = np.asarray(pred_data['centrality_centers'])
    exp_max = exp_data['centrality'].max()
    exp_min = exp_data['centrality'].min()
    mask    = (cent >= exp_min) & (cent <= exp_max)
    clipped = pred_data.copy()
    clipped['centrality_centers'] = cent[mask]
    for key in ('prior', 'original_posterior', 'posterior'):
        if clipped.get(key) is not None:
            clipped[key] = clipped[key][:, mask]
    return clipped


# =============================================================================
# LOAD EXPERIMENTAL DATA  (unchanged from original)
# =============================================================================

def load_experimental_v0_global(filepath):
    if not Path(filepath).exists():
        print(f"Warning: Experimental data file '{filepath}' not found!")
        return None

    data = pd.read_csv(filepath, comment='#')
    first_section_end = None
    for i, row in data.iterrows():
        if i > 0 and str(row.iloc[0]) == 'Centrality [%]':
            first_section_end = i
            break
    if first_section_end is not None:
        data = data.iloc[:first_section_end].copy()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.columns = ['centrality', 'v0', 'stat_plus', 'stat_minus', 'sys_plus', 'sys_minus']

    stat_err  = np.abs(data['stat_plus'].values)
    sys_err   = np.abs(data['sys_plus'].values)
    total_err = np.sqrt(stat_err**2 + sys_err**2)

    print(f"  ✓ Loaded ATLAS 5.02 TeV data: {len(data)} centrality bins")
    print(f"  Centrality values: {data['centrality'].values}")
    return {
        'centrality': data['centrality'].values,
        'v0':         data['v0'].values,
        'stat_err':   stat_err,
        'sys_err':    sys_err,
        'total_err':  total_err,
    }


def load_predictions(filepath):
    if not Path(filepath).exists():
        print(f"ERROR: Prediction file '{filepath}' not found!")
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def preprocess_predictions(all_predictions, exp_data):
    processed = {}
    for corr, pred_data in all_predictions.items():
        print(f"\n  Processing {corr}:")
        remapped = remap_predictions_to_correct_centrality(pred_data)
        if exp_data is not None:
            clipped = clip_predictions_to_exp_range(remapped, exp_data)
            n_clipped = len(clipped['centrality_centers'])
            print(f"    Clipped to {n_clipped} bins within exp range "
                  f"[{exp_data['centrality'].min():.1f}%, "
                  f"{exp_data['centrality'].max():.1f}%]")
            processed[corr] = clipped
        else:
            processed[corr] = remapped
    return processed


# =============================================================================
# CMS OVERLAY HELPERS
# =============================================================================

# CMS marker style — distinct from the ATLAS 'o'
CMS_MARKER    = 's'          # filled square
CMS_COLOR     = 'red'      # same ink colour as ATLAS points
CMS_MARKERSIZE = 7
CMS_CAPSIZE   = 4
CMS_LINEWIDTH  = 2
CMS_ALPHA      = 0.95
# Small horizontal jitter so CMS squares don't sit exactly on ATLAS circles
# (ATLAS bins at 55 and 65 happen to coincide with CMS centres)
CMS_X_JITTER  = 0          # shift CMS points to the right


def _draw_cms_main_panel(ax, cms_data, fontsize, label_first=True):
    """
    Add CMS 2.76 TeV points with total error bars to the main v0 panel.
    Systematic uncertainty is also shown as a horizontal cap box.
    """
    if cms_data is None:
        return

    x      = cms_data["centrality"] + CMS_X_JITTER
    y      = cms_data["v0"]
    stat   = cms_data["stat_err"]
    syst   = cms_data["syst_err"]
    total  = cms_data["total_err"]

    lbl = r"CMS 2.76 TeV" if label_first else None

    # Total error bar (stat ⊕ syst)
    ax.errorbar(
        x, y,
        yerr=total,
        fmt=CMS_MARKER,
        color=CMS_COLOR,
        markersize=CMS_MARKERSIZE,
        capsize=CMS_CAPSIZE,
        linewidth=CMS_LINEWIDTH,
        alpha=CMS_ALPHA,
        label=lbl,
        zorder=10,
    )

    # Systematic-only box (open rectangle spanning ±syst)
    for xi, yi, systi in zip(x, y, syst):
        box_w = 1.2   # half-width of the syst box in centrality %
        rect = plt.Rectangle(
            (xi - box_w, yi - systi), 2 * box_w, 2 * systi,
            linewidth=1.2, edgecolor=CMS_COLOR, facecolor='none',
            linestyle='-', zorder=11,
        )
        ax.add_patch(rect)


def _draw_cms_ratio_panel(ax, cms_data, atlas_data):
    """
    Add CMS / ATLAS-model ratio points to the ratio panel.

    Because the CMS data is at a *different energy* (2.76 TeV vs 5.02 TeV),
    we do NOT divide CMS by an ATLAS model prediction.  Instead we divide
    CMS v0 by the nearest-centrality ATLAS *data* point so the ratio
    shows the 2.76 / 5.02 TeV data comparison at the same centrality.

    This gives two ratio points, one per CMS centrality bin.
    """
    if cms_data is None or atlas_data is None:
        return

    # For each CMS centrality bin, find the nearest ATLAS data centrality
    cms_cent  = cms_data["centrality"]    # [55, 65]
    atlas_v0  = np.interp(cms_cent, atlas_data["centrality"], atlas_data["v0"])
    atlas_err = np.interp(cms_cent, atlas_data["centrality"], atlas_data["total_err"])

    ratio      = cms_data["v0"] / atlas_v0
    # Error propagation: δ(A/B)/( A/B) = sqrt( (δA/A)² + (δB/B)² )
    ratio_err  = ratio * np.sqrt(
        (cms_data["total_err"] / cms_data["v0"])**2 +
        (atlas_err / atlas_v0)**2
    )

    x = cms_cent + CMS_X_JITTER

    ax.errorbar(
        x, ratio,
        yerr=ratio_err,
        fmt=CMS_MARKER,
        color=CMS_COLOR,
        markersize=CMS_MARKERSIZE,
        capsize=CMS_CAPSIZE,
        linewidth=CMS_LINEWIDTH,
        alpha=CMS_ALPHA,
        label=r"CMS 2.76 / ATLAS 5.02 TeV",
        zorder=10,
    )

    # Annotate with the ratio values
#    for xi, ri in zip(x, ratio):
#        ax.annotate(
#            f"{ri:.2f}",
#            xy=(xi, ri),
#            xytext=(3, 6),
#            textcoords="offset points",
#            fontsize=9,
#            color=CMS_COLOR,
#            alpha=0.85,
#        )


# =============================================================================
# RATIO BAND HELPER  (unchanged from original)
# =============================================================================

def _ratio_band(ax, cent_centers, pred_arr, exp_data, color, alpha, lw, label=None):
    env = np.percentile(pred_arr, [5, 50, 95], axis=0)
    env_interp = np.zeros((3, len(exp_data['centrality'])))
    for i in range(3):
        env_interp[i, :] = np.interp(
            exp_data['centrality'], cent_centers, env[i, :]
        )
    ax.fill_between(
        exp_data['centrality'],
        env_interp[0, :] / exp_data['v0'],
        env_interp[2, :] / exp_data['v0'],
        color=color, alpha=alpha,
    )
    ax.plot(
        exp_data['centrality'],
        env_interp[1, :] / exp_data['v0'],
        color=color, lw=lw, alpha=0.9, label=label,
    )


# =============================================================================
# PLOTTING FUNCTIONS  (updated to accept cms_data)
# =============================================================================

def plot_global_v0_single_correction(corr, pred_data, exp_data, cms_data=None):
    """
    Plot global v0 for a single correction with model/data ratio panel.
    CMS 2.76 TeV data overlaid in both panels when cms_data is provided.
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 10),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )
    ax_top = axes[0]
    ax_bot = axes[1]
    fontsize = 20

    cent_centers = pred_data['centrality_centers']
    x_min = exp_data['centrality'].min() - 3 if exp_data is not None else 0
    x_max = exp_data['centrality'].max() + 3 if exp_data is not None else 95

    # -- Layer 1: Prior --
    if pred_data.get('prior') is not None:
        pr_env = np.percentile(pred_data['prior'], [5, 50, 95], axis=0)
        ax_top.fill_between(
            cent_centers, pr_env[0, :], pr_env[2, :],
            hatch='///', edgecolor='gray', facecolor='gray',
            alpha=0.35, lw=1.5, label='Prior 90% C.I.',
        )
        ax_top.plot(cent_centers, pr_env[1, :], color='gray',
                    lw=1.5, ls='--', alpha=0.8)
        if exp_data is not None:
            _ratio_band(ax_bot, cent_centers, pred_data['prior'],
                        exp_data, color='gray', alpha=0.25, lw=1.5)

    # -- Layer 2: Original JETSCAPE posterior --
    if pred_data.get('original_posterior') is not None:
        orig_env = np.percentile(pred_data['original_posterior'], [5, 50, 95], axis=0)
        ax_top.fill_between(
            cent_centers, orig_env[0, :], orig_env[2, :],
            color='orange', alpha=0.45,
            label='Original JETSCAPE Posterior 90% C.I.',
        )
        ax_top.plot(cent_centers, orig_env[1, :], color='orange',
                    lw=2.5, alpha=0.9)
        if exp_data is not None:
            _ratio_band(ax_bot, cent_centers, pred_data['original_posterior'],
                        exp_data, color='orange', alpha=0.35, lw=2.5)

    # -- Layer 3: Correction-specific posterior --
    if pred_data.get('posterior') is not None:
        post_env = np.percentile(pred_data['posterior'], [5, 50, 95], axis=0)
        ax_top.fill_between(
            cent_centers, post_env[0, :], post_env[2, :],
            color=COLOR_NAMES[corr], alpha=0.4,
            label=f'{corr} Posterior 90% C.I.',
        )
        ax_top.plot(cent_centers, post_env[1, :],
                    color=COLOR_NAMES[corr], lw=2.5, alpha=0.9)
        if exp_data is not None:
            _ratio_band(ax_bot, cent_centers, pred_data['posterior'],
                        exp_data, color=COLOR_NAMES[corr], alpha=0.35, lw=2.5)

    # -- Layer 4: ATLAS 5.02 TeV data --
    if exp_data is not None:
        ax_top.errorbar(
            exp_data['centrality'], exp_data['v0'],
            yerr=exp_data['total_err'],
            fmt='o', color='black', markersize=7, capsize=4,
            linewidth=2, alpha=0.95, label='ATLAS 5.02 TeV', zorder=10,
        )
        exp_rel_err = exp_data['total_err'] / exp_data['v0']
        ax_bot.errorbar(
            exp_data['centrality'],
            np.ones_like(exp_data['centrality']),
            yerr=exp_rel_err,
            fmt='o', color='black', markersize=7, capsize=4,
            linewidth=2, alpha=0.95, zorder=10,
        )

    # ── NEW: CMS 2.76 TeV overlay ────────────────────────────────────────────
    _draw_cms_main_panel(ax_top, cms_data, fontsize, label_first=True)
    _draw_cms_ratio_panel(ax_bot, cms_data, exp_data)

    # -- Formatting --
    ax_top.set_ylabel(r"$v_0$", fontsize=fontsize + 8)
    ax_top.legend(loc='best', frameon=False, fontsize=fontsize - 4)
    ax_top.set_ylim(0, 0.10)
    ax_top.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_top.set_xlim(x_min, x_max)
    configure_axis(ax_top, fontsize - 2)

    ax_bot.axhline(1.0, color='k', lw=1.5, ls='--')
    ax_bot.set_xlabel("Centrality [%]", fontsize=fontsize + 8)
    ax_bot.set_ylabel("Model/Data", fontsize=fontsize - 2)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlim(x_min, x_max)
    configure_axis(ax_bot, fontsize - 2)

    # Add a secondary legend in the ratio panel for the CMS entry
    handles, labels = ax_bot.get_legend_handles_labels()
    if handles:
        ax_bot.legend(handles, labels, fontsize=fontsize - 7, loc='upper left',
                      frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.05)
    return fig


def plot_global_v0_all_corrections(all_predictions, exp_data, cms_data=None):
    """
    Plot global v0 for all corrections overlaid with model/data ratio panel.
    CMS 2.76 TeV data overlaid in both panels when cms_data is provided.
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )
    ax_top = axes[0]
    ax_bot = axes[1]
    fontsize = 20

    x_min = exp_data['centrality'].min() - 3 if exp_data is not None else 0
    x_max = exp_data['centrality'].max() + 3 if exp_data is not None else 95

    # -- Layer 1: Prior --
    prior_plotted = False
    for corr in VISCOUS_CORRECTIONS:
        if corr not in all_predictions or prior_plotted:
            continue
        pred_data = all_predictions[corr]
        if pred_data.get('prior') is None:
            continue
        cent_centers = pred_data['centrality_centers']
        pr_env = np.percentile(pred_data['prior'], [5, 50, 95], axis=0)
        ax_top.fill_between(
            cent_centers, pr_env[0, :], pr_env[2, :],
            hatch='///', edgecolor='gray', facecolor='gray',
            alpha=0.35, lw=1.5, label='Prior 90% C.I.',
        )
        ax_top.plot(cent_centers, pr_env[1, :], color='gray',
                    lw=1.5, ls='--', alpha=0.8)
        if exp_data is not None:
            _ratio_band(ax_bot, cent_centers, pred_data['prior'],
                        exp_data, color='gray', alpha=0.25, lw=1.5)
        prior_plotted = True

    # -- Layer 2: Original JETSCAPE posterior --
    for corr in VISCOUS_CORRECTIONS:
        if corr not in all_predictions:
            continue
        pred_data = all_predictions[corr]
        if pred_data.get('original_posterior') is None:
            continue
        cent_centers = pred_data['centrality_centers']
        orig_env = np.percentile(pred_data['original_posterior'], [5, 50, 95], axis=0)
        ax_top.fill_between(
            cent_centers, orig_env[0, :], orig_env[2, :],
            color='orange', alpha=0.45,
            label='Original JETSCAPE Posterior 90% C.I.',
        )
        ax_top.plot(cent_centers, orig_env[1, :], color='orange',
                    lw=2.5, alpha=0.9)
        if exp_data is not None:
            _ratio_band(ax_bot, cent_centers, pred_data['original_posterior'],
                        exp_data, color='orange', alpha=0.35, lw=2.5)

    # -- Layer 3: Correction-specific posteriors --
    for corr in VISCOUS_CORRECTIONS:
        if corr not in all_predictions:
            continue
        pred_data = all_predictions[corr]
        if pred_data.get('posterior') is None:
            continue
        cent_centers = pred_data['centrality_centers']
        post_env = np.percentile(pred_data['posterior'], [5, 50, 95], axis=0)
        ax_top.fill_between(
            cent_centers, post_env[0, :], post_env[2, :],
            color=COLOR_NAMES[corr], alpha=0.3,
        )
        ax_top.plot(cent_centers, post_env[1, :], color=COLOR_NAMES[corr],
                    lw=2.5, alpha=0.9, label=f'{corr} Posterior')
        if exp_data is not None:
            _ratio_band(ax_bot, cent_centers, pred_data['posterior'],
                        exp_data, color=COLOR_NAMES[corr], alpha=0.25, lw=2.5)

    # -- Layer 4: ATLAS 5.02 TeV data --
    if exp_data is not None:
        ax_top.errorbar(
            exp_data['centrality'], exp_data['v0'],
            yerr=exp_data['total_err'],
            fmt='o', color='black', markersize=7, capsize=4,
            linewidth=2, alpha=0.95, label='ATLAS 5.02 TeV', zorder=10,
        )
        exp_rel_err = exp_data['total_err'] / exp_data['v0']
        ax_bot.errorbar(
            exp_data['centrality'],
            np.ones_like(exp_data['centrality']),
            yerr=exp_rel_err,
            fmt='o', color='black', markersize=7, capsize=4,
            linewidth=2, alpha=0.95, zorder=10,
        )

    # ── NEW: CMS 2.76 TeV overlay ────────────────────────────────────────────
    _draw_cms_main_panel(ax_top, cms_data, fontsize, label_first=True)
    _draw_cms_ratio_panel(ax_bot, cms_data, exp_data)

    # -- Formatting --
    ax_top.set_ylabel(r"$v_0$", fontsize=fontsize + 8)
    ax_top.legend(loc='best', frameon=False, fontsize=fontsize - 4, ncol=2)
    ax_top.set_ylim(0, 0.10)
    ax_top.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_top.set_xlim(x_min, x_max)
    configure_axis(ax_top, fontsize - 2)

    ax_bot.axhline(1.0, color='k', lw=1.5, ls='--')
    ax_bot.set_xlabel("Centrality [%]", fontsize=fontsize + 8)
    ax_bot.set_ylabel("Model/Data", fontsize=fontsize - 2)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlim(x_min, x_max)
    configure_axis(ax_bot, fontsize - 2)

    handles, labels = ax_bot.get_legend_handles_labels()
    if handles:
        ax_bot.legend(handles, labels, fontsize=fontsize - 7, loc='upper left',
                      frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.05)
    return fig


def print_comparison_summary(all_predictions, exp_data):
    print("\n" + "=" * 80)
    print("MODEL VS DATA COMPARISON SUMMARY")
    print("=" * 80)

    if exp_data is not None:
        print(f"\nExperimental Data (ATLAS 5.02 TeV):")
        print(f"  Centrality bins: {len(exp_data['centrality'])}")
        print(f"  Centrality centers: {exp_data['centrality']}")
        print(f"  v0 range: {exp_data['v0'].min():.5f} - {exp_data['v0'].max():.5f}")

    print(f"\nModel Predictions (centrality centers after remapping):")
    for corr in VISCOUS_CORRECTIONS:
        if corr not in all_predictions:
            continue
        pred_data = all_predictions[corr]
        if pred_data['posterior'] is None:
            continue

        cent_centers = pred_data['centrality_centers']
        post_pred    = pred_data['posterior']
        post_median  = np.median(post_pred, axis=0)

        print(f"\n  {corr}:")
        print(f"    Centrality centers: {cent_centers}")
        print(f"    Median v0 range: {post_median.min():.5f} - {post_median.max():.5f}")

        if exp_data is not None:
            post_med_interp = np.interp(exp_data['centrality'], cent_centers, post_median)
            ratios = post_med_interp / exp_data['v0']
            print(f"    Model/Data ratio range: {ratios.min():.3f} - {ratios.max():.3f}")
            print(f"    Mean Model/Data ratio:  {ratios.mean():.3f} ± {ratios.std():.3f}")

    print("\n" + "=" * 80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PLOTTING GLOBAL v0 PREDICTIONS VS ATLAS 5.02 TeV DATA")
    print("(+ CMS Pb–Pb 2.76 TeV integrated v0 overlay)")
    print("=" * 80)

    # Load ATLAS data
    print("\nLoading ATLAS 5.02 TeV experimental data...")
    exp_data = load_experimental_v0_global(EXP_V0_FILE)

    # Load CMS data from ROOT files
    print("\nLoading CMS Pb–Pb 2.76 TeV data from ROOT files...")
    cms_data = load_cms_v0_data(CMS_MAIN_ROOT, CMS_SYST_ROOT)

    # Load predictions
    print("\nLoading prediction data...")
    all_predictions_raw = load_predictions(PREDICTIONS_FILE)
    if all_predictions_raw is None:
        print("\nERROR: Could not load predictions. Run predict_global_v0.py first.")
        exit(1)
    print(f"✓ Loaded predictions for: {list(all_predictions_raw.keys())}")

    # Remap centrality bins
    print("\nRemapping and clipping prediction centrality bins...")
    all_predictions = preprocess_predictions(all_predictions_raw, exp_data)

    print_comparison_summary(all_predictions, exp_data)

    print("\n" + "=" * 80)
    print("CREATING PLOTS  (with CMS 2.76 TeV overlay)")
    print("=" * 80)

    # Combined plot
    print("\n1. Combined plot with all corrections...")
    if exp_data is not None:
        fig_combined = plot_global_v0_all_corrections(all_predictions, exp_data,
                                                       cms_data=cms_data)
        save_fig(fig_combined, 'global_v0_all_corrections_vs_exp.pdf')

    # Individual plots per correction
    print("\n2. Individual plots per correction...")
    for corr in VISCOUS_CORRECTIONS:
        if corr not in all_predictions:
            continue
        print(f"   Plotting {corr}...")
        fig = plot_global_v0_single_correction(
            corr, all_predictions[corr], exp_data, cms_data=cms_data
        )
        if fig is not None:
            save_fig(fig, f'global_v0_{corr}_vs_exp.pdf')

    print("\n" + "=" * 80)
    print("✓ PLOTTING COMPLETE")
    print(f"  Output folder: {OUTPUT_FOLDER}/")
    print("=" * 80 + "\n")
