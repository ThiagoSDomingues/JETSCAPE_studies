#!/usr/bin/env python3
"""
CMS PbPb 2.76 TeV  — v0(pT) open data reader and plotter
==========================================================
Reads main.root and SystUncs.root without ROOT or uproot.
Uses only Python stdlib (struct, zlib) + numpy/pandas/matplotlib.

Files
-----
main.root     : TGraphErrors  — data points + statistical errors
                Objects: sv0pt_ptref_1_{cent}, v0pt_ptref_1_{cent}, v0ptv0_ptref_1_{cent}
                (ptref-1 = <pT> reference window 0.5–2.0 GeV/c)
                sv0pt  = v0(pT) / v0   (ratio)
                v0pt   = v0(pT)        (differential radial flow)
                v0ptv0 = v0(pT) × v0   (?)
                cent   = 5060 or 6070

SystUncs.root : TGraphAsymmErrors — data points + total systematic errors
                Objects: gr_sv0pt_{cent}_total, gr_v0pt_{cent}_total, ...

Output
------
  cms_pbpb_data/  ← organised CSV files
  plots/          ← PDF plots
"""

import struct
import zlib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3, "ytick.minor.size": 3,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "axes.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False,
    "font.family": "serif",
})

# ── Paths ─────────────────────────────────────────────────────────────────────
MAIN_ROOT  = Path("main.root")
SYST_ROOT  = Path("SystUncs.root")
CSV_DIR    = Path("cms_pbpb_data")
PLOT_DIR   = Path("cms_pbpb_plots")
CSV_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

CENTRALITIES = {"5060": "50–60%", "6070": "60–70%"}


# ═══════════════════════════════════════════════════════════════════════════════
# ROOT FILE READER  (pure Python, no ROOT/uproot needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _decompress_root(filepath: Path) -> list[bytes]:
    """
    Extract all zlib-compressed data blocks from a ROOT file.
    ROOT stores each TGraph object in its own compressed chunk.
    """
    raw = filepath.read_bytes()
    blocks = []
    i = 0
    while i < len(raw) - 2:
        # zlib magic bytes: 0x78 followed by 0x01, 0x5E, 0x9C, or 0xDA
        if raw[i] == 0x78 and raw[i + 1] in (0x9C, 0x01, 0xDA, 0x5E):
            try:
                blocks.append(zlib.decompress(raw[i:]))
            except zlib.error:
                pass
        i += 1
    return blocks


def _read_doubles(block: bytes, offset: int, n: int) -> list[float]:
    """Read n big-endian doubles starting at `offset`."""
    return [struct.unpack(">d", block[offset + k * 8: offset + k * 8 + 8])[0]
            for k in range(n)]


def _find_x_offset(block: bytes, n: int = 18) -> int | None:
    """
    Find the byte offset where the monotone-ascending pT array begins.
    ROOT stores TGraph arrays as big-endian IEEE-754 double (8 bytes each),
    preceded by a 4-byte int32 length.  There is 1 separator byte between
    consecutive arrays, and ~57 bytes of TList metadata between y and exl.
    """
    for i in range(70, 140):
        if i + n * 8 > len(block):
            break
        trial = _read_doubles(block, i, n)
        if (all(0.3 < v < 15 for v in trial)
                and all(trial[k] < trial[k + 1] for k in range(n - 1))):
            return i
    return None


def read_tgrapherrors(block: bytes, n: int = 18) -> dict | None:
    """
    Parse a TGraphErrors block.
    Layout: [header] x[n] [1 sep] y[n] [TList ~57B] ey[n]
    Returns dict with keys: x, y, ey
    """
    x_off = _find_x_offset(block, n)
    if x_off is None:
        return None
    y_off  = x_off  + n * 8 + 1          # 1 separator byte
    # ey is located after the TList metadata (empirically 57 bytes after y array ends)
    ey_off = y_off  + n * 8 + 57

    # Search for ey by scanning for a sequence of n small positive values
    for ey_try in range(y_off + n * 8, len(block) - n * 8):
        trial = _read_doubles(block, ey_try, n)
        if all(0 < v < 5 for v in trial):
            ey_off = ey_try
            break

    return {
        "x":  np.array(_read_doubles(block, x_off,  n)),
        "y":  np.array(_read_doubles(block, y_off,   n)),
        "ey": np.array(_read_doubles(block, ey_off,  n)),
    }


def read_tgraphasymerrors(block: bytes, n: int = 18) -> dict | None:
    """
    Parse a TGraphAsymmErrors block.
    Layout: [header] x[n] [1 sep] y[n] [TList ~57B] exl[n] [1 sep] exh[n] [1 sep] eyl[n] [1 sep] eyh[n]
    Returns dict with keys: x, y, exl, exh, eyl, eyh
    """
    x_off = _find_x_offset(block, n)
    if x_off is None:
        return None

    y_off   = x_off   + n * 8 + 1       # 1 byte gap after x
    exl_off = y_off   + n * 8 + 57      # TList metadata after y
    exh_off = exl_off + n * 8 + 1
    eyl_off = exh_off + n * 8 + 1
    eyh_off = eyl_off + n * 8 + 1

    if eyh_off + n * 8 > len(block):
        return None

    return {
        "x":   np.array(_read_doubles(block, x_off,   n)),
        "y":   np.array(_read_doubles(block, y_off,    n)),
        "exl": np.array(_read_doubles(block, exl_off,  n)),
        "exh": np.array(_read_doubles(block, exh_off,  n)),
        "eyl": np.array(_read_doubles(block, eyl_off,  n)),
        "eyh": np.array(_read_doubles(block, eyh_off,  n)),
    }


def _block_name(block: bytes) -> str:
    """Extract object name from a decompressed ROOT block."""
    text = block.decode("latin-1")
    m = re.search(r"(sv0pt|v0ptv0|v0pt|gr_sv0pt|gr_v0ptv0|gr_v0pt|gr_v0)[_\w]*", text)
    return m.group(0) if m else ""


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT ALL DATA
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all() -> dict:
    """
    Returns a nested dict:
        data[cent][observable] = {
            "pT":       array of pT bin centres (GeV/c) — cleaned
            "y":        array of observable values
            "stat":     statistical error
            "syst":     symmetric systematic error  = (eyl + eyh) / 2
            "total":    combined error sqrt(stat² + syst²)
        }
    Where observable ∈ {"sv0pt", "v0pt"}
    And   cent       ∈ {"5060", "6070"}
    """
    print("Reading main.root (statistical errors)...")
    main_blocks = _decompress_root(MAIN_ROOT)

    stat = {}   # stat[name] = {"x", "y", "ey"}
    for b in main_blocks:
        name = _block_name(b)
        if not name:
            continue
        r = read_tgrapherrors(b)
        if r is not None:
            stat[name] = r

    print(f"  Found {len(stat)} objects: {list(stat.keys())}")

    print("Reading SystUncs.root (systematic errors)...")
    syst_blocks = _decompress_root(SYST_ROOT)

    syst = {}   # syst[name] = {"x", "y", "exl", "exh", "eyl", "eyh"}
    for b in syst_blocks:
        name = _block_name(b)
        if not name or "_total" not in name:
            continue
        r = read_tgraphasymerrors(b)
        if r is not None:
            syst[name] = r

    print(f"  Found {len(syst)} _total objects: {list(syst.keys())}")

    # ── Combine into clean dataset ────────────────────────────────────────────
    # The x-errors in SystUncs.root encode the pT bin half-widths in log-space.
    # Convert to symmetric linear half-widths for display:
    #   x_centre ± x_err  (use exh as the representative half-width)
    #
    # ptref-1 objects are the ones with reference window 0.5–2.0 GeV/c.
    # We keep sv0pt (= v0(pT)/v0) and v0pt (= v0(pT)) for both centralities.

    data = {}
    for cent in ("5060", "6070"):
        data[cent] = {}
        for obs in ("sv0pt", "v0pt"):
            stat_key = f"{obs}_ptref_1_{cent}"
            syst_key = f"gr_{obs}_{cent}_total"

            if stat_key not in stat:
                print(f"  WARNING: {stat_key} not found in main.root")
                continue
            if syst_key not in syst:
                print(f"  WARNING: {syst_key} not found in SystUncs.root")
                continue

            s   = stat[stat_key]
            sy  = syst[syst_key]

            # pT bin centres: use values from SystUncs (same as main)
            pT_raw = sy["x"]

            # The x-errors in ROOT are stored as log-scale half-widths for
            # aesthetic reasons (equal visual width on log-x plots).
            # Convert to actual bin half-widths: dx = x * (exh / pT) ≈ exh
            # (exh is already in GeV/c — just use it directly as x-error)
            pT_exl = sy["exl"]   # lower x-error (GeV/c)
            pT_exh = sy["exh"]   # upper x-error (GeV/c)

            stat_err = s["ey"]                       # from main.root
            syst_err = (sy["eyl"] + sy["eyh"]) / 2  # symmetric average of asymm syst

            # Skip points with zero/denormal stat error (edge/underflow bins)
            mask = stat_err > 1e-10
            data[cent][obs] = {
                "pT":        pT_raw[mask],
                "pT_exl":    pT_exl[mask],
                "pT_exh":    pT_exh[mask],
                "y":         s["y"][mask],
                "stat":      stat_err[mask],
                "syst":      syst_err[mask],
                "total":     np.sqrt(stat_err[mask]**2 + syst_err[mask]**2),
            }
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE TO CSV
# ═══════════════════════════════════════════════════════════════════════════════

def save_csvs(data: dict):
    """Save each (centrality, observable) combination to a clean CSV file."""
    for cent, obs_dict in data.items():
        for obs, d in obs_dict.items():
            cent_label = CENTRALITIES[cent].replace("–", "-").replace("%", "")
            fname = CSV_DIR / f"CMS_PbPb2760_{obs}_ptref1_cent{cent}.csv"
            df = pd.DataFrame({
                "pT_GeVc":   d["pT"],
                "pT_err_lo": d["pT_exl"],
                "pT_err_hi": d["pT_exh"],
                f"{obs}":    d["y"],
                "stat_err":  d["stat"],
                "syst_err":  d["syst"],
                "total_err": d["total"],
            })
            df.to_csv(fname, index=False, float_format="%.8g")
            print(f"  Saved: {fname}  ({len(df)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {"5060": "#e07b39", "6070": "#4c72b0"}   # orange, blue
OBS_LABELS = {
    "sv0pt": r"$v_0(p_T)\,/\,v_0$",
    "v0pt":  r"$v_0(p_T)$",
}


def _draw_dataset(ax, d, color, label):
    """
    Draw one (centrality, observable) dataset:
      • Systematic error: grey shaded box
      • Statistical error: capped error bar
    """
    pT   = d["pT"]
    y    = d["y"]
    exl  = d["pT_exl"]
    exh  = d["pT_exh"]
    stat = d["stat"]
    syst = d["syst"]

    # Systematic uncertainty boxes
    for xi, yi, exli, exhi, systi in zip(pT, y, exl, exh, syst):
        rect = plt.Rectangle(
            (xi - exli, yi - systi),
            exli + exhi,
            2 * systi,
            linewidth=0.8,
            edgecolor=color,
            facecolor=color,
            alpha=0.25,
            zorder=3,
        )
        ax.add_patch(rect)

    # Statistical error bars
    ax.errorbar(
        pT, y,
        xerr=[exl, exh],
        yerr=stat,
        fmt="o",
        color=color,
        markersize=6,
        linewidth=1.4,
        capsize=3,
        capthick=1.2,
        zorder=5,
        label=label,
    )


def plot_sv0pt(data: dict):
    """v0(pT)/v0 for both centralities on one panel."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for cent, color in COLORS.items():
        if "sv0pt" not in data[cent]:
            continue
        _draw_dataset(ax, data[cent]["sv0pt"], color, CENTRALITIES[cent])

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_T$ (GeV/$c$)", fontsize=16)
    ax.set_ylabel(OBS_LABELS["sv0pt"], fontsize=16)
    ax.set_xlim(0.45, 12)
    ax.set_ylim(-6, 22)
    ax.legend(fontsize=13, title="Centrality", title_fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_title(r"CMS Pb–Pb $\sqrt{s_{NN}}=2.76$ TeV  —  $v_0(p_T)/v_0$, $p_T^{\rm ref}\in[0.5,2.0]$ GeV/$c$",
                 fontsize=12)

    fpath = PLOT_DIR / "sv0pt_both_centralities.pdf"
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {fpath}")
    plt.close(fig)


def plot_v0pt(data: dict):
    """v0(pT) for both centralities on one panel."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for cent, color in COLORS.items():
        if "v0pt" not in data[cent]:
            continue
        _draw_dataset(ax, data[cent]["v0pt"], color, CENTRALITIES[cent])

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_T$ (GeV/$c$)", fontsize=16)
    ax.set_ylabel(OBS_LABELS["v0pt"], fontsize=16)
    ax.set_xlim(0.45, 12)
    ax.legend(fontsize=13, title="Centrality", title_fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_title(r"CMS Pb–Pb $\sqrt{s_{NN}}=2.76$ TeV  —  $v_0(p_T)$, $p_T^{\rm ref}\in[0.5,2.0]$ GeV/$c$",
                 fontsize=12)

    fpath = PLOT_DIR / "v0pt_both_centralities.pdf"
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {fpath}")
    plt.close(fig)


def plot_two_panel(data: dict):
    """Side-by-side: v0(pT) and v0(pT)/v0."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, obs in zip(axes, ("v0pt", "sv0pt")):
        for cent, color in COLORS.items():
            if obs not in data[cent]:
                continue
            _draw_dataset(ax, data[cent][obs], color, CENTRALITIES[cent])

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_T$ (GeV/$c$)", fontsize=15)
        ax.set_ylabel(OBS_LABELS[obs], fontsize=15)
        ax.set_xlim(0.45, 12)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12, title="Centrality", title_fontsize=11)

    axes[0].set_ylim(-0.12, 0.45)
    axes[1].set_ylim(-6, 22)

    fig.suptitle(r"CMS Pb–Pb $\sqrt{s_{NN}}=2.76$ TeV  —  $p_T^{\rm ref}\in[0.5,2.0]$ GeV/$c$",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    fpath = PLOT_DIR / "v0pt_and_sv0pt_two_panel.pdf"
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {fpath}")
    plt.close(fig)


def plot_per_centrality(data: dict):
    """One figure per centrality with both observables stacked."""
    for cent, color in COLORS.items():
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        for ax, obs in zip(axes, ("v0pt", "sv0pt")):
            if obs not in data[cent]:
                continue
            _draw_dataset(ax, data[cent][obs], color, CENTRALITIES[cent])
            ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
            ax.set_xscale("log")
            ax.set_ylabel(OBS_LABELS[obs], fontsize=15)
            ax.set_xlim(0.45, 12)
            ax.tick_params(labelsize=12)

        axes[0].set_ylim(-0.12, 0.45)
        axes[1].set_ylim(-6, 22)
        axes[1].set_xlabel(r"$p_T$ (GeV/$c$)", fontsize=15)
        axes[0].set_title(
            rf"CMS Pb–Pb $\sqrt{{s_{{NN}}}}=2.76$ TeV  —  {CENTRALITIES[cent]}  —  $p_T^{{\rm ref}}\in[0.5,2.0]$ GeV/$c$",
            fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)

        fpath = PLOT_DIR / f"v0pt_sv0pt_cent{cent}.pdf"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {fpath}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  CMS PbPb 2.76 TeV  —  v0(pT) data extraction")
    print("=" * 65)

    # 1. Extract
    data = extract_all()

    # 2. Save CSVs
    print("\nSaving CSVs...")
    save_csvs(data)

    # 3. Print summary
    print("\nData summary:")
    for cent, obs_dict in data.items():
        for obs, d in obs_dict.items():
            print(f"  {obs} [{CENTRALITIES[cent]}]: {len(d['pT'])} points, "
                  f"pT = [{d['pT'][0]:.3f}, {d['pT'][-1]:.2f}] GeV/c")

    # 4. Plots
    print("\nCreating plots...")
    plot_sv0pt(data)
    plot_v0pt(data)
    plot_two_panel(data)
    plot_per_centrality(data)

    print("\n" + "=" * 65)
    print(f"  CSVs  → {CSV_DIR}/")
    print(f"  Plots → {PLOT_DIR}/")
    print("=" * 65 + "\n")
