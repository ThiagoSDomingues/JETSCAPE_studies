"""plot_all.py - redraw every digitised figure from the JSON files written by extract_all.py.
Usage:  python3 plot_all.py DATADIR OUTDIR
"""
import sys, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.9,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "xtick.minor.visible": True, "ytick.minor.visible": True, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
MK = {"circle": "o", "square": "s", "triangle-up": "^", "triangle-down": "v", "diamond": "D", "star": "*", "star4": "*", "cross": "P"}
TEX = {"v0(pT)": r"$v_0(p_T)$", "v0(pT)/v0": r"$v_0(p_T)/v_0$", "v0": r"$v_0$", "Nch": r"$N_{\rm ch}$",
       "pT [GeV/c]": r"$p_T$ (GeV/$c$)", "pT/<pT>": r"$p_T/\langle p_T\rangle$", "v0(pT) x v0 x 1e3": r"$v_0(p_T)\times v_0\times10^{3}$"}

def draw(ax, p, legend=True, title=None, ms=4.0, fs=7, leg_kw=None, curve_labels=True):
    for c in p["curves"]:
        col = c["color"]
        if c["kind"] == "band":
            ax.fill(c["x"], c["y"], color=col, alpha=0.30, lw=0, label=c["name"] if curve_labels else None, zorder=1)
        else:
            ax.plot(c["x"], c["y"], color=col, lw=1.1, ls="--" if c["dashed"] else "-", label=c["name"] if curve_labels else None, zorder=2)
    for s in p["series"]:
        col = s["marker"]["color"]; R = s["points"]
        x = np.array([r["x"] for r in R]); y = np.array([r["y"] for r in R])
        for r in R:                                    # systematic boxes
            if r["syst_up"] is not None:
                ax.add_patch(Rectangle((r["x_lo"], r["y"]-r["syst_dn"]), r["x_hi"]-r["x_lo"], r["syst_up"]+r["syst_dn"],
                                       facecolor=col, edgecolor="none", alpha=0.25, zorder=3))
        up = np.array([r["stat_up"] or 0 for r in R]); dn = np.array([r["stat_dn"] or 0 for r in R])
        ax.errorbar(x, y, yerr=[dn, up], fmt="none", ecolor=col, elinewidth=0.8, zorder=4)
        m = MK.get(s["marker"]["shape"], "o")
        ax.plot(x, y, ls="none", marker=m, ms=ms*(1.35 if m == "*" else 1), mec=col, mew=0.7,
                mfc=col if s["marker"]["fill"] == "filled" else "none", label=s["name"], zorder=5)
    ax.axhline(0, color="k", lw=0.6, ls="--", zorder=0)
    if p["x_log"]:
        ax.set_xscale("log"); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        short = p["x_range"][1]/p["x_range"][0] < 30
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if short and round(v, 3) in (0.4, 2, 3) else ""))
    ax.set_xlim(*p["x_range"]); ax.set_ylim(*p["y_range"])
    ax.set_xlabel(TEX.get(p["x_name"], p["x_name"]))
    if title: ax.set_title(title, fontsize=8.5)
    if legend: ax.legend(fontsize=fs, loc="upper left", handletextpad=0.3, labelspacing=0.25, borderaxespad=0.6, **(leg_kw or {}))

def row(P, ids, titles, suptitle, w=3.0, h=3.6, sharey=True, **kw):
    fig, axes = plt.subplots(1, len(ids), figsize=(w*len(ids)+0.7, h), sharey=sharey, squeeze=False)
    for i, (ax, k) in enumerate(zip(axes[0], ids)):
        draw(ax, P[k], title=titles[i], **kw)
        if i == 0 or not sharey: ax.set_ylabel(TEX.get(P[k]["y_name"], P[k]["y_name"]), fontsize=11)
    fig.suptitle(suptitle, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if sharey: fig.subplots_adjust(wspace=0)
    return fig

def grid(P, ids, titles, suptitle, **kw):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    for ax, k, t in zip(axes.flat, ids, titles):
        draw(ax, P[k], title=t, **kw); ax.set_ylabel(TEX.get(P[k]["y_name"], P[k]["y_name"]), fontsize=11)
    fig.suptitle(suptitle, fontsize=10, x=0.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig

def main(datadir, outdir):
    os.makedirs(outdir, exist_ok=True)
    S = {p["id"]: p for p in json.load(open(f"{datadir}/STAR_2608.00190.json"))["panels"]}
    C = {p["id"]: p for p in json.load(open(f"{datadir}/CMS-PAS-HIN-25-016.json"))["panels"]}
    src_s = "Digitised from STAR, arXiv:2608.00190 (STAR Preliminary)"
    src_c = "Digitised from CMS-PAS-HIN-25-016 (CMS Preliminary)"
    figs = {}
    figs["STAR_fig1"] = row(S, ["fig1a", "fig1b", "fig1c"],
        [r"(a) Au+Au (filled), O+O (open), $\sqrt{s_{NN}}$ = 200 GeV", "(b) Au+Au 200 GeV, charged hadrons", "(c) Au+Au 200 GeV, charged hadrons"],
        src_s+" - Fig. 1", w=3.6, h=4.0, sharey=False, fs=6.5, ms=3.5, leg_kw=dict(ncol=1))
    figs["STAR_fig1"].axes[0].legend(fontsize=6.5, loc="upper right", ncol=2, columnspacing=0.8, handletextpad=0.2)
    figs["STAR_fig2"] = row(S, ["fig2a", "fig2b", "fig2c"],
        ["(a) Au+Au 200 GeV, 0-5%", "(b) O+O 200 GeV, 0-5%", r"(c) Au+Au 200 GeV, 0-5%, $\pi^\pm$"],
        src_s+" - Fig. 2", w=3.6, h=4.0, sharey=False, fs=6.5, ms=3.5)
    figs["STAR_fig3"] = grid(S, ["fig3a", "fig3b", "fig3c", "fig3d"],
        [r"(a) $p$", r"(b) $\pi^+$", r"(c) $\bar p$", r"(d) $\pi^-$"], src_s+" - Fig. 3: Au+Au, 0-40%", fs=7.5, ms=5)
    sysl = {"pp": "pp 13 TeV", "pPb": "pPb 8.16 TeV", "OO": "OO 5.36 TeV", "NeNe": "NeNe 5.36 TeV"}
    order = ["pp", "pPb", "OO", "NeNe"]
    figs["CMS_fig1"] = row(C, [f"fig1_{s}" for s in order], [sysl[s] for s in order],
        src_c+r" - Fig. 1 ($\eta_{\rm gap}$ = 2, 0.3 < $p_T^{\rm ref}$ < 3 GeV)", ms=3, fs=6.5)
    n2 = ["10-20", "20-50", "50-80", "80-150"]
    figs["CMS_fig2"] = row(C, [f"fig2_Ntrk{n}" for n in n2], [rf"pp 13 TeV, {n.replace('-', r' $\leq N_{{\rm trk}}^{{\rm offline}}<$ ')}" for n in n2],
        src_c+r" - Fig. 2 ($\eta_{\rm gap}$ dependence)", ms=3.5, fs=7)
    n3 = ["10-20", "20-50", "50-80", "80-100", "100-120"]
    figs["CMS_fig3"] = row(C, [f"fig3_Ntrk{n}" for n in n3], [rf"pp, {n.replace('-', r' $\leq N_{{\rm trk}}<$ ')}" for n in n3],
        src_c+r" - Fig. 3 ($p_T^{\rm ref}$ dependence)", w=2.6, ms=3.5, fs=7)
    figs["CMS_fig4"] = row(C, [f"fig4_{s}" for s in order], [sysl[s] for s in order],
        src_c+r" - Fig. 4 ($\eta_{\rm gap}$ = 2, 0.3 < $p_T^{\rm ref}$ < 3 GeV)", ms=3, fs=6.5)
    o5 = ["OO", "NeNe", "pPb", "pp"]
    figs["CMS_fig5"] = grid(C, [f"fig5_{s}" for s in o5], [sysl[s] for s in o5], src_c+" - Fig. 5 (highest multiplicity interval)", fs=7.5, ms=4)
    figs["CMS_fig6"] = row(C, ["fig6_pp", "fig6_pPb"], [sysl["pp"], sysl["pPb"]], src_c+" - Fig. 6", w=4.2, h=4.6, sharey=False, fs=7.5, ms=4.5)
    for tag in ("STAR", "CMS"):
        with PdfPages(f"{outdir}/{tag}_all_figures.pdf") as pdf:
            for k, f in figs.items():
                if k.startswith(tag):
                    f.savefig(f"{outdir}/{k}.pdf", bbox_inches="tight"); f.savefig(f"{outdir}/{k}.png", dpi=110, bbox_inches="tight")
                    pdf.savefig(f, bbox_inches="tight")
    print("wrote", sorted(os.listdir(outdir)))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
