"""extract_all.py - vector digitisation of
   * STAR  arXiv:2608.00190      (Figs. 1-3)
   * CMS   CMS-PAS-HIN-25-016    (Figs. 1-6)
Usage:  python3 extract_all.py STAR.pdf CMS.pdf OUTDIR
"""
import sys, json, re, csv, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vecdig import *

BLACK, BLUE, GREEN, ORANGE = (0, 0, 0), (0, 0, 1), (0, .6, 0), (1, .4, 0)

def ntrk(t):
    m = re.search(r"(\d+)\s*≤.*<\s*(\d+)", t or "")
    return f"Ntrk {m.group(1)}-{m.group(2)}" if m else t
def etagap(t):
    m = re.search(r"=\s*(\d)", t or ""); return f"eta_gap = {m.group(1)}" if m else t
def ptref(t):
    m = re.search(r"=\s*([\d.]+-[\d.]+)", t or ""); return f"pTref {m.group(1)} GeV" if m else t

STAR_SPECIES = {BLACK: "charged hadrons", BLUE: "p(pbar)", GREEN: "K+-", ORANGE: "pi+-"}

# Each panel: page, frame index (as returned by Page.frames()), identifiers, axis options.
#   xlog/ylog, xvals/yvals (manual values of the major ticks, top-to-bottom / left-to-right),
#   xminor (labels sit on minor ticks), legend_from (panel id whose legend applies), namer.
STAR = dict(tag="STAR_2608.00190", mode="single", panels=[
  dict(id="fig1a", page=3, fr=0, system="Au+Au & O+O", energy="200 GeV", sel="", xname="Nch", yname="v0",
       xlog=True, xvals=[100, 1000],
       namer=lambda sig, t: ("Au+Au " if sig[0] == "filled" else "O+O ")+STAR_SPECIES[sig[1]]),
  dict(id="fig1b", page=3, fr=1, system="Au+Au", energy="200 GeV", sel="charged hadrons", xname="pT [GeV/c]", yname="v0(pT)",
       xlog=True, legend_from="fig1c", namer=lambda sig, t: t,
       curves={("line", GREEN, ""): "O+O 10-20% (central)", ("band", GREEN, ""): "O+O 10-20% (band)"}),
  dict(id="fig1c", page=3, fr=2, system="Au+Au", energy="200 GeV", sel="charged hadrons", xname="pT [GeV/c]", yname="v0(pT)/v0",
       xlog=True, namer=lambda sig, t: t,
       curves={("line", GREEN, ""): "O+O 10-20% (central)", ("band", GREEN, ""): "O+O 10-20% (band)"}),
  dict(id="fig2a", page=3, fr=3, system="Au+Au", energy="200 GeV", sel="0-5%", xname="pT [GeV/c]", yname="v0(pT)",
       namer=lambda sig, t: STAR_SPECIES[sig[1]],
       curves={("band", c, ""): f"3DGlauber+MUSIC {n}" for c, n in STAR_SPECIES.items()}),
  dict(id="fig2b", page=3, fr=4, system="O+O", energy="200 GeV", sel="0-5%", xname="pT [GeV/c]", yname="v0(pT)",
       namer=lambda sig, t: STAR_SPECIES[sig[1]],
       curves={("band", c, ""): f"3DGlauber+MUSIC {n}" for c, n in STAR_SPECIES.items()}),
  dict(id="fig2c", page=3, fr=5, system="Au+Au", energy="200 GeV", sel="0-5%, pi+-", xname="pT/<pT>", yname="v0(pT)/v0",
       namer=lambda sig, t: "pi+-",
       curves={("band", (.8, 0, 0), ""): "TRENTo+MUSIC+SMASH ideal", ("band", (0, .8, 0), ""): "TRENTo+MUSIC+SMASH shear",
               ("band", (0, 0, .8), ""): "TRENTo+MUSIC+SMASH bulk+shear"}),
] + [
  dict(id=f"fig3{l}", page=4, fr=i, system="Au+Au", energy="BES", sel=f"0-40%, {sp}", xname="pT [GeV/c]", yname="v0(pT)",
       mode="text", legend_re=r"GeV$", legend_from="fig3d", namer=lambda sig, t: f"sqrt(sNN) = {t}")
  for i, (l, sp) in enumerate([("a", "p"), ("b", "pi+"), ("c", "pbar"), ("d", "pi-")])
])

SYS = [("pp", "13 TeV"), ("pPb", "8.16 TeV"), ("OO", "5.36 TeV"), ("NeNe", "5.36 TeV")]
F2 = ["10-20", "20-50", "50-80", "80-150"]; F3 = ["10-20", "20-50", "50-80", "80-100", "100-120"]
F5 = [("OO", "5.36 TeV", "250-400"), ("NeNe", "5.36 TeV", "400-600"), ("pPb", "8.16 TeV", "250-400"), ("pp", "13 TeV", "120-150")]
RED = (1, 0, 0)
CMS = dict(tag="CMS-PAS-HIN-25-016", mode="text", panels=[
  dict(id=f"fig1_{s}", page=7, fr=i, system=s, energy=e, sel="eta_gap=2, 0.3<pTref<3 GeV", xname="pT [GeV/c]", yname="v0(pT)",
       xlog=True, namer=lambda sig, t: ntrk(t)) for i, (s, e) in enumerate(SYS)
] + [
  dict(id=f"fig2_Ntrk{n}", page=8, fr=i, system="pp", energy="13 TeV", sel=f"Ntrk {n}, 0.3<pTref<3 GeV", xname="pT [GeV/c]", yname="v0(pT)",
       xlog=True, legend_from="fig2_Ntrk10-20", namer=lambda sig, t: etagap(t)) for i, n in enumerate(F2)
] + [
  dict(id=f"fig3_Ntrk{n}", page=8, fr=4+i, system="pp", energy="13 TeV", sel=f"Ntrk {n}, eta_gap=2", xname="pT [GeV/c]", yname="v0(pT)",
       xlog=True, legend_from="fig3_Ntrk10-20", namer=lambda sig, t: ptref(t)) for i, n in enumerate(F3)
] + [
  dict(id=f"fig4_{s}", page=9, fr=i, system=s, energy=e, sel="eta_gap=2, 0.3<pTref<3 GeV", xname="pT/<pT>", yname="v0(pT)/v0",
       xlog=True, namer=lambda sig, t: ntrk(t)) for i, (s, e) in enumerate(SYS)
] + [
  dict(id=f"fig5_{s}", page=10, fr=i, system=s, energy=e, sel=f"Ntrk {n}, eta_gap=2, 0.3<pTref<3 GeV", xname="pT/<pT>", yname="v0(pT)/v0",
       xlog=True, xminor=True, yvals=[12, 10, 8, 6, 4, 2, 0, -2], namer=lambda sig, t: ntrk(t),
       curves={("band", RED, ""): "TRENTo+MUSIC (0-5%) band", ("line", RED, ""): "TRENTo+MUSIC (0-5%) central",
               ("line", GREEN, "d"): "HIJING (150<=Ntrk<200)", ("line", BLUE, "d"): "PYTHIA8 (80<=Ntrk<150)"})
  for i, (s, e, n) in enumerate(F5)
] + [
  dict(id="fig6_pp", page=11, fr=0, system="pp", energy="13 TeV", sel="eta_gap=2, 0.3<pTref<3 GeV", xname="pT [GeV/c]", yname="v0(pT) x v0 x 1e3",
       xlog=True, xminor=True, yvals=[6, 4, 2, 0, -2], namer=lambda sig, t: ntrk(t), curves="legend"),
  dict(id="fig6_pPb", page=11, fr=1, system="pPb", energy="8.16 TeV", sel="eta_gap=2, 0.3<pTref<3 GeV", xname="pT [GeV/c]", yname="v0(pT) x v0 x 1e3",
       xlog=True, xminor=True, yvals=[10, 8, 6, 4, 2, 0, -2, -4], namer=lambda sig, t: ntrk(t), curves="legend"),
])

def run(pdf, cfg, log):
    pages, frames, axes, res, legends = {}, {}, {}, {}, {}
    for pn in cfg["panels"]:
        pg = pn["page"]
        if pg not in pages: pages[pg] = Page(pdf, pg); frames[pg] = pages[pg].frames()
        P, fr = pages[pg], frames[pg][pn["fr"]]
        X = robust_axis(P, fr, "x", pn.get("xlog", False), vals=pn.get("xvals"), use_minor=pn.get("xminor", False),
                        tol=3.0 if pn.get("xminor") else 6.0)
        Y = robust_axis(P, fr, "y", pn.get("ylog", False), vals=pn.get("yvals"))
        axes[pn["id"]] = [fr, X, Y]
    for pn in cfg["panels"]:                      # inherit axes from panels sharing the same row / column
        fr, X, Y = axes[pn["id"]]
        for q in cfg["panels"]:
            f2, X2, Y2 = axes[q["id"]]
            if q["page"] != pn["page"]: continue
            if X is None and X2 is not None and abs(f2[0]-fr[0]) < 0.3 and abs(f2[2]-fr[2]) < 0.3: X = X2; axes[pn["id"]][1] = X
            if Y is None and Y2 is not None and abs(f2[1]-fr[1]) < 0.3 and abs(f2[3]-fr[3]) < 0.3: Y = Y2; axes[pn["id"]][2] = Y
    for pn in cfg["panels"]:
        fr, X, Y = axes[pn["id"]]
        assert X is not None and Y is not None, pn["id"]
        series, leg, cv = extract_panel(pages[pn["page"]], fr, X, Y, pn.get("mode", cfg["mode"]), legend_re=pn.get("legend_re"))
        res[pn["id"]] = (series, cv); legends[pn["id"]] = leg
    out = []
    for pn in cfg["panels"]:
        fr, X, Y = axes[pn["id"]]
        series, cv = res[pn["id"]]
        leg = legends[pn.get("legend_from", pn["id"])]
        zero = [c["y"] for c in cv if c["kind"] == "zero-line"]
        log.append(f"{cfg['tag']} {pn['id']:16s} x: n_ref={X.n} resid={X.resid:.1e}{' (log)' if X.log else ''}   "
                   f"y: n_ref={Y.n} resid={Y.resid:.1e}   dashed zero line at y={zero[0]:+.4f}" if zero else
                   f"{cfg['tag']} {pn['id']:16s} x: n_ref={X.n} resid={X.resid:.1e}{' (log)' if X.log else ''}   y: n_ref={Y.n} resid={Y.resid:.1e}")
        S = []
        for sig, rows in series.items():
            t = leg.get(sig, [(None,)])[0][0]
            S.append(dict(name=pn["namer"](sig, t), marker=dict(fill=sig[0], color=list(sig[1]), shape=sig[2]), points=rows))
        cl = {}
        if pn.get("curves") == "legend":
            for c in cv:
                if c["kind"] == "legend" and c["label"]: cl[(c["of"], c["color"], c["dash"])] = c["label"]
        C = []
        for c in cv:
            if c["kind"] not in ("line", "band"): continue
            if pn.get("curves") == "legend":
                lab = cl.get((c["kind"], c["color"], c["dash"]))
                if lab: lab = ("PYTHIA, " if pn["system"] == "pp" else "HIJING, ")+ntrk(lab.replace("PYTHIA,", "").strip())
            else:
                lab = (pn.get("curves") or {}).get((c["kind"], c["color"], "d" if c["dash"] else ""))
            if lab: C.append(dict(name=lab, kind=c["kind"], color=list(c["color"]), dashed=bool(c["dash"]), x=c["x"], y=c["y"]))
        out.append(dict(id=pn["id"], page=pn["page"], system=pn["system"], energy=pn["energy"], selection=pn["sel"],
                        x_name=pn["xname"], y_name=pn["yname"], x_log=bool(pn.get("xlog")),
                        frame_pdf_pt=list(fr), x_range=[float(X(fr[0])), float(X(fr[2]))], y_range=[float(Y(fr[3])), float(Y(fr[1]))],
                        series=S, curves=C))
    return out

def write(tag, panels, outdir):
    os.makedirs(outdir, exist_ok=True)
    json.dump(dict(source=tag, method="vector extraction from the PDF drawing commands", panels=panels),
              open(f"{outdir}/{tag}.json", "w"), indent=1)
    cols = ["figure", "system", "energy", "selection", "series", "x_name", "y_name", "x", "y", "stat_up", "stat_dn",
            "syst_up", "syst_dn", "box_x_lo", "box_x_hi", "hidden_stat_max", "clipped"]
    with open(f"{outdir}/{tag}_points.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for p in panels:
            for s in p["series"]:
                for r in s["points"]:
                    g = lambda k: "" if r[k] is None else f"{r[k]:.6g}"
                    w.writerow([p["id"], p["system"], p["energy"], p["selection"], s["name"], p["x_name"], p["y_name"],
                                g("x"), g("y"), g("stat_up"), g("stat_dn"), g("syst_up"), g("syst_dn"), g("x_lo"), g("x_hi"),
                                g("hidden_stat_max"), int(r["clipped"])])
    with open(f"{outdir}/{tag}_curves.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["figure", "system", "curve", "kind", "point_index", "x", "y"])
        for p in panels:
            for c in p["curves"]:
                for i, (x, y) in enumerate(zip(c["x"], c["y"])):
                    w.writerow([p["id"], p["system"], c["name"], c["kind"], i, f"{x:.6g}", f"{y:.6g}"])

if __name__ == "__main__":
    star_pdf, cms_pdf, outdir = sys.argv[1:4]
    log = []
    for pdf, cfg in ((star_pdf, STAR), (cms_pdf, CMS)):
        panels = run(pdf, cfg, log)
        write(cfg["tag"], panels, outdir)
        for p in panels:
            print(f"{cfg['tag']} {p['id']:16s} x[{p['x_range'][0]:.3g},{p['x_range'][1]:.3g}] y[{p['y_range'][0]:.3g},{p['y_range'][1]:.3g}]  "
                  + "; ".join(f"{s['name']}: {len(s['points'])}" for s in p["series"])
                  + ("  | curves: " + ", ".join(c["name"] for c in p["curves"]) if p["curves"] else ""))
    open(f"{outdir}/calibration_log.txt", "w").write("\n".join(log)+"\n")
    print("\n".join(log))
