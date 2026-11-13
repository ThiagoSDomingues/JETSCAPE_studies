"""Independent consistency checks of the digitised values (no input from the calibration itself)."""
import json, sys, numpy as np
d = sys.argv[1]
S = {p["id"]: p for p in json.load(open(f"{d}/STAR_2608.00190.json"))["panels"]}
C = {p["id"]: p for p in json.load(open(f"{d}/CMS-PAS-HIN-25-016.json"))["panels"]}
ser = lambda p, n: next(s for s in p["series"] if s["name"] == n)
xy = lambda s: (np.array([r["x"] for r in s["points"]]), np.array([r["y"] for r in s["points"]]))
out = []
out.append("STAR: v0 = v0(pT) [Fig 1b] / (v0(pT)/v0) [Fig 1c], compared with Fig 1a (Au+Au charged hadrons, ordered by Nch)")
a = sorted(ser(S["fig1a"], "Au+Au charged hadrons")["points"], key=lambda r: -r["x"])
derived = {"STAR": [], "CMS": []}
for i, s in enumerate(S["fig1b"]["series"]):
    xb, yb = xy(s); xc, yc = xy(ser(S["fig1c"], s["name"]))
    n = min(len(xb), len(xc)); m = (np.abs(yb[:n]) > 0.05) & (np.abs(xb[:n]/xc[:n]-1) < 1e-3)
    r = yb[:n][m]/yc[:n][m]
    out.append(f"  {s['name']:7s} v0(1b/1c) = {np.median(r):.4f} (spread {np.std(r):.4f}, {m.sum()} pts)   Fig 1a: v0 = {a[i]['y']:.4f} at Nch = {a[i]['x']:.0f}   diff = {100*(np.median(r)/a[i]['y']-1):+.1f}%")
    derived["STAR"].append(dict(system="Au+Au", centrality=s["name"], Nch=a[i]["x"], v0_fig1a=a[i]["y"], v0_from_ratio=float(np.median(r))))
out.append("")
out.append("CMS: <pT> = pT [Fig 1] / (pT/<pT>) [Fig 4]   and   v0 = v0(pT) [Fig 1] / (v0(pT)/v0) [Fig 4]")
v0 = {}
for sysname in ("pp", "pPb", "OO", "NeNe"):
    for s in C[f"fig1_{sysname}"]["series"]:
        x1, y1 = xy(s); x4, y4 = xy(ser(C[f"fig4_{sysname}"], s["name"]))
        n = min(len(x1), len(x4)); mp = x1[:n]/x4[:n]; m = np.abs(y1[:n]) > 0.05
        r = y1[:n][m]/y4[:n][m]
        v0[(sysname, s["name"])] = np.median(r)
        out.append(f"  {sysname:4s} {s['name']:13s} <pT> = {np.median(mp):.4f} GeV (spread {np.std(mp):.4f})   v0 = {np.median(r):.4f} (spread {np.std(r):.4f}, {m.sum()} pts)")
        derived["CMS"].append(dict(system=sysname, Ntrk=s["name"].replace("Ntrk ", ""), mean_pT=float(np.median(mp)), mean_pT_spread=float(np.std(mp)),
                                   v0=float(np.median(r)), v0_spread=float(np.std(r))))
out.append("")
out.append("CMS: Fig 6 [v0(pT) v0 1e3] vs Fig 1 [v0(pT)] x derived v0 x 1e3, interpolated in log pT")
for sysname in ("pp", "pPb"):
    for s in C[f"fig6_{sysname}"]["series"]:
        x6, y6 = xy(s); x1, y1 = xy(ser(C[f"fig1_{sysname}"], s["name"]))
        pred = np.interp(np.log(x6), np.log(x1), y1)*v0[(sysname, s["name"])]*1e3
        m = np.abs(y6) > 0.5
        out.append(f"  {sysname:4s} {s['name']:13s} median ratio Fig6/pred = {np.median(y6[m]/pred[m]):.3f} (spread {np.std(y6[m]/pred[m]):.3f}, {m.sum()} pts)")
out.append("")
out.append("CMS: Fig 5 vs Fig 4 (highest-multiplicity class), interpolated in log(pT/<pT>)")
for sysname in ("OO", "NeNe", "pPb", "pp"):
    s5 = C[f"fig5_{sysname}"]["series"][0]; x5, y5 = xy(s5); x4, y4 = xy(ser(C[f"fig4_{sysname}"], s5["name"]))
    pred = np.interp(np.log(x5), np.log(x4), y4); m = np.abs(y5) > 0.5
    out.append(f"  {sysname:4s} {s5['name']:13s} median ratio Fig5/Fig4 = {np.median(y5[m]/pred[m]):.3f} (spread {np.std(y5[m]/pred[m]):.3f}, {m.sum()} pts)")
open(f"{d}/crosschecks.txt", "w").write("\n".join(out)+"\n")
json.dump(derived, open(f"{d}/derived_v0_meanpT.json", "w"), indent=1)
print("\n".join(out))
