"""vecdig.py - vector digitisation helpers for ROOT-made figures embedded in PDFs.

Markers, error bars and systematic boxes are read from the PDF drawing commands
(exact coordinates), not from pixels.  Axes are calibrated from tick marks and
their printed labels.
"""
import re, math, collections
import numpy as np
import pymupdf

def rc(c):
    return None if c is None else tuple(round(float(x), 2) for x in c)

def pts_of(item):
    k = item[0]
    if k == "l":  return [item[1], item[2]]
    if k == "c":  return [item[1], item[4]]          # on-curve end points only
    if k == "re":
        r = item[1]; return [pymupdf.Point(r.x0, r.y0), pymupdf.Point(r.x1, r.y0), pymupdf.Point(r.x1, r.y1), pymupdf.Point(r.x0, r.y1)]
    if k == "qu":
        q = item[1]; return [q.ul, q.ur, q.lr, q.ll]
    return []

def subpaths(d, tol=1e-3):
    """Split one drawing into connected sub-paths (one per marker)."""
    out, cur, last = [], [], None
    for it in d["items"]:
        if it[0] in ("re", "qu"):
            if cur: out.append(cur); cur = []
            out.append([it]); last = None; continue
        start, end = it[1], it[-1]
        if last is not None and (abs(start.x-last.x) > tol or abs(start.y-last.y) > tol):
            out.append(cur); cur = []
        cur.append(it); last = end
    if cur: out.append(cur)
    return out

def bbox(items):
    P = [p for it in items for p in pts_of(it)]
    xs = [p.x for p in P]; ys = [p.y for p in P]
    return min(xs), min(ys), max(xs), max(ys)

def centre(items):
    x0, y0, x1, y1 = bbox(items)
    return 0.5*(x0+x1), 0.5*(y0+y1)

class Page:
    def __init__(self, pdf, pno):
        self.page = pymupdf.open(pdf)[pno-1]
        self.dr = self.page.get_drawings()
        self.spans = []
        for b in self.page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    t = s["text"].strip()
                    if t: self.spans.append((t, pymupdf.Rect(s["bbox"]), s["size"]))
        # all black straight segments
        self.segs = []
        for d in self.dr:
            if d["type"] != "s" or rc(d.get("color")) != (0, 0, 0): continue
            if d.get("dashes") not in (None, "[] 0"): continue
            for it in d["items"]:
                if it[0] == "l":
                    self.segs.append((it[1].x, it[1].y, it[2].x, it[2].y))
                elif it[0] == "re":
                    r = it[1]
                    self.segs += [(r.x0, r.y0, r.x1, r.y0), (r.x0, r.y1, r.x1, r.y1), (r.x0, r.y0, r.x0, r.y1), (r.x1, r.y0, r.x1, r.y1)]

    def frames(self, minsize=40):
        H = collections.defaultdict(set); V = collections.defaultdict(set)
        for x0, y0, x1, y1 in self.segs:
            if abs(y0-y1) < 0.05 and abs(x1-x0) > minsize:
                H[(round(min(x0, x1), 1), round(max(x0, x1), 1))].add(round(y0, 1))
            if abs(x0-x1) < 0.05 and abs(y1-y0) > minsize:
                V[(round(min(y0, y1), 1), round(max(y0, y1), 1))].add(round(x0, 1))
        fr = set()
        for (xa, xb), ys in H.items():
            for (ya, yb), xs in V.items():
                if any(abs(y-ya) < 0.3 for y in ys) and any(abs(y-yb) < 0.3 for y in ys) \
                   and any(abs(x-xa) < 0.3 for x in xs) and any(abs(x-xb) < 0.3 for x in xs):
                    fr.add((xa, ya, xb, yb))
        return sorted(fr, key=lambda f: (round(f[1]/30), f[0]))

    def ticks(self, fr, maxlen=9):
        xa, ya, xb, yb = fr
        bx, ly = [], []
        for x0, y0, x1, y1 in self.segs:
            L = math.hypot(x1-x0, y1-y0)
            if L > maxlen or L < 0.3: continue
            if abs(x0-x1) < 0.05 and abs(max(y0, y1)-yb) < 0.3 and xa-0.3 <= x0 <= xb+0.3:
                bx.append((x0, L))
            if abs(y0-y1) < 0.05 and abs(min(x0, x1)-xa) < 0.3 and ya-0.3 <= y0 <= yb+0.3:
                ly.append((y0, L))
        ded = lambda a: sorted({(round(p, 2), round(L, 1)) for p, L in a})
        return ded(bx), ded(ly)

    def labels(self, fr, pad=16):
        xa, ya, xb, yb = fr
        xl, yl = [], []
        for t, r, sz in self.spans:
            cx, cy = 0.5*(r.x0+r.x1), 0.5*(r.y0+r.y1)
            if yb < cy < yb+pad and xa-8 < cx < xb+8: xl.append((round(cx, 1), t))
            if xa-28 < r.x1 <= xa+1 and ya-5 < cy < yb+5: yl.append((round(cy, 1), t))
        return sorted(xl), sorted(yl)

NUM = re.compile(r"^[−\-+]?\d*\.?\d+$")
def tonum(t):
    t = t.replace("−", "-")
    return float(t) if NUM.match(t) else None

class Axis:
    """Linear or log map fitted to (pdf coordinate, value) reference points."""
    def __init__(self, refs, log=False):
        self.log = log
        p = np.array([r[0] for r in refs], float); v = np.array([r[1] for r in refs], float)
        if log: v = np.log10(v)
        self.a, self.b = np.polyfit(p, v, 1)
        self.resid = float(np.max(np.abs(self.a*p+self.b-v)))
        self.n = len(refs)
    def __call__(self, p):
        v = self.a*np.asarray(p, float)+self.b
        return 10**v if self.log else v

def auto_axis(ticks, labels, log=False, tol=6.0):
    """Match numeric labels to the nearest major tick and fit the axis."""
    if not ticks: return None
    Lmax = max(L for _, L in ticks)
    major = [p for p, L in ticks if L > 0.75*Lmax]
    refs = []
    for pos, t in labels:
        v = tonum(t)
        if v is None or (log and v <= 0): continue
        j = min(major, key=lambda m: abs(m-pos))
        if abs(j-pos) < tol: refs.append((j, v))
    refs = sorted(set(refs))
    if len(refs) < 2: return None
    return Axis(refs, log)

# ----------------------------------------------------------------------------
# robust axis fit: largest consistent subset of (tick, label) pairs
# ----------------------------------------------------------------------------
def robust_axis(P, fr, which, log=False, tol=6.0, vals=None, use_minor=False, labfilter=None):
    bx, ly = P.ticks(fr)
    ticks = bx if which == "x" else ly
    if not ticks: return None
    Lmax = max(L for _, L in ticks)
    major = sorted(p for p, L in ticks if L > 0.75*Lmax)
    if vals is not None:                       # manual: values of the major ticks, in page order
        assert len(vals) == len(major), (which, fr, len(vals), major)
        return Axis(list(zip(major, vals)), log)
    cand = sorted(p for p, _ in ticks) if use_minor else major
    xa, ya, xb, yb = fr
    labs = []
    for t, r, sz in P.spans:
        cx, cy = 0.5*(r.x0+r.x1), 0.5*(r.y0+r.y1)
        v = tonum(t)
        if v is None: continue
        if which == "x":
            if not (yb < cy < yb+16 and xa-8 < cx < xb+8): continue
            pos = cx
        else:
            if not (xa-28 < r.x1 <= xa+1 and ya-5 < cy < yb+5): continue
            pos = cy
        for t2, r2, _ in P.spans:              # detached minus sign
            if t2 in ("−", "-") and abs(0.5*(r2.y0+r2.y1)-cy) < 2.5 and -8 < r2.x1-r.x0 < 1.5:
                v = -abs(v)
        if labfilter and not labfilter(pos, v): continue
        if log and v <= 0: continue
        labs.append((pos, v))
    pairs = []
    for pos, v in labs:
        j = min(cand, key=lambda m: abs(m-pos))
        if abs(j-pos) < tol: pairs.append((j, v))
    pairs = sorted(set(pairs))
    best = None
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            if abs(pairs[i][0]-pairs[j][0]) < 1: continue
            try: ax = Axis([pairs[i], pairs[j]], log)
            except Exception: continue
            f = (lambda v: np.log10(v)) if log else (lambda v: v)
            span = abs(ax.a)*(fr[2]-fr[0] if which == "x" else fr[3]-fr[1])
            inl = [p for p in pairs if abs(ax.a*p[0]+ax.b-f(p[1])) < 0.004*span]
            if best is None or len(inl) > len(best): best = inl
    if not best or len(best) < 2: return None
    ax = Axis(best, log); ax.npairs = len(pairs)
    return ax

# ----------------------------------------------------------------------------
# panel content
# ----------------------------------------------------------------------------
def _orient(items):
    P = {(round(p.x, 2), round(p.y, 2)) for it in items for p in pts_of(it)}
    ys = [p[1] for p in P]
    if len(P) != 3: return 0
    c = sum(ys)/3; m = 0.5*(min(ys)+max(ys))
    return 1 if c > m+0.05 else (-1 if c < m-0.05 else 0)      # +1: apex up (page y grows downwards)

def shape_of(items, kinds):
    if set(kinds) == {"c"}: return "circle"
    V = sorted({(round(p.x, 2), round(p.y, 2)) for it in items for p in pts_of(it)})
    if len(V) == 3: return "triangle-up" if _orient(items) > 0 else "triangle-down"
    if len(V) == 4:
        xs = [v[0] for v in V]; ys = [v[1] for v in V]
        corner = any(abs(v[0]-min(xs)) < 0.05 and abs(v[1]-min(ys)) < 0.05 for v in V)
        return "square" if corner else "diamond"
    return {5: "star", 8: "star4", 10: "star", 12: "cross"}.get(len(V), f"poly{len(V)}")

def collect(P, fr, msize_max=9.0):
    xa, ya, xb, yb = fr
    inside = lambda x, y, m=0.5: xa-m <= x <= xb+m and ya-m <= y <= yb+m
    markers = collections.defaultdict(list)     # signature -> [(cx, cy, half)]
    vseg = collections.defaultdict(list)        # colour -> [(x, y0, y1)]
    hseg = collections.defaultdict(list)        # colour -> [(y, x0, x1)]
    boxes = collections.defaultdict(list)       # colour -> [(x0, y0, x1, y1)]
    curves = []
    for d in P.dr:
        r = d["rect"]
        if not inside(0.5*(r.x0+r.x1), 0.5*(r.y0+r.y1), 2): continue
        typ, col, fill = d["type"], rc(d.get("color")), rc(d.get("fill"))
        if (col == (1, 1, 1) or col is None) and fill in ((1, 1, 1), None): continue
        op = d.get("fill_opacity"); dash = d.get("dashes") if d.get("dashes") not in (None, "[] 0") else ""
        sps = subpaths(d)
        big = []
        for sp in sps:
            x0, y0, x1, y1 = bbox(sp); w, h = x1-x0, y1-y0
            kinds = "".join(it[0][0] for it in sp)
            c = col if typ == "s" else fill
            # straight segments -> error bars
            if kinds == "l" and typ == "s" and not dash:
                if w < 0.06 and 0.05 < h < 80: vseg[c].append((x0, y0, y1))
                elif h < 0.06 and 0.05 < w < 40: hseg[c].append((y0, x0, x1))
                if max(w, h) > 12: big.append(sp)
                continue
            if kinds == "r" and typ == "f" and w < 0.06:          # zero-width filled rect = bar
                vseg[c].append((x0, y0, y1)); continue
            # rectangles that are (probably) systematic boxes
            if kinds in ("r", "q") and w < 40 and h < 80 and ((typ == "f" and op is not None and op < 0.99) or typ == "s"):
                boxes[c].append((x0, y0, x1, y1))
                if typ == "f": continue
            if max(w, h) <= msize_max and (len(sp) >= 2 or kinds in ("r", "q")) and max(w, h) > 0.8 \
               and not (typ == "f" and op is not None and op < 0.99):
                cx, cy = 0.5*(x0+x1), 0.5*(y0+y1)
                if not inside(cx, cy): continue
                sig = ("filled" if typ in ("f", "fs") else "open", c, shape_of(sp, kinds))
                markers[sig].append((cx, cy, w, h, len(sps)))
            elif max(w, h) > 12 or (dash and max(w, h) > 4):
                big.append(sp)
        if big and not (col == (0, 0, 0) and typ == "s" and not dash and all(len(sp) == 1 for sp in big)):
            for sp in big:
                x0, y0, x1, y1 = bbox(sp)
                if abs((x1-x0)-(xb-xa)) < 1 and abs((y1-y0)-(yb-ya)) < 1: continue      # the frame itself
                if len(sp) == 1 and sp[0][0] in ("re", "qu"): continue
                pts = []
                for it in sp:
                    if it[0] == "l": pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
                    elif it[0] == "c":
                        p1, p2, p3, p4 = it[1:5]
                        for t in np.linspace(0, 1, 9):
                            pts.append(((1-t)**3*p1.x+3*(1-t)**2*t*p2.x+3*(1-t)*t*t*p3.x+t**3*p4.x,
                                        (1-t)**3*p1.y+3*(1-t)**2*t*p2.y+3*(1-t)*t*t*p3.y+t**3*p4.y))
                q = [pts[0]] + [p for i, p in enumerate(pts[1:]) if abs(p[0]-pts[i][0])+abs(p[1]-pts[i][1]) > 1e-4]
                curves.append(dict(kind="band" if typ in ("f", "fs") else "line", color=col if typ == "s" else fill,
                                   dash=dash, opacity=op, pts=q))
    # de-duplicate markers drawn twice
    for sig, L in markers.items():
        seen, out = set(), []
        for m in L:
            k = (round(m[0], 1), round(m[1], 1))
            if k in seen: continue
            seen.add(k); out.append(m)
        markers[sig] = out
    for sig in list(markers):
        L = markers[sig]
        if sig[0] == "open" and sig[2] == "square" and len(L) > 2:
            dev = [max(abs(m[2]-np.median([q[2] for q in L])), abs(m[3]-np.median([q[3] for q in L]))) for m in L]
            if np.mean(np.array(dev) > 0.08) > 0.3: del markers[sig]
    return markers, vseg, hseg, boxes, curves

def legend_text(P, fr, x, y, s, reach=95, gap=14):
    xa, ya, xb, yb = fr
    got = []
    for t, r, sz in P.spans:
        cy = 0.5*(r.y0+r.y1)
        if abs(cy-y) < 4.6 and x+s+0.5 < r.x0 < x+reach and r.x0 < xb: got.append((r.x0, t, r.x1))
    got.sort()
    if not got or got[0][0]-(x+s) > gap: return None
    out, last = [], None
    for x0, t, x1 in got:
        if last is not None and x0-last > 12: break
        out.append(t); last = x1
    return " ".join(out)

def split_legend(P, fr, markers, mode="text"):
    """Separate legend samples from data points.
    mode="single": a legend sample is a drawing holding exactly one marker (ROOT draws each
                   series as one multi-marker path) -- used for the STAR figures.
    mode="text"  : a legend sample is a marker immediately followed by a text label."""
    data, legend = {}, {}
    for sig, L in markers.items():
        pts = []
        for cx, cy, w, h, nsub in L:
            s = 0.5*max(w, h)
            if mode == "single":
                isleg = nsub == 1
                t = legend_text(P, fr, cx, cy, s, gap=40) if isleg else None
            else:
                t = legend_text(P, fr, cx, cy, s)
                isleg = t is not None and len(t) >= 4 and not t.startswith("(")
            if isleg: legend.setdefault(sig, []).append((t, cx, cy))
            else: pts.append((cx, cy, s))
        if pts: data[sig] = sorted(pts)
    return data, legend

def attach_errors(pts, c, vseg, hseg, boxes):
    out = []
    for cx, cy, s in pts:
        up = dn = None
        for x, y0, y1 in vseg.get(c, []):
            if abs(x-cx) > 0.3: continue
            lo, hi = min(y0, y1), max(y0, y1)
            if lo <= cy <= hi or min(abs(lo-cy), abs(hi-cy)) <= s+0.9:
                if lo < cy: up = max(up or 0, cy-lo)
                if hi > cy: dn = max(dn or 0, hi-cy)
        left = right = None
        for y, x0, x1 in hseg.get(c, []):
            if abs(y-cy) > 0.3: continue
            lo, hi = min(x0, x1), max(x0, x1)
            if lo <= cx <= hi or min(abs(lo-cx), abs(hi-cx)) <= s+0.9:
                if lo < cx: left = max(left or 0, cx-lo)
                if hi > cx: right = max(right or 0, hi-cx)
        box = None
        for b in boxes.get(c, []):
            if abs((b[2]-b[0])-2*s) < 0.05 and abs((b[3]-b[1])-2*s) < 0.05 and abs(0.5*(b[0]+b[2])-cx) < 0.05: continue
            if b[0]-0.3 <= cx <= b[2]+0.3 and b[1]-0.3 <= cy <= b[3]+0.3:
                dist = abs(0.5*(b[0]+b[2])-cx)+abs(0.5*(b[1]+b[3])-cy)
                if box is None or dist < box[0]: box = (dist, b)
        out.append(dict(px=cx, py=cy, half=s, up=up, dn=dn, left=left, right=right, box=box[1] if box else None))
    return out


# ----------------------------------------------------------------------------
# full panel extraction
# ----------------------------------------------------------------------------
def extract_panel(P, fr, X, Y, mode="text", legend=None, legend_re=None):
    """Return (series, legend, curves) with everything converted to data units.
    series: {signature: [point dict, ...]}"""
    markers, vseg, hseg, boxes, curves = collect(P, fr)
    data, leg = split_legend(P, fr, markers, mode)
    if legend_re is not None:                      # only labels matching the regex are legend samples
        for sig in list(leg):
            keep = [l for l in leg[sig] if l[0] and re.search(legend_re, l[0])]
            back = [(l[1], l[2]) for l in leg[sig] if not (l[0] and re.search(legend_re, l[0]))]
            if back:
                half = {(round(m[0], 2), round(m[1], 2)): 0.5*max(m[2], m[3]) for m in markers[sig]}
                data[sig] = sorted(data.get(sig, []) + [(x, y, half[(round(x, 2), round(y, 2))]) for x, y in back])
            if keep: leg[sig] = keep
            else: del leg[sig]
    xa, ya, xb, yb = fr
    out = {}
    for sig, pts in data.items():
        rows = []
        for q in attach_errors(pts, sig[1], vseg, hseg, boxes):
            x, y = float(X(q["px"])), float(Y(q["py"]))
            r = dict(x=x, y=y, px=q["px"], py=q["py"],
                     stat_up=float(Y(q["py"]-q["up"])-y) if q["up"] else None,
                     stat_dn=float(y-Y(q["py"]+q["dn"])) if q["dn"] else None,
                     hidden_stat_max=float(Y(q["py"]-q["half"])-y),
                     syst_up=None, syst_dn=None, x_lo=None, x_hi=None, clipped=False)
            if q["box"]:
                b = q["box"]
                r.update(syst_up=float(Y(b[1])-y), syst_dn=float(y-Y(b[3])), x_lo=float(X(b[0])), x_hi=float(X(b[2])))
                if b[1] < ya+0.3 or b[3] > yb-0.3: r["clipped"] = True
            if q["up"] and q["py"]-q["up"] < ya+0.3: r["clipped"] = True
            if q["dn"] and q["py"]+q["dn"] > yb-0.3: r["clipped"] = True
            rows.append(r)
        out[sig] = sorted(rows, key=lambda r: r["x"])
    cv = []
    for c in curves:
        pts = c["pts"]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        horiz = max(ys)-min(ys) < 0.05
        if c["color"] == (0, 0, 0) and c["dash"] and horiz and len(pts) == 2 and max(xs)-min(xs) > 40:
            cv.append(dict(kind="zero-line", y=float(Y(ys[0])))); continue
        if len(pts) <= 4 and max(xs)-min(xs) < 30:          # legend sample of a line / band
            t = legend_text(P, fr, max(xs), 0.5*(min(ys)+max(ys)), 0, gap=14)
            cv.append(dict(kind="legend", of=c["kind"], color=c["color"], dash=c["dash"], label=t)); continue
        cv.append(dict(kind=c["kind"], color=c["color"], dash=c["dash"], opacity=c["opacity"],
                       x=[float(X(p[0])) for p in pts], y=[float(Y(p[1])) for p in pts]))
    return out, leg, cv
