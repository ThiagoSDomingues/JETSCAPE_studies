"""load_v0_data.py - convenience access to the digitised v0 / v0(pT) tables.

    from load_v0_data import load, series
    df = load("data")                                   # all points, STAR + CMS, one tidy DataFrame
    x, y, stat, syst = series(df, "CMS", "fig1_OO", "Ntrk 250-400")
"""
import pandas as pd, numpy as np, os

FILES = {"STAR": "STAR_2608.00190_points.csv", "CMS": "CMS-PAS-HIN-25-016_points.csv"}

def load(datadir="data"):
    out = []
    for exp, f in FILES.items():
        d = pd.read_csv(os.path.join(datadir, f)); d.insert(0, "experiment", exp); out.append(d)
    return pd.concat(out, ignore_index=True)

def series(df, experiment, figure, name):
    """x, y, stat (mean of up/down; NaN where the bar is hidden by the marker), syst."""
    d = df[(df.experiment == experiment) & (df.figure == figure) & (df.series == name)].sort_values("x")
    if d.empty: raise KeyError((experiment, figure, name))
    stat = d[["stat_up", "stat_dn"]].mean(axis=1).to_numpy()
    syst = d[["syst_up", "syst_dn"]].mean(axis=1).to_numpy()
    return d.x.to_numpy(), d.y.to_numpy(), stat, syst

if __name__ == "__main__":
    df = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    print(df.groupby(["experiment", "figure"]).agg(series=("series", "nunique"), points=("x", "size")).to_string())
