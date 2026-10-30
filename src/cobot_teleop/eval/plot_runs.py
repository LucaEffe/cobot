#!/usr/bin/env python3
"""
plot_runs.py -- per-planner run overlay + success rates.

For a results/<label>/ produced by `planner_logger.py --runs N`, this makes ONE
plot per planner showing all its successful runs overlaid (so you see how
consistent or variable the planner is) together with the virtual barrier, plus
a success-rate table across all planners (e.g. 8/10 solved).

Reads   results/<label>/tcp_path_*.csv + meta.json
Writes  results/<label>/runs_<planner>.html   (interactive 3D, per planner)
        results/<label>/runs_<planner>.png    (3D + 3 projections, per planner)
        results/<label>/success.csv           (planner, attempts, successes, rate)

Usage:
    python3 .../plot_runs.py <label> [--root <dir>]

Run it in the ~/plotenv venv for the PNGs; the HTML needs plotly (ros_venv).
Reuses the barrier/colour helpers from plot_comparison.py (same folder).
"""

import os
import sys
import csv

import numpy as np

from plot_comparison import (
    load_all, barrier_from_meta, barrier_corners, _BOX_FACES,
    _convex_hull_2d, color_for, BARRIER_COLOR,
)


def runs_of(segs):
    return sorted({k[0] for k in segs})


def run_path(segs, run):
    keys = sorted(k for k in segs if k[0] == run)
    parts = [segs[k]["xyz"] for k in keys if len(segs[k]["xyz"])]
    return np.vstack(parts) if parts else np.empty((0, 3))


# --------------------------------------------------------------------------
def write_success(meta, out_csv):
    cols = ["planner", "attempts", "successes", "success_rate"]
    rows = []
    for name, s in meta.get("planners", {}).items():
        rows.append({
            "planner": name,
            "attempts": s.get("attempts", ""),
            "successes": s.get("successes", ""),
            "success_rate": s.get("success_rate", ""),
        })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out_csv}\n")
    width = max((len(r["planner"]) for r in rows), default=7)
    print(f"  {'planner'.ljust(width)}  solved   rate")
    print("  " + "-" * (width + 16))
    for r in rows:
        print(f"  {str(r['planner']).ljust(width)}  "
              f"{r['successes']}/{r['attempts']}   {r['success_rate']}")
    return rows


# --------------------------------------------------------------------------
def html_for_planner(name, segs, obs, folder, rate_str):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  (plotly not installed -> no HTML)")
        return
    fig = go.Figure()
    if obs is not None:
        c = barrier_corners(obs)
        tri = []
        for (a, b, d, e) in _BOX_FACES:
            tri += [(a, b, d), (a, d, e)]
        fig.add_trace(go.Mesh3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            i=[t[0] for t in tri], j=[t[1] for t in tri], k=[t[2] for t in tri],
            color=BARRIER_COLOR, opacity=0.35, flatshading=True,
            name="barrier", showlegend=True, hoverinfo="name"))
    col = color_for(name, 0)
    rlist = runs_of(segs)
    for r in rlist:
        p = run_path(segs, r)
        if len(p) == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
            line=dict(color=col, width=3), opacity=0.5,
            name="runs" if r == rlist[0] else None,
            showlegend=(r == rlist[0]), legendgroup="runs"))
    p0 = run_path(segs, rlist[0]) if rlist else np.empty((0, 3))
    if len(p0):
        ends = np.vstack([p0[0], p0[-1]])
        fig.add_trace(go.Scatter3d(
            x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="markers",
            marker=dict(size=5, color="black", symbol="diamond"),
            name="grasp / place"))
    fig.update_layout(
        title=f"{name} - {rate_str}",
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
                   aspectmode="data"))
    out = os.path.join(folder, f"runs_{name}.html")
    fig.write_html(out)
    print(f"  wrote {out}")


def png_for_planner(name, segs, obs, folder, rate_str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception:
        Poly3DCollection = None

    col = color_for(name, 0)
    rlist = runs_of(segs)

    fig = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axxy = fig.add_subplot(2, 2, 2)
    axxz = fig.add_subplot(2, 2, 3)
    axyz = fig.add_subplot(2, 2, 4)

    if obs is not None:
        cc = barrier_corners(obs)
        if Poly3DCollection is not None:
            faces = [[cc[idx] for idx in face] for face in _BOX_FACES]
            ax3d.add_collection3d(Poly3DCollection(
                faces, facecolor=BARRIER_COLOR, alpha=0.25,
                edgecolor=BARRIER_COLOR, linewidths=0.5))
        for ax, (a, b) in [(axxy, (0, 1)), (axxz, (0, 2)), (axyz, (1, 2))]:
            hull = _convex_hull_2d([(cc[r, a], cc[r, b]) for r in range(len(cc))])
            if len(hull) >= 3:
                ax.add_patch(mpatches.Polygon(hull, closed=True,
                             facecolor=BARRIER_COLOR, alpha=0.25,
                             edgecolor=BARRIER_COLOR, lw=0.8))

    for r in rlist:
        p = run_path(segs, r)
        if len(p) == 0:
            continue
        ax3d.plot(p[:, 0], p[:, 1], p[:, 2], color=col, alpha=0.5, lw=1.1)
        axxy.plot(p[:, 0], p[:, 1], color=col, alpha=0.5, lw=1.1)
        axxz.plot(p[:, 0], p[:, 2], color=col, alpha=0.5, lw=1.1)
        axyz.plot(p[:, 1], p[:, 2], color=col, alpha=0.5, lw=1.1)

    if len(rlist):
        p0 = run_path(segs, rlist[0])
        if len(p0):
            for ax, (a, b) in [(axxy, (0, 1)), (axxz, (0, 2)), (axyz, (1, 2))]:
                ax.scatter([p0[0, a], p0[-1, a]], [p0[0, b], p0[-1, b]],
                           facecolors="none", edgecolors="black", marker="o",
                           s=45, zorder=5)

    ax3d.set_title("3D view")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.view_init(elev=20, azim=-60)
    for ax, (la, lb), title in [
        (axxy, ("x [m]", "y [m]"), "top (x-y)"),
        (axxz, ("x [m]", "z [m]"), "front (x-z)"),
        (axyz, ("y [m]", "z [m]"), "side (y-z)"),
    ]:
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.set_title(title)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{name} - {rate_str} - {len(rlist)} run(s)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(folder, f"runs_{name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:]]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if "--root" in args:
        i = args.index("--root")
        root = args[i + 1]
        del args[i:i + 2]
    if not args:
        print("usage: plot_runs.py <label> [--root <dir>]")
        return
    label = args[0]
    folder = os.path.join(root, label)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return

    planners, meta = load_all(folder)
    obs = barrier_from_meta(meta)
    stats = meta.get("planners", {})

    write_success(meta, os.path.join(folder, "success.csv"))

    for name, segs in planners.items():
        s = stats.get(name, {})
        rate_str = f"{s.get('successes', '?')}/{s.get('attempts', '?')} solved"
        html_for_planner(name, segs, obs, folder, rate_str)
        try:
            png_for_planner(name, segs, obs, folder, rate_str)
        except Exception as e:
            print(f"  ({name}: PNG skipped -> {e})")


if __name__ == "__main__":
    main()