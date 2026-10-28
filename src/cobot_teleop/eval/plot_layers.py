#!/usr/bin/env python3
"""
Three-layer plot: demonstration vs planned vs executed (one planner).

Reads from results/<label>/:
    demo.csv                 the teleoperated demonstration (from the rosbag)
    planned_<planner>.csv    what MoveIt planned   (falls back to tcp_path_<planner>.csv)
    executed_<planner>.csv   what the robot drove  (/joint_states)

Writes into the same folder (each best-effort, depending on the environment):
    three_layers_<planner>.html   interactive, rotatable 3D (plotly; browser)
    three_layers_<planner>.png    static 3D + 2D projections + tracking error
                                  over time (matplotlib)

The HTML needs plotly, the PNG a working matplotlib -- you get whichever your
environment can produce (like plot_comparison.py).

Usage:
    python3 plot_layers.py <label> [--planner NAME] [--show]
"""

import os
import sys
import csv
import glob

import numpy as np


def load_path(path):
    """Return dict with xyz (Nx3), t (N), run (N). None if file missing."""
    if not os.path.isfile(path):
        return None
    xyz, t, run = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
            t.append(float(row["t"]))
            run.append(int(row.get("run", 0)))
    if not xyz:
        return None
    return {"xyz": np.array(xyz), "t": np.array(t), "run": np.array(run)}


def nearest_dist(points, ref):
    """Min distance from each point to the set of ref points (brute force)."""
    out = np.empty(len(points))
    for i, p in enumerate(points):
        out[i] = np.min(np.linalg.norm(ref - p, axis=1))
    return out


def pick_planner(folder, requested):
    if requested:
        return requested
    for pat in ("executed_*.csv", "planned_*.csv", "tcp_path_*.csv"):
        hits = sorted(glob.glob(os.path.join(folder, pat)))
        if hits:
            base = os.path.basename(hits[0])
            return base.split("_", 1)[1].rsplit(".", 1)[0] if "_" in base \
                else base.rsplit(".", 1)[0]
    return None


def write_layers_html(layers, folder, label, planner):
    """Best-effort interactive 3D HTML of the three layers (needs plotly)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  (plotly not installed -> no HTML; `pip install plotly` in this env)")
        return
    # Scatter3d lines have no dash, so layers differ by colour + width.
    style = {
        "demonstration": dict(color="#555555", width=6),
        "planned":       dict(color="#1f77b4", width=4),
        "executed":      dict(color="#d62728", width=3),
    }
    fig = go.Figure()
    for name in ["demonstration", "planned", "executed"]:
        d = layers.get(name)
        if d is None:
            continue
        p = d["xyz"]
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
            line=dict(color=style[name]["color"], width=style[name]["width"]),
            name=name))
    fig.update_layout(
        title=f"demonstration vs planned vs executed - {label} / {planner}",
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
                   aspectmode="data"))
    out = os.path.join(folder, f"three_layers_{planner}.html")
    fig.write_html(out)
    print(f"  wrote {out}  (open in a browser)")


def main():
    args = sys.argv[1:]
    show = False
    if "--show" in args:
        show = True
        args.remove("--show")
    planner = None
    if "--planner" in args:
        i = args.index("--planner")
        planner = args[i + 1]
        del args[i:i + 2]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results")
    if "--root" in args:
        i = args.index("--root")
        root = args[i + 1]
        del args[i:i + 2]
    if not args:
        print("usage: plot_layers.py <label> [--planner NAME] [--show]")
        return
    label = args[0]
    folder = os.path.join(root, label)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return

    planner = pick_planner(folder, planner)
    if planner is None:
        print(f"No planned/executed/tcp_path CSV found in {folder}")
        return
    print(f"Planner: {planner}")

    demo = load_path(os.path.join(folder, "demo.csv"))
    planned = load_path(os.path.join(folder, f"planned_{planner}.csv"))
    if planned is None:
        planned = load_path(os.path.join(folder, f"tcp_path_{planner}.csv"))
    executed = load_path(os.path.join(folder, f"executed_{planner}.csv"))

    layers = {"demonstration": demo, "planned": planned, "executed": executed}
    have = [k for k, v in layers.items() if v is not None]
    print("Layers found:", ", ".join(have) if have else "none")
    if not have:
        print("Nothing to plot.")
        return

    write_layers_html(layers, folder, label, planner)

    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib missing -> run in ~/plotenv")
        return
    try:
        _t = plt.figure(); _t.add_subplot(111, projection="3d"); plt.close(_t)
    except Exception as e:
        print(f"  matplotlib 3D not available here ({e}); run in ~/plotenv")
        return

    style = {
        "demonstration": dict(color="#555555", lw=2.4, ls="--", alpha=0.9),
        "planned":       dict(color="#1f77b4", lw=1.8, ls="-", alpha=0.9),
        "executed":      dict(color="#d62728", lw=1.0, ls="-", alpha=0.8),
    }

    fig = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axxy = fig.add_subplot(2, 2, 2)
    axxz = fig.add_subplot(2, 2, 3)
    axerr = fig.add_subplot(2, 2, 4)

    allpts = []
    for name in ["demonstration", "planned", "executed"]:
        d = layers[name]
        if d is None:
            continue
        p = d["xyz"]
        allpts.append(p)
        st = style[name]
        ax3d.plot(p[:, 0], p[:, 1], p[:, 2], label=name, **st)
        axxy.plot(p[:, 0], p[:, 1], **st)
        axxz.plot(p[:, 0], p[:, 2], **st)

    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title("3D")
    ax3d.legend(loc="upper left", fontsize=8)
    if allpts:
        P = np.vstack(allpts)
        spans = P.max(0) - P.min(0)
        spans[spans < 1e-6] = 1e-6
        try:
            ax3d.set_box_aspect(tuple(spans))
        except Exception:
            pass
    ax3d.view_init(elev=20, azim=-60)

    axxy.set_xlabel("x [m]"); axxy.set_ylabel("y [m]"); axxy.set_title("top (x-y)")
    axxy.set_aspect("equal", "box"); axxy.grid(True, alpha=0.3)
    axxz.set_xlabel("x [m]"); axxz.set_ylabel("z [m]"); axxz.set_title("front (x-z)")
    axxz.set_aspect("equal", "box"); axxz.grid(True, alpha=0.3)

    # tracking error over time: executed point -> nearest planned point
    if executed is not None and planned is not None:
        for r in sorted(set(executed["run"].tolist())):
            m = executed["run"] == r
            err = nearest_dist(executed["xyz"][m], planned["xyz"]) * 1000.0  # mm
            axerr.plot(executed["t"][m], err, lw=1.0,
                       label=f"run {r}" if len(set(executed['run'])) > 1 else None)
        axerr.set_xlabel("time [s]"); axerr.set_ylabel("deviation from plan [mm]")
        axerr.set_title("tracking error (executed vs planned)")
        axerr.grid(True, alpha=0.3)
        if len(set(executed["run"])) > 1:
            axerr.legend(fontsize=7)
    else:
        axerr.text(0.5, 0.5, "need planned + executed\nfor tracking error",
                   ha="center", va="center", transform=axerr.transAxes)
        axerr.set_axis_off()

    fig.suptitle(f"Demonstration vs planned vs executed - {label} / {planner}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(folder, f"three_layers_{planner}.png")
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()