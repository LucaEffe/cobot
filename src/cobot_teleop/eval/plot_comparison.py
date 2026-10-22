#!/usr/bin/env python3
"""
Plot and compare the planner CSVs written by planner_logger.py.

Reads   planner_comparison/<label>/tcp_path_*.csv  (+ meta.json)
Writes  into the same folder:
    comparison_3d.png    static 3D overlay of the TCP paths (matplotlib)
    metrics.csv          one row per planner (path length, duration, ...)
and prints the metrics table to the console.

Usage:
    python3 .../plot_comparison.py <label>
    python3 .../plot_comparison.py <label> --root /some/other/planner_comparison

No ROS needed -- pure offline analysis, only numpy + matplotlib. Run it with
the SYSTEM python (NOT the ros_venv), so matplotlib and numpy versions match:
    deactivate                 # leave the venv
    python3 .../plot_comparison.py <label>
"""

import os
import re
import sys
import csv
import json
import math

import numpy as np

ARM_JOINTS = ["joint_0", "joint_1", "joint_2", "joint_3",
              "joint_4", "joint_5", "joint_6"]
REVOLUTE = ARM_JOINTS[1:]          # joint_1..6 (rad); joint_0 is prismatic (m)

# stable colour per planner (hex); unknown names fall back to a cycle
PLANNER_COLORS = {
    "rrtconnect": "#1f77b4",
    "rrt":        "#ff7f0e",
    "rrtstar":    "#2ca02c",
    "prm":        "#d62728",
    "chomp":      "#9467bd",
    "stomp":      "#8c564b",
    "pilz_ptp":   "#e377c2",
}
FALLBACK_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def planner_from_filename(fname):
    m = re.match(r"tcp_path_(.+)\.csv$", fname)
    return m.group(1) if m else None


def load_csv(path):
    """Return dict: (run, segment) -> Nx(3+7) array [x,y,z, j0..j6], and t array."""
    segs = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (int(row["run"]), int(row["segment"]))
            rec = [float(row["x"]), float(row["y"]), float(row["z"]),
                   float(row["t"])] + [float(row[j]) for j in ARM_JOINTS]
            segs.setdefault(key, []).append(rec)
    out = {}
    for key, rows in segs.items():
        arr = np.array(rows, dtype=float)
        out[key] = {"xyz": arr[:, 0:3], "t": arr[:, 3], "joints": arr[:, 4:]}
    return out


def load_all(folder):
    planners = {}
    for fname in sorted(os.listdir(folder)):
        name = planner_from_filename(fname)
        if name is None:
            continue
        planners[name] = load_csv(os.path.join(folder, fname))
    meta = {}
    meta_path = os.path.join(folder, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return planners, meta


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def seg_path_length(xyz):
    if len(xyz) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def seg_direct_length(xyz):
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(xyz[-1] - xyz[0]))


def seg_deviation(xyz):
    """Distance of each point to the straight start->end line. Returns (mean,max)."""
    if len(xyz) < 3:
        return 0.0, 0.0
    S, E = xyz[0], xyz[-1]
    d = E - S
    L = np.linalg.norm(d)
    if L < 1e-9:
        return 0.0, 0.0
    d = d / L
    rel = xyz - S
    proj = rel @ d
    perp = rel - np.outer(proj, d)
    dist = np.linalg.norm(perp, axis=1)
    return float(np.mean(dist)), float(np.max(dist))


def seg_jerk_rms(xyz, t):
    """RMS magnitude of TCP jerk (3rd time derivative). Robust to uneven dt."""
    if len(xyz) < 4:
        return 0.0
    tt = np.array(t, dtype=float)
    # ensure strictly increasing time; fall back to uniform if not
    if np.any(np.diff(tt) <= 0):
        tt = np.arange(len(xyz), dtype=float)
    try:
        vel = np.gradient(xyz, tt, axis=0)
        acc = np.gradient(vel, tt, axis=0)
        jerk = np.gradient(acc, tt, axis=0)
    except Exception:
        return 0.0
    mag = np.linalg.norm(jerk, axis=1)
    return float(np.sqrt(np.mean(mag ** 2)))


def joint_travel(joints, idxs):
    if len(joints) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(joints[:, idxs], axis=0))))


def compute_metrics(segs, run=0):
    """Aggregate metrics over all segments of one run."""
    keys = sorted(k for k in segs if k[0] == run)
    path_len = direct_len = 0.0
    dev_means, dev_maxes, jerks = [], [], []
    rev_idx = [ARM_JOINTS.index(j) for j in REVOLUTE]
    rev_travel = 0.0
    lift_travel = 0.0
    n_points = 0
    duration = 0.0
    for k in keys:
        xyz = segs[k]["xyz"]
        t = segs[k]["t"]
        joints = segs[k]["joints"]
        n_points += len(xyz)
        duration = max(duration, float(t[-1]) if len(t) else duration)
        path_len += seg_path_length(xyz)
        direct_len += seg_direct_length(xyz)
        m, mx = seg_deviation(xyz)
        dev_means.append(m)
        dev_maxes.append(mx)
        jerks.append(seg_jerk_rms(xyz, t))
        rev_travel += joint_travel(joints, rev_idx)
        lift_travel += joint_travel(joints, [ARM_JOINTS.index("joint_0")])
    detour = (path_len / direct_len) if direct_len > 1e-9 else float("nan")
    return {
        "segments": ",".join(str(k[1]) for k in keys),
        "n_segments": len(keys),
        "points": n_points,
        "duration_s": round(duration, 4),
        "tcp_path_len_m": round(path_len, 5),
        "tcp_direct_len_m": round(direct_len, 5),
        "detour_ratio": round(detour, 4),
        "dev_mean_m": round(float(np.mean(dev_means)) if dev_means else 0.0, 5),
        "dev_max_m": round(float(np.max(dev_maxes)) if dev_maxes else 0.0, 5),
        "jerk_rms": round(float(np.mean(jerks)) if jerks else 0.0, 4),
        "joint_travel_rad": round(rev_travel, 5),
        "lift_travel_m": round(lift_travel, 5),
    }


# --------------------------------------------------------------------------
# keyframes (taught segment endpoints, planner-independent)
# --------------------------------------------------------------------------
def keyframe_points(planners, run=0):
    ref = None
    for name in ["rrtconnect", "rrt", "rrtstar", "prm", "stomp", "pilz_ptp"]:
        if name in planners:
            ref = planners[name]
            break
    if ref is None and planners:
        ref = next(iter(planners.values()))
    if ref is None:
        return np.empty((0, 3))
    pts = []
    for k in sorted(x for x in ref if x[0] == run):
        xyz = ref[k]["xyz"]
        if len(xyz):
            pts.append(xyz[0])
            pts.append(xyz[-1])
    if not pts:
        return np.empty((0, 3))
    pts = np.array(pts)
    # dedupe near-identical keyframes
    uniq = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < 1e-4 for q in uniq):
            uniq.append(p)
    return np.array(uniq)


def ordered_path(segs, run=0):
    keys = sorted(k for k in segs if k[0] == run)
    if not keys:
        return np.empty((0, 3))
    return np.vstack([segs[k]["xyz"] for k in keys])


def color_for(name, i):
    return PLANNER_COLORS.get(name, FALLBACK_CYCLE[i % len(FALLBACK_CYCLE)])


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot(planners, kf, folder, label):
    """Write comparison_3d.png: a static 3D overlay of the TCP paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed -> no plot; run with system python)")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i, (name, segs) in enumerate(planners.items()):
        path = ordered_path(segs)
        if len(path) == 0:
            continue
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                color=color_for(name, i), alpha=0.7, linewidth=1.8, label=name)
    if len(kf):
        ax.scatter(kf[:, 0], kf[:, 1], kf[:, 2], c="black", marker="D",
                   s=35, label="taught keyframes", depthshade=False)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"TCP paths per planner - {label}")
    ax.legend(loc="upper left", fontsize=8)
    try:
        ax.set_box_aspect(_equal_aspect(planners))
    except Exception:
        pass
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    out_png = os.path.join(folder, "comparison_3d.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}")


def _equal_aspect(planners):
    all_pts = [ordered_path(s) for s in planners.values()]
    all_pts = [p for p in all_pts if len(p)]
    if not all_pts:
        return (1, 1, 1)
    P = np.vstack(all_pts)
    spans = P.max(axis=0) - P.min(axis=0)
    spans[spans < 1e-6] = 1e-6
    return tuple(spans)


# --------------------------------------------------------------------------
# metrics output
# --------------------------------------------------------------------------
METRIC_COLS = [
    "planner", "n_segments", "points", "duration_s",
    "tcp_path_len_m", "detour_ratio", "dev_mean_m", "dev_max_m",
    "jerk_rms", "joint_travel_rad", "lift_travel_m",
]


def write_metrics(planners, out_csv):
    rows = []
    for name, segs in planners.items():
        m = compute_metrics(segs)
        m["planner"] = name
        rows.append(m)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out_csv}\n")
    # pretty console table
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in METRIC_COLS}
    header = "  ".join(c.ljust(widths[c]) for c in METRIC_COLS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in METRIC_COLS))
    return rows


# --------------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:]]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "planner_comparison")
    if "--root" in args:
        i = args.index("--root")
        root = args[i + 1]
        del args[i:i + 2]
    if not args:
        print("usage: plot_comparison.py <label> [--root <planner_comparison dir>]")
        return
    label = args[0]
    folder = os.path.join(root, label)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return

    planners, meta = load_all(folder)
    if not planners:
        print(f"No tcp_path_*.csv found in {folder}")
        return
    print(f"Loaded {len(planners)} planners from {folder}: {', '.join(planners)}\n")

    kf = keyframe_points(planners)
    plot(planners, kf, folder, label)
    write_metrics(planners, os.path.join(folder, "metrics.csv"))


if __name__ == "__main__":
    main()