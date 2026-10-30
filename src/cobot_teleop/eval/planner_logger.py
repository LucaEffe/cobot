#!/usr/bin/env python3
"""
Planner comparison logger.

Replays a recorded teach route (keyframe -> keyframe) purely by PLANNING (no
execution) and writes the end-effector (gripper_tcp) path to a CSV -- once per
planner. One run of this script plans the whole route with EVERY planner in the
PLANNERS list below, so you don't have to edit + rerun for each one.

Optionally inserts a VIRTUAL barrier between the grasp and the place point (a
thin, fictitious wall that only exists in the planning scene). On an open route
every planner finds the same straight line, so their paths overlap; with the
barrier in the way each planner has to route OVER/AROUND it, and that is where
they start to differ -- which is the whole point of the comparison. The barrier
is virtual, so on the real robot nothing is physically there (the arm just arcs
higher). See cobot_teleop/scene.py.

Output goes to:   <this script's folder>/results/<label>/
                      tcp_path_rrtconnect.csv
                      tcp_path_rrtstar.csv
                      ...
                      meta.json

Usage (sim bringup running, workspace sourced):
    # CLEAN comparison: one move grasp->place over the barrier (recommended)
    python3 .../planner_logger.py <recording_dir> <label> --single-move

    # N attempts per planner (success rate + run-to-run variability)
    python3 .../planner_logger.py <recording_dir> <label> --single-move --runs 10

    # give every planner more time (do rrt/rrtstar/chomp find one then?)
    python3 .../planner_logger.py <recording_dir> <label> --single-move --runs 10 --planning-time 20

    # baseline: same single move, no barrier (planners collapse onto one line)
    python3 .../planner_logger.py <recording_dir> <label> --single-move --no-obstacle

    # whole route with the barrier (NOTE: the wall also interferes with the
    # approach/grasp/place segments, so some of those may fail -- use
    # --single-move for a clean planner comparison)
    python3 .../planner_logger.py <recording_dir> <label>

Barrier knobs (metres): --thickness --width --clearance --drop-below
Pick the two points yourself (waypoint indices) if auto-detect is wrong:
    --between <grasp_idx> <place_idx>

To add/remove a planner, edit the PLANNERS list below.
"""

import os
import sys
import csv
import json
import argparse
import datetime

import numpy as np
import rclpy
from rclpy.logging import get_logger

from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState

from py_utils.planner_utils import wait_for_joint_states, generate_moveit_config

from cobot_teleop.scene import add_barrier


# ===========================================================================
# Planners to run in one go. Comment out any you don't want.
#   name        -> used in the output file name (tcp_path_<name>.csv)
#   pipeline    -> ompl | chomp | stomp | pilz_industrial_motion_planner
#   planner_id  -> OMPL/Pilz need one; CHOMP/STOMP use ""
#   planning_time (optional) -> seconds; useful for the optimizing planners
# ===========================================================================
PLANNERS = [
    {"name": "rrtconnect", "pipeline": "ompl",  "planner_id": "RRTConnectkConfigDefault", "planning_time": 5.0},
    {"name": "rrt",        "pipeline": "ompl",  "planner_id": "RRTkConfigDefault", "planning_time": 5.0},
    {"name": "rrtstar",    "pipeline": "ompl",  "planner_id": "RRTstarkConfigDefault", "planning_time": 5.0},
    {"name": "prm",        "pipeline": "ompl",  "planner_id": "PRMkConfigDefault", "planning_time": 5.0},
    {"name": "chomp",      "pipeline": "chomp", "planner_id": ""},
    {"name": "stomp",      "pipeline": "stomp", "planner_id": ""},
    {"name": "pilz_ptp",   "pipeline": "pilz_industrial_motion_planner", "planner_id": "PTP"},
]

PLAN_CONFIG_NS = "ompl_rrtc"   # PlanRequestParameters namespace (defaults source)
# Outputs always land next to this script, in results/<label>/.
OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")
# ===========================================================================

ARM_JOINTS = ["joint_0", "joint_1", "joint_2", "joint_3",
              "joint_4", "joint_5", "joint_6"]
ARM_GROUP = "arm_group"
TOOL_LINK = "gripper_tcp"
SAME_POSE_TOL = 1e-3


def resolve_waypoints(arg):
    """Accept either a waypoints.jsonl or a recording dir (then look inside)."""
    if os.path.isdir(arg):
        cand = os.path.join(arg, "teach", "waypoints.jsonl")
        if os.path.isfile(cand):
            return cand
        cand = os.path.join(arg, "waypoints.jsonl")
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"No waypoints.jsonl found under {arg}")
    return arg


def load_waypoints(path):
    """Return a list of records: {joints, gripper, event}."""
    recs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wp = json.loads(line)
            j = wp["joints"]
            recs.append({
                "joints": {a: float(j[a]) for a in ARM_JOINTS},
                "gripper": wp.get("gripper"),
                "event": wp.get("event"),
            })
    return recs


def same_pose(a, b):
    return all(abs(a[j] - b[j]) <= SAME_POSE_TOL for j in ARM_JOINTS)


def tcp_xyz(state):
    state.update()
    T = np.array(state.get_global_link_transform(TOOL_LINK))
    return T[0, 3], T[1, 3], T[2, 3]


def tcp_of(robot_model, joints):
    st = RobotState(robot_model)
    st.joint_positions = joints
    st.update()
    return tcp_xyz(st)


def find_grasp_place(recs, logger):
    """Grasp = first waypoint where the gripper is closed; place = first
    waypoint after that where it is open again. Returns (grasp_idx, place_idx)
    or (None, None) if it can't be determined from the gripper states."""
    grasp = None
    for i, r in enumerate(recs):
        if grasp is None:
            if r.get("gripper") == "closed":
                grasp = i
        else:
            if r.get("gripper") == "open":
                return grasp, i
    return grasp, None


def plan_route(arm, robot_model, recs, params, planner, logger, run=0):
    """Plan the whole route once with one planner; return rows tagged with `run`."""
    rows = []  # run, segment, t, x, y, z, joint_0..6
    t_acc = 0.0
    for k in range(len(recs) - 1):
        start_j, goal_j = recs[k]["joints"], recs[k + 1]["joints"]
        if same_pose(start_j, goal_j):
            continue

        start = RobotState(robot_model)
        start.joint_positions = start_j
        start.update()
        goal = RobotState(robot_model)
        goal.joint_positions = goal_j
        goal.update()

        arm.set_start_state(robot_state=start)
        arm.set_goal_state(robot_state=goal)

        result = arm.plan(single_plan_parameters=params)
        if not result:
            logger.error(f"[{planner['name']}] run {run} segment {k}: planning FAILED")
            continue

        traj = result.trajectory
        durations = traj.get_waypoint_durations()
        n = len(traj)
        for i in range(n):
            if i == 0 and k > 0:
                continue
            state = traj[i]
            x, y, z = tcp_xyz(state)
            if i > 0:
                t_acc += float(durations[i])
            jp = state.joint_positions
            rows.append(
                [run, k, round(t_acc, 6), x, y, z]
                + [jp[a] for a in ARM_JOINTS]
            )
        logger.info(f"[{planner['name']}] run {run} segment {k}: {n} points")
    return rows


def plan_single(arm, robot_model, start_j, goal_j, params, planner, logger, run=0):
    """Plan ONE move start->goal once; return CSV rows tagged with `run`."""
    if same_pose(start_j, goal_j):
        logger.error(f"[{planner['name']}] grasp == place -- nothing to plan")
        return []
    start = RobotState(robot_model)
    start.joint_positions = start_j
    start.update()
    goal = RobotState(robot_model)
    goal.joint_positions = goal_j
    goal.update()
    arm.set_start_state(robot_state=start)
    arm.set_goal_state(robot_state=goal)
    result = arm.plan(single_plan_parameters=params)
    if not result:
        logger.error(f"[{planner['name']}] run {run}: planning FAILED")
        return []
    traj = result.trajectory
    durations = traj.get_waypoint_durations()
    rows = []
    t_acc = 0.0
    for i in range(len(traj)):
        state = traj[i]
        x, y, z = tcp_xyz(state)
        if i > 0:
            t_acc += float(durations[i])
        jp = state.joint_positions
        rows.append([run, 0, round(t_acc, 6), x, y, z]
                    + [jp[a] for a in ARM_JOINTS])
    logger.info(f"[{planner['name']}] run {run}: {len(traj)} points")
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "segment", "t", "x", "y", "z"] + ARM_JOINTS)
        w.writerows(rows)


def write_meta(out_dir, label, source, obstacle, stats):
    """Write meta.json with per-planner success stats for ALL attempted planners."""
    meta = {
        "label": label,
        "source_recording": source,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "obstacle": obstacle,
        "planners": stats,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def parse_args():
    ap = argparse.ArgumentParser(description="Plan a teach route with every planner.")
    ap.add_argument("waypoints", help="waypoints.jsonl or a recording dir")
    ap.add_argument("label", help="short name for results/<label>/")
    ap.add_argument("--obstacle", dest="obstacle", action="store_true", default=True,
                    help="insert the virtual barrier (default)")
    ap.add_argument("--no-obstacle", dest="obstacle", action="store_false",
                    help="plan without the barrier (baseline)")
    # Defaults tuned for the Festo cell: block the grasp->place line without the
    # wall touching the table (lin_axis) or the wrist (segment6/7) at the endpoints.
    ap.add_argument("--thickness", type=float, default=0.02, help="wall thickness along travel [m]")
    ap.add_argument("--width", type=float, default=0.12, help="wall width across travel [m]")
    ap.add_argument("--clearance", type=float, default=0.12, help="wall top above the line [m]")
    ap.add_argument("--drop-below", dest="drop_below", type=float, default=0.02,
                    help="how far the wall reaches below the line [m]")
    ap.add_argument("--between", nargs=2, type=int, default=None, metavar=("GRASP", "PLACE"),
                    help="waypoint indices for grasp/place (override auto-detect)")
    ap.add_argument("--single-move", dest="single_move", action="store_true",
                    help="plan ONE move grasp->place (clean obstacle comparison) "
                         "instead of the whole route")
    ap.add_argument("--runs", type=int, default=1,
                    help="plan each planner N times (success rate + variability)")
    ap.add_argument("--planning-time", dest="planning_time", type=float, default=None,
                    help="override planning time [s] for ALL planners")
    return ap.parse_args()


def main():
    args = parse_args()
    waypoints_path = resolve_waypoints(args.waypoints)
    label = args.label

    out_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)

    # Clear stale outputs for this label first, so a planner that FAILS this
    # run can't leave an old CSV/meta behind and get mixed into the comparison.
    for f in os.listdir(out_dir):
        if f.startswith("tcp_path_") and f.endswith(".csv"):
            os.remove(os.path.join(out_dir, f))
    _meta_old = os.path.join(out_dir, "meta.json")
    if os.path.isfile(_meta_old):
        os.remove(_meta_old)

    rclpy.init()
    logger = get_logger("planner_logger")

    recs = load_waypoints(waypoints_path)
    logger.info(f"Loaded {len(recs)} waypoints from {waypoints_path}")

    logger.info("Waiting for joint states ...")
    wait_for_joint_states(logger)

    logger.info("Starting MoveIt ...")
    cfg = generate_moveit_config()
    cfg["planning_scene_monitor_options"] = {
        "name": "planning_scene_monitor",
        "robot_description": "robot_description",
        "joint_state_topic": "/joint_states",
        "attached_collision_object_topic": "/moveit_cpp/planning_scene_monitor",
        "publish_planning_scene_topic": "/moveit_cpp/publish_planning_scene",
        "monitored_planning_scene_topic": "/moveit_cpp/monitored_planning_scene",
        "wait_for_initial_state_timeout": 10.0,
    }
    cobot = MoveItPy(node_name="planner_logger", config_dict=cfg)
    robot_model = cobot.get_robot_model()
    arm = cobot.get_planning_component(ARM_GROUP)

    # --- grasp / place (needed for the barrier and for --single-move) -----
    gi = pi = None
    if args.obstacle or args.single_move:
        if args.between is not None:
            gi, pi = args.between
        else:
            gi, pi = find_grasp_place(recs, logger)
        if gi is None or pi is None:
            logger.error(
                "Could not auto-detect grasp/place from gripper states "
                f"(got grasp={gi}, place={pi}). Pass --between <grasp> <place>, "
                "or use --no-obstacle. Gripper states per waypoint: "
                f"{[r.get('gripper') for r in recs]}")
            sys.stdout.flush()
            os._exit(1)

    # --- virtual barrier (so the planners actually differ) ----------------
    obstacle_meta = {"enabled": False}
    if args.obstacle:
        g = tcp_of(robot_model, recs[gi]["joints"])
        p = tcp_of(robot_model, recs[pi]["joints"])
        if args.single_move:
            # One direct move at the grasp/place height -> key the wall to that
            # height (there is no lift to account for).
            ref_top = None
        else:
            # Whole route lifts before carrying -> key the wall to the highest
            # waypoint between grasp and place (the transport height).
            ref_top = max(tcp_of(robot_model, recs[i]["joints"])[2]
                          for i in range(gi, pi + 1))
        pos, quat, dims = add_barrier(
            cobot, g, p, logger,
            thickness=args.thickness, width=args.width,
            clearance=args.clearance, drop_below=args.drop_below,
            ref_top_z=ref_top)
        logger.info(f"Barrier between waypoint {gi} {np.round(g, 3).tolist()} "
                    f"and waypoint {pi} {np.round(p, 3).tolist()}")
        obstacle_meta = {
            "enabled": True,
            "mode": "single_move" if args.single_move else "route",
            "grasp_idx": gi, "place_idx": pi,
            "grasp_xyz": [float(v) for v in g], "place_xyz": [float(v) for v in p],
            "position": [float(v) for v in pos],
            "dims": [float(v) for v in dims],
            "thickness": args.thickness, "width": args.width,
            "clearance": args.clearance, "drop_below": args.drop_below,
        }

    # run every planner, each `runs` times, one after another
    runs = max(1, args.runs)
    stats = {}
    for planner in PLANNERS:
        logger.info("=" * 60)
        logger.info(f"Planner: {planner['name']} "
                    f"(pipeline='{planner['pipeline']}' id='{planner['planner_id']}')")
        pt = args.planning_time if args.planning_time is not None else planner.get("planning_time")
        all_rows = []
        successes = 0
        points_per_run = []
        for run in range(runs):
            try:
                params = PlanRequestParameters(cobot, PLAN_CONFIG_NS)
                params.planning_pipeline = planner["pipeline"]
                params.planner_id = planner["planner_id"]
                if pt is not None:
                    params.planning_time = float(pt)
                if args.single_move:
                    rows = plan_single(arm, robot_model, recs[gi]["joints"],
                                       recs[pi]["joints"], params, planner, logger, run)
                else:
                    rows = plan_route(arm, robot_model, recs, params, planner, logger, run)
            except Exception as e:  # one bad attempt must not kill the rest
                logger.error(f"[{planner['name']}] run {run} skipped: {e}")
                rows = []
            if rows:
                all_rows.extend(rows)
                successes += 1
                points_per_run.append(len(rows))

        stats[planner["name"]] = {
            "pipeline": planner["pipeline"],
            "planner_id": planner["planner_id"],
            "planning_time": pt,
            "attempts": runs,
            "successes": successes,
            "success_rate": round(successes / runs, 3),
            "points_per_run": points_per_run,
            "csv": f"tcp_path_{planner['name']}.csv" if all_rows else None,
        }
        logger.info(f"[{planner['name']}] solved {successes}/{runs}")
        if all_rows:
            write_csv(os.path.join(out_dir, f"tcp_path_{planner['name']}.csv"), all_rows)

    write_meta(out_dir, label, waypoints_path, obstacle_meta, stats)
    logger.info("=" * 60)
    logger.info(f"Done. Results in {out_dir}")
    logger.info("Success rates:")
    for name, s in stats.items():
        logger.info(f"  {name:12s} {s['successes']}/{s['attempts']}")

    # Skip MoveItPy/rclpy teardown (it segfaults on MoveItCpp deletion). All
    # files are already written and closed, so a hard exit is safe and clean.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()