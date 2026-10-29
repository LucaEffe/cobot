#!/usr/bin/env python3
"""
extract_tcp.py -- the ONE rosbag -> TCP-path extractor (a consumer).

Reads the joint states from any recording rosbag (a teach bag OR an executed
episode bag), computes forward kinematics to gripper_tcp, and writes a CSV with
columns: run, segment, t, x, y, z, joint_0..6  (run/segment are 0 -- one
continuous path). Same format the planner/plot tools expect, so demo and
executed layers overlay with the planned CSVs.

Reuses cobot_teleop.config for the joint names, and the same MoveItPy forward
kinematics as the planner tools, so every layer uses identical FK.

Usage (workspace sourced; MoveIt bringup running so the robot model loads):
    python3 extract_tcp.py <bag_dir | recording_dir | episode_dir> <output.csv>

Examples:
    # demonstration layer
    python3 extract_tcp.py .../2026-09-24_11-07-11/teach \
        .../eval/results/pickplace/demo.csv
    # executed layer
    python3 extract_tcp.py .../2026-09-24_11-07-11/episodes/rrtstar_00 \
        .../eval/results/pickplace/executed_rrtstar.csv
"""

import glob
import os
import sys
import csv

import numpy as np
import rclpy
from rclpy.logging import get_logger
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState

import rosbag2_py

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from py_utils.planner_utils import wait_for_joint_states, generate_moveit_config

from cobot_teleop.config import ARM_JOINTS

TOOL_LINK = "gripper_tcp"
JS_TYPE = "sensor_msgs/msg/JointState"
STRIDE = 1   # keep every STRIDE-th joint-state sample


def find_bag_dir(path):
    """Accept a bag dir, a recording dir, or an episode dir."""
    cands = [
        path,
        os.path.join(path, "rosbag"),
        os.path.join(path, "teach", "rosbag"),
    ]
    for d in cands:
        if os.path.isdir(d) and (
            glob.glob(os.path.join(d, "*.mcap"))
            or glob.glob(os.path.join(d, "*.db3"))
        ):
            return d
    raise FileNotFoundError(f"No rosbag (*.mcap/*.db3) found near {path}")


def storage_id(bag_dir):
    return "mcap" if glob.glob(os.path.join(bag_dir, "*.mcap")) else "sqlite3"


def pick_js_topic(reader):
    topics = reader.get_all_topics_and_types()
    js = [t.name for t in topics if t.type == JS_TYPE]
    if not js:
        raise RuntimeError(
            f"No {JS_TYPE} topic in the bag. Topics: "
            f"{[(t.name, t.type) for t in topics]}"
        )
    # Prefer the dense OPC-UA stream (keeps publishing during motion) over the
    # ros2_control /joint_states (which freezes while the SPS executes a move).
    # Teach bags only have /joint_states (dense there, no SPS) -> used as well.
    for pref in ("/measured_joint_states", "/joint_states", "joint_states"):
        if pref in js:
            return pref
    return js[0]


def tcp_xyz(state):
    state.update()
    T = np.array(state.get_global_link_transform(TOOL_LINK))
    return T[0, 3], T[1, 3], T[2, 3]


def main():
    if len(sys.argv) < 3:
        print("usage: extract_tcp.py <bag|recording|episode dir> <output.csv>")
        return
    bag_dir = find_bag_dir(sys.argv[1])
    out_csv = sys.argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    rclpy.init()
    logger = get_logger("extract_tcp")

    logger.info("Waiting for joint states (robot model) ...")
    wait_for_joint_states(logger)

    logger.info("Starting MoveIt (for forward kinematics) ...")
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
    cobot = MoveItPy(node_name="extract_tcp", config_dict=cfg)
    robot_model = cobot.get_robot_model()

    logger.info(f"Reading bag: {bag_dir} ({storage_id(bag_dir)})")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id(bag_dir)),
        rosbag2_py.ConverterOptions("", ""),
    )
    js_topic = pick_js_topic(reader)
    logger.info(f"Using joint-state topic: {js_topic}")

    rows = []
    t0 = None
    i = 0
    kept = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != js_topic:
            continue
        i += 1
        if (i - 1) % STRIDE != 0:
            continue
        msg = deserialize_message(data, JointState)
        d = dict(zip(msg.name, msg.position))
        if not all(j in d for j in ARM_JOINTS):
            continue
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t0 is None:
            t0 = t
        st = RobotState(robot_model)
        st.joint_positions = {a: float(d[a]) for a in ARM_JOINTS}
        x, y, z = tcp_xyz(st)
        rows.append([0, 0, round(t - t0, 6), x, y, z]
                    + [float(d[a]) for a in ARM_JOINTS])
        kept += 1

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "segment", "t", "x", "y", "z"] + ARM_JOINTS)
        w.writerows(rows)
    logger.info(f"Wrote {kept} points to {out_csv}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()