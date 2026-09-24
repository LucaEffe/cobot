"""
Central configuration for the teach / replay pipeline.

All constants live here so the nodes and the core logic share a single
source of truth. Values that make sense to change per run are also exposed
as ROS parameters by the nodes (see pedal_node.py / replay_node.py); the
values here are the defaults.
"""

# --- joints ---------------------------------------------------------------

# joint_0 is the prismatic height axis (metres), joint_1..6 are the
# revolute arm joints (radians).
ARM_JOINTS = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

ARM_GROUP = "arm_group"
GRIPPER_GROUP = "gripper_group"

# --- init pose ------------------------------------------------------------

# The canonical home pose. Every recording starts here and replay only
# runs when the robot is at this pose (safety gate).
INIT_JOINTS = {
    "joint_0": 0.7,
    "joint_1": 0.0,
    "joint_2": 0.0,
    "joint_3": 0.0,
    "joint_4": 0.0,
    "joint_5": 0.0,
    "joint_6": 0.0,
}

# joint_0 (height) is kept constant while teaching.
FIXED_JOINT0 = INIT_JOINTS["joint_0"]

# Tolerance for the "is the robot at init?" check
# (radians for the revolute joints, metres for joint_0).
DEFAULT_INIT_TOL = 0.05

# --- gripper --------------------------------------------------------------

# Recorded gripper states (what ends up in waypoints.jsonl).
GRIPPER_OPEN = "open"
GRIPPER_CLOSED = "closed"

# MoveIt SRDF configuration names used in simulation.
SIM_GRIPPER_OPEN_STATE = "open"
SIM_GRIPPER_CLOSE_STATE = "close"

# Package that provides the standalone OPC-UA commands
# (gripper_open / gripper_close / balancer) for the real robot.
HW_PACKAGE = "cobot_hardware"

# --- recording ------------------------------------------------------------

DEFAULT_DATA_ROOT = "/workspace/cobot_recordings"

DEFAULT_PEDAL_DEVICE = "/dev/input/event27"

# Topics recorded continuously by rosbag (the raw "archive" alongside the
# distilled waypoints.jsonl).
BAG_TOPICS = [
    "/joint_states",
    "/tf",
    "/tf_static",
    "/dynamic_joint_states",
    "/arm_group_controller/controller_state",
    "/camera/camera/color/image_raw",
    "/camera/camera/depth/color/points",
]
