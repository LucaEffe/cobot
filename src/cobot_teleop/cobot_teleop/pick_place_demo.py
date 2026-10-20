#!/usr/bin/env python3
"""
Minimal scripted pick-and-place demo (NO MoveIt Task Constructor).

Three interfaces:
  1. Planning Scene   -> spawn a cube as a collision object (RViz draws it).
  2. Motion Planning  -> move the tool link (gripper_tcp) to Cartesian poses.
  3. Attach / Detach   -> carry the cube with the gripper.

Robot-specific:
  * The arm drives ONCE to the config init pose (joint_0 = 0.3), then joint_0
    is LOCKED via a path constraint -- the height axis never moves again.
  * There is no physics here. "Grasping" is modelled by (a) allowing the
    gripper's collision shapes to overlap the cube and (b) attaching the cube
    to the tool link. The cube stays visible the whole time.

Run (sim + RViz up, workspace sourced):
    python3 src/cobot_teleop/cobot_teleop/pick_place_demo.py
"""

import time

import numpy as np
import rclpy
from rclpy.logging import get_logger

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    CollisionObject,
    AttachedCollisionObject,
    ObjectColor,
    Constraints,
    JointConstraint,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA

from py_utils.planner_utils import wait_for_joint_states, generate_moveit_config


# --- robot constants (match config.py / the SRDF) -------------------------

ARM_GROUP = "arm_group"
GRIPPER_GROUP = "gripper_group"
TOOL_LINK = "gripper_tcp"
BASE_FRAME = "world"
HEIGHT_JOINT = "joint_0"

GRIPPER_OPEN_STATE = "open"
GRIPPER_CLOSE_STATE = "close"

# The whole gripper head that may touch the cube -- INCLUDING the vacuum
# links, which stick out and otherwise collide with the cube.
GRIPPER_TOUCH_LINKS = [
    "finger1", "finger2", "gripper_tcp", "segment7",
    "vacuum_gripper_up", "vacuum_gripper_down",
]

INIT_JOINTS = {
    "joint_0": 0.3,
    "joint_1": 0.0,
    "joint_2": 0.0,
    "joint_3": 0.0,
    "joint_4": 0.0,
    "joint_5": 0.0,
    "joint_6": 0.0,
}

CUBE_ID = "cube"

# --- demo geometry (metres), relative to the tool pose at init ------------

CUBE_SIZE = 0.04
PICK_DROP = 0.10
LIFT = 0.10
PLACE_SHIFT = (0.0, 0.15, 0.0)


# ---------------------------------------------------------------------------
# rotation matrix -> quaternion (x, y, z, w)
# ---------------------------------------------------------------------------

def rotmat_to_quat(R):
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif m00 >= m11 and m00 >= m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 >= m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


# ---------------------------------------------------------------------------
# generic plan + execute
# ---------------------------------------------------------------------------

def plan_and_execute(cobot, component, label, logger):
    logger.info(f"Planning: {label}")
    try:
        plan_result = component.plan()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Planning exception for '{label}': {e}")
        return False
    if not plan_result:
        logger.error(f"Planning FAILED: {label}")
        return False
    logger.info(f"Executing: {label}")
    try:
        ok = cobot.execute(plan_result.trajectory, controllers=[])
    except Exception as e:  # noqa: BLE001
        logger.error(f"Execution exception for '{label}': {e}")
        return False
    if not ok:
        logger.error(f"Execution FAILED: {label}")
        return False
    logger.info(f"Execution finished: {label}")
    time.sleep(0.5)
    return True


# ---------------------------------------------------------------------------
# INTERFACE 1: planning scene
# ---------------------------------------------------------------------------

def make_box(obj_id, position, size, frame, quat=(0.0, 0.0, 0.0, 1.0)):
    obj = CollisionObject()
    obj.header.frame_id = frame
    obj.id = obj_id
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [float(size[0]), float(size[1]), float(size[2])]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
    obj.primitives.append(box)
    obj.primitive_poses.append(pose)
    obj.operation = CollisionObject.ADD
    return obj


def color_for(obj_id, rgba):
    c = ObjectColor()
    c.id = obj_id
    c.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
    return c


def _allow_cube_touch(scene):
    """Tell MoveIt the cube may overlap the gripper head (no physics here)."""
    acm = scene.allowed_collision_matrix
    for link in GRIPPER_TOUCH_LINKS:
        acm.set_entry(CUBE_ID, link, True)


def print_scene_objects(cobot, logger):
    psm = cobot.get_planning_scene_monitor()
    with psm.read_only() as scene:
        objs = scene.planning_scene_message.world.collision_objects
    if not objs:
        logger.info("World collision objects: none.")
        return
    logger.info("World collision objects:")
    for o in objs:
        if o.primitive_poses:
            p = o.primitive_poses[0].position
            logger.info(f"  '{o.id}'  pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
        else:
            logger.info(f"  '{o.id}'  (mesh / no primitive pose)")


def spawn_cube(cobot, logger, cube_pos):
    psm = cobot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        cube = make_box(CUBE_ID, cube_pos, (CUBE_SIZE,) * 3, BASE_FRAME)
        scene.apply_collision_object(cube, color_for(CUBE_ID, (0.9, 0.1, 0.1, 1.0)))
        _allow_cube_touch(scene)
        scene.current_state.update()
    logger.info("Spawned cube and allowed gripper-head contact.")


# ---------------------------------------------------------------------------
# INTERFACE 3: attach / detach
# ---------------------------------------------------------------------------

def attach_cube(cobot, logger):
    """Attach the existing world cube to the tool link (it stays visible)."""
    psm = cobot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        aco = AttachedCollisionObject()
        aco.link_name = TOOL_LINK
        aco.object.id = CUBE_ID
        aco.object.operation = CollisionObject.ADD
        aco.touch_links = GRIPPER_TOUCH_LINKS
        scene.process_attached_collision_object(aco)
        scene.current_state.update()
    logger.info("Cube attached to gripper.")


def detach_cube(cobot, logger):
    psm = cobot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        aco = AttachedCollisionObject()
        aco.link_name = TOOL_LINK
        aco.object.id = CUBE_ID
        aco.object.operation = CollisionObject.REMOVE
        scene.process_attached_collision_object(aco)
        _allow_cube_touch(scene)   # re-allow now that it is a world object again
        scene.current_state.update()
    logger.info("Cube detached (stays at the place location).")


# ---------------------------------------------------------------------------
# INTERFACE 2: move the arm
# ---------------------------------------------------------------------------

def move_to_joints(cobot, arm, robot_model, logger, joints, label):
    goal_state = RobotState(robot_model)
    goal_state.joint_positions = {j: float(v) for j, v in joints.items()}
    goal_state.update()
    arm.set_start_state_to_current_state()
    arm.set_goal_state(robot_state=goal_state)
    return plan_and_execute(cobot, arm, label, logger)


def lock_height_joint(arm, logger, value):
    c = Constraints()
    c.name = "lock_joint_0"
    jc = JointConstraint()
    jc.joint_name = HEIGHT_JOINT
    jc.position = float(value)
    jc.tolerance_above = 0.001
    jc.tolerance_below = 0.001
    jc.weight = 1.0
    c.joint_constraints.append(jc)
    arm.set_path_constraints(c)
    logger.info(f"Locked {HEIGHT_JOINT} at {value} for all following motions.")


def move_tool_to(cobot, arm, logger, position, quat, label):
    ps = PoseStamped()
    ps.header.frame_id = BASE_FRAME
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in position)
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = quat
    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=ps, pose_link=TOOL_LINK)
    return plan_and_execute(cobot, arm, label, logger)


def set_gripper(cobot, gripper, logger, state):
    gripper.set_start_state_to_current_state()
    gripper.set_goal_state(configuration_name=state)
    return plan_and_execute(cobot, gripper, f"gripper -> {state}", logger)


def current_tool_pose(cobot):
    psm = cobot.get_planning_scene_monitor()
    with psm.read_only() as scene:
        T = np.array(scene.get_frame_transform(TOOL_LINK))
    pos = T[:3, 3].copy()
    quat = rotmat_to_quat(T[:3, :3])
    return pos, quat


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    logger = get_logger("pick_place_demo")

    logger.info("Waiting for joint states ...")
    wait_for_joint_states(logger)

    logger.info("Starting MoveIt ...")
    moveit_config = generate_moveit_config()
    moveit_config["planning_scene_monitor_options"] = {
        "name": "planning_scene_monitor",
        "robot_description": "robot_description",
        "joint_state_topic": "/joint_states",
        "attached_collision_object_topic": "/moveit_cpp/planning_scene_monitor",
        "publish_planning_scene_topic": "/moveit_cpp/publish_planning_scene",
        "monitored_planning_scene_topic": "/moveit_cpp/monitored_planning_scene",
        "wait_for_initial_state_timeout": 10.0,
    }
    cobot = MoveItPy(node_name="pick_place_demo", config_dict=moveit_config)
    robot_model = cobot.get_robot_model()
    arm = cobot.get_planning_component(ARM_GROUP)
    gripper = cobot.get_planning_component(GRIPPER_GROUP)
    logger.info("MoveIt ready.")
    time.sleep(3.0)

    # ----- 1. drive ONCE to init, then lock joint_0 -----------------------
    if not move_to_joints(cobot, arm, robot_model, logger, INIT_JOINTS, "move to init"):
        logger.error("Could not reach init pose, aborting.")
        rclpy.shutdown()
        return
    lock_height_joint(arm, logger, INIT_JOINTS[HEIGHT_JOINT])

    # ----- 2. anchor to the tool pose AT init ------------------------------
    p0, q0 = current_tool_pose(cobot)
    logger.info(f"Tool position at init: {np.round(p0, 3).tolist()}")

    grasp_pos = p0 + np.array([0.0, 0.0, -PICK_DROP])
    pre_grasp_pos = grasp_pos + np.array([0.0, 0.0, LIFT])
    place_pos = grasp_pos + np.array(PLACE_SHIFT)
    pre_place_pos = place_pos + np.array([0.0, 0.0, LIFT])

    # ----- 3. spawn cube (stays visible the whole time) --------------------
    spawn_cube(cobot, logger, grasp_pos)
    print_scene_objects(cobot, logger)
    time.sleep(1.0)

    # ----- 4. pick-and-place ----------------------------------------------
    steps = [
        ("open gripper",       lambda: set_gripper(cobot, gripper, logger, GRIPPER_OPEN_STATE)),
        ("move above cube",    lambda: move_tool_to(cobot, arm, logger, pre_grasp_pos, q0, "pre-grasp")),
        ("move down to cube",  lambda: move_tool_to(cobot, arm, logger, grasp_pos, q0, "grasp")),
        ("close gripper",      lambda: set_gripper(cobot, gripper, logger, GRIPPER_CLOSE_STATE)),
        ("attach cube",        lambda: (attach_cube(cobot, logger) or True)),
        ("lift cube",          lambda: move_tool_to(cobot, arm, logger, pre_grasp_pos, q0, "lift")),
        ("move over drop",     lambda: move_tool_to(cobot, arm, logger, pre_place_pos, q0, "pre-place")),
        ("move down to drop",  lambda: move_tool_to(cobot, arm, logger, place_pos, q0, "place")),
        ("open gripper",       lambda: set_gripper(cobot, gripper, logger, GRIPPER_OPEN_STATE)),
        ("detach cube",        lambda: (detach_cube(cobot, logger) or True)),
        ("retreat",            lambda: move_tool_to(cobot, arm, logger, pre_place_pos, q0, "retreat")),
    ]

    for name, action in steps:
        logger.info("---------------------------------------------")
        logger.info(f"STEP: {name}")
        if not action():
            logger.error(f"Aborted at step: {name}")
            break
    else:
        logger.info("=== pick-and-place finished ===")

    rclpy.shutdown()


if __name__ == "__main__":
    main()