#!/usr/bin/env python3
"""
Scripted pick-and-place on the worktable (NO MoveIt Task Constructor).

You set WHERE the cube is and WHERE it goes, in world coordinates. MoveIt
does the rest (inverse kinematics + planning). No joint angles to touch.

  * joint_0 (height axis) is driven ONCE to 0.3 and then kept ~fixed, like
    teach/replay.
  * The gripper approaches with a fixed orientation (GRASP_QUAT) -- that is
    just "how the gripper is turned", MoveIt figures out the joints for it.
  * No physics: grasping = allow gripper/cube overlap + attach. Cube visible.

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


# ===========================================================================
# THE ONLY THINGS YOU CHANGE: where the cube is, and where it goes.
# World coordinates in metres. Change x and y; keep z about the table height.
# ===========================================================================

CUBE_POS = (0.4, -0.285, 0.773)   # where the cube lies on the table
DROP_POS = (0.534, -0.130, 0.773)   # where to put it down (fixed spot)

# How the gripper is turned when it grasps (quaternion x, y, z, w). This is an
# orientation, not a joint angle -- MoveIt solves the joints for it. Leave it.
GRASP_QUAT = (0.673105, 0.738588, 0.031103, -0.021228)

# ===========================================================================


# --- robot constants (match config.py / the SRDF) -------------------------

ARM_GROUP = "arm_group"
GRIPPER_GROUP = "gripper_group"
TOOL_LINK = "gripper_tcp"
BASE_FRAME = "world"
HEIGHT_JOINT = "joint_0"

GRIPPER_OPEN_STATE = "open"
GRIPPER_CLOSE_STATE = "close"

GRIPPER_TOUCH_LINKS = [
    "finger1", "finger2", "gripper_tcp", "segment7",
    "vacuum_gripper_up", "vacuum_gripper_down",
]
# The cube rests on the table (the workbench mesh on these links).
TABLE_TOUCH_LINKS = ["lin_axis", "lin_axis_carriage"]
CUBE_ALLOWED_LINKS = GRIPPER_TOUCH_LINKS + TABLE_TOUCH_LINKS

# The home pose (= config.py INIT_JOINTS). Reached once; joint_0 then ~fixed.
INIT_JOINTS = {
    "joint_0": 0.3, "joint_1": 0.0, "joint_2": 0.0, "joint_3": 0.0,
    "joint_4": 0.0, "joint_5": 0.0, "joint_6": 0.0,
}

CUBE_ID = "cube"
CUBE_SIZE = 0.04

# Tool point sits this far above the cube centre (so the cube rests on the
# table while the gripper grasps from above). Also the grasp offset.
CUBE_DROP = 0.05
# Vertical clearance for lifting/lowering around pick and drop.
RETREAT_LIFT = 0.10
# How much joint_0 may drift (small = ~fixed, but too small makes the
# constrained goal sampler fail). 0.01 keeps it ~fixed yet plannable.
JOINT0_TOL = 0.01


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
    acm = scene.allowed_collision_matrix
    for link in CUBE_ALLOWED_LINKS:
        acm.set_entry(CUBE_ID, link, True)


def spawn_cube(cobot, logger, cube_pos):
    psm = cobot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        cube = make_box(CUBE_ID, cube_pos, (CUBE_SIZE,) * 3, BASE_FRAME)
        scene.apply_collision_object(cube, color_for(CUBE_ID, (0.9, 0.1, 0.1, 1.0)))
        _allow_cube_touch(scene)
        scene.current_state.update()
    logger.info(f"Spawned cube at {np.round(cube_pos, 3).tolist()}.")


def remove_cube_from_world(cobot, logger):
    psm = cobot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        obj = CollisionObject()
        obj.id = CUBE_ID
        obj.header.frame_id = BASE_FRAME
        obj.operation = CollisionObject.REMOVE
        scene.apply_collision_object(obj)
        scene.current_state.update()
    logger.info("Reset: cube removed from scene.")


# ---------------------------------------------------------------------------
# INTERFACE 3: attach / detach
# ---------------------------------------------------------------------------

def attach_cube(cobot, logger):
    psm = cobot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        aco = AttachedCollisionObject()
        aco.link_name = TOOL_LINK
        aco.object.id = CUBE_ID
        aco.object.operation = CollisionObject.ADD
        aco.touch_links = GRIPPER_TOUCH_LINKS
        scene.process_attached_collision_object(aco)
        _allow_cube_touch(scene)   # re-apply (attach resets the ACM)
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
        _allow_cube_touch(scene)
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
    jc.tolerance_above = JOINT0_TOL
    jc.tolerance_below = JOINT0_TOL
    jc.weight = 1.0
    c.joint_constraints.append(jc)
    arm.set_path_constraints(c)
    logger.info(f"Locked {HEIGHT_JOINT} at {value} (+/- {JOINT0_TOL}).")


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

    # ----- 1. drive ONCE to init, then keep joint_0 ~fixed ----------------
    if not move_to_joints(cobot, arm, robot_model, logger, INIT_JOINTS, "move to init"):
        logger.error("Could not reach init pose, aborting.")
        rclpy.shutdown()
        return
    lock_height_joint(arm, logger, INIT_JOINTS[HEIGHT_JOINT])

    # ----- 2. poses from the positions you set ----------------------------
    cube_pos = np.array(CUBE_POS, dtype=float)
    drop_pos = np.array(DROP_POS, dtype=float)
    up = np.array([0.0, 0.0, RETREAT_LIFT])
    offset = np.array([0.0, 0.0, CUBE_DROP])

    grasp_pos = cube_pos + offset     # tool point above the cube
    grasp_up = grasp_pos + up
    place_pos = drop_pos + offset     # tool point above the drop spot
    place_up = place_pos + up

    # ----- 3. spawn cube on the table -------------------------------------
    spawn_cube(cobot, logger, cube_pos)
    time.sleep(1.0)

    # ----- 4. pick-and-place: approach, lift, carry, lower, release -------
    steps = [
        ("open gripper",        lambda: set_gripper(cobot, gripper, logger, GRIPPER_OPEN_STATE)),
        ("drive to cube",       lambda: move_tool_to(cobot, arm, logger, grasp_pos, GRASP_QUAT, "approach")),
        ("close gripper",       lambda: set_gripper(cobot, gripper, logger, GRIPPER_CLOSE_STATE)),
        ("attach cube",         lambda: (attach_cube(cobot, logger) or True)),
        ("lift cube",           lambda: move_tool_to(cobot, arm, logger, grasp_up, GRASP_QUAT, "lift")),
        ("carry over drop",     lambda: move_tool_to(cobot, arm, logger, place_up, GRASP_QUAT, "carry")),
        ("lower to drop",       lambda: move_tool_to(cobot, arm, logger, place_pos, GRASP_QUAT, "lower")),
        ("open gripper",        lambda: set_gripper(cobot, gripper, logger, GRIPPER_OPEN_STATE)),
        ("detach cube",         lambda: (detach_cube(cobot, logger) or True)),
        ("lift off",            lambda: move_tool_to(cobot, arm, logger, place_up, GRASP_QUAT, "retreat")),
        ("return to init",      lambda: move_to_joints(cobot, arm, robot_model, logger, INIT_JOINTS, "return to init")),
        ("reset: remove cube",  lambda: (remove_cube_from_world(cobot, logger) or True)),
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