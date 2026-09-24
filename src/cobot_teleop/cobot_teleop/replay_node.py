"""
Replay entry point (thin ROS glue).

Wires together:
    waypoints.py    -> load + validate the recording
    replay_core.py  -> init safety check + action plan (pure logic)
    arm.py          -> execute arm motions (MoveIt)
    grippers.py     -> execute gripper changes (sim MoveIt / real OPC-UA)

Safety: replay only runs when the robot is at the init pose. With
--move-to-init the robot is homed to init first instead of aborting.

    ros2 run cobot_teleop replay_node <waypoints.jsonl>
        [--controller-type sim|real] [--move-to-init] [--init-tol T]
        [--skip-init]
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.logging import get_logger

from moveit.planning import MoveItPy

from py_utils.planner_utils import (
    wait_for_joint_states,
    generate_moveit_config,
)

from .arm import ArmExecutor, read_current_arm_joints
from .config import DEFAULT_INIT_TOL, INIT_JOINTS
from .grippers import make_gripper
from .replay_core import (
    MoveArm,
    SetGripper,
    build_action_plan,
    check_at_init,
)
from .waypoints import load_waypoints


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replay Cobot teach waypoints using MoveIt."
    )
    parser.add_argument("waypoints", help="Path to teach/waypoints.jsonl")
    parser.add_argument(
        "--controller-type",
        choices=["sim", "real"],
        default="sim",
        help="sim -> sim controllers + MoveIt gripper. "
        "real -> real controllers + standalone OPC-UA gripper.",
    )
    parser.add_argument(
        "--move-to-init",
        action="store_true",
        help="If not at init, home to init first instead of aborting.",
    )
    parser.add_argument(
        "--init-tol",
        type=float,
        default=DEFAULT_INIT_TOL,
        help=f"Tolerance for the init check. Default {DEFAULT_INIT_TOL}.",
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Do not execute the synthetic init waypoint from the file.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Arm velocity/acceleration scaling (0..1). "
        "Use e.g. 0.1 to slow the real robot down for first tests.",
    )
    parser.add_argument(
        "--planner-config",
        default="ompl_rrtc",
        help="Name of the plan_request_params block used for speed "
        "scaling. Default 'ompl_rrtc'.",
    )
    return parser.parse_args()


def _run_plan(plan, arm: ArmExecutor, gripper, logger) -> bool:
    total = len(plan)
    logger.info("=" * 40)
    logger.info(f"REPLAY START: {total} actions")
    logger.info("=" * 40)

    for i, action in enumerate(plan, start=1):
        if isinstance(action, MoveArm):
            if not arm.move_to(action.joints, action.label):
                logger.error(f"[{i}/{total}] Replay aborted: arm motion failed.")
                return False
        elif isinstance(action, SetGripper):
            logger.info(f"[{i}/{total}] Gripper -> {action.state}")
            if not gripper.set_state(action.state):
                logger.error(f"[{i}/{total}] Replay aborted: gripper failed.")
                return False

    logger.info("=" * 40)
    logger.info("REPLAY FINISHED SUCCESSFULLY")
    logger.info("=" * 40)
    return True


def main():
    args = _parse_args()
    use_real = (args.controller_type == "real")

    rclpy.init()
    logger = get_logger("cobot_replay")

    try:
        # --- load recording ---
        try:
            waypoints = load_waypoints(args.waypoints)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not load recording: {e}")
            return

        logger.info(
            f"Controller type: {args.controller_type} (use_real={use_real})"
        )
        logger.info(f"Loaded {len(waypoints)} waypoints.")

        # --- wait for state ---
        logger.info("Waiting for joint states ...")
        wait_for_joint_states(logger)

        # --- init safety gate (before MoveItPy, own short-lived node) ---
        logger.info("Checking whether the robot is at the init pose ...")
        current = read_current_arm_joints(logger)
        if current is None:
            logger.error("Could not read current joint states -> abort.")
            return

        at_init, bad = check_at_init(current, args.init_tol)
        need_home = False
        if not at_init:
            if args.move_to_init:
                logger.warn(
                    f"Robot NOT at init (joint '{bad}': {current[bad]:.3f} "
                    f"vs {INIT_JOINTS[bad]:.3f}). Homing first."
                )
                need_home = True
            else:
                logger.error(
                    f"Robot is NOT at the init pose (joint '{bad}': "
                    f"{current[bad]:.3f} vs {INIT_JOINTS[bad]:.3f}, "
                    f"tol={args.init_tol}). Replay ABORTED. Move to init "
                    f"first, or start with --move-to-init."
                )
                return
        else:
            logger.info("Robot is at the init pose. Proceeding.")

        # --- MoveIt ---
        logger.info("Starting MoveIt ...")
        moveit_config = generate_moveit_config(use_real=use_real)
        moveit_config["planning_scene_monitor_options"] = {
            "name": "planning_scene_monitor",
            "robot_description": "robot_description",
            "joint_state_topic": "/joint_states",
            "attached_collision_object_topic": "/moveit_cpp/planning_scene_monitor",
            "publish_planning_scene_topic": "/moveit_cpp/publish_planning_scene",
            "monitored_planning_scene_topic": "/moveit_cpp/monitored_planning_scene",
            "wait_for_initial_state_timeout": 10.0,
        }
        cobot = MoveItPy(node_name="replay_moveit", config_dict=moveit_config)
        logger.info("MoveIt ready. Waiting for controller discovery ...")
        time.sleep(3.0)

        arm = ArmExecutor(
            cobot, logger,
            speed=args.speed,
            planner_config=args.planner_config,
        )
        gripper = make_gripper(
            use_real=use_real,
            cobot=cobot,
            logger=logger,
            restore_hand_guiding=False,   # replay: keep arm active
        )

        if need_home:
            if not arm.move_to_init():
                logger.error("Moving to init failed -> abort.")
                return
            logger.info("Init pose reached.")

        # --- build pure plan, then execute it ---
        plan = build_action_plan(waypoints, skip_init=args.skip_init)
        if not plan:
            logger.error("No playable actions.")
            return

        if not _run_plan(plan, arm, gripper, logger):
            logger.error("Replay failed.")

    except KeyboardInterrupt:
        logger.warn("Replay interrupted by user.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Unexpected replay error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()