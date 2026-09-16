#!/usr/bin/env python3

import argparse
import json
import math
import os
import time

import rclpy
from rclpy.logging import get_logger

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

from py_utils.planner_utils import (
    wait_for_joint_states,
    generate_moveit_config,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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

# Must match pedal_node.py
GRIPPER_GROUP = "gripper_group"
OPEN_STATE = "open"
CLOSE_STATE = "close"


# ---------------------------------------------------------------------------
# Load recording
# ---------------------------------------------------------------------------

def load_waypoints(path):
    """
    Load all JSONL waypoints from a teach recording.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Waypoint file does not exist: {path}"
        )

    waypoints = []

    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                waypoint = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in line {line_no}: {e}"
                ) from e

            validate_waypoint(waypoint, line_no)
            waypoints.append(waypoint)

    if not waypoints:
        raise ValueError(
            "Waypoint file contains no waypoints."
        )

    return waypoints


def validate_waypoint(waypoint, line_no):
    """
    Check that one recorded waypoint contains everything required
    for replay.
    """

    required_fields = [
        "event",
        "joints",
        "gripper",
    ]

    for field in required_fields:
        if field not in waypoint:
            raise ValueError(
                f"Line {line_no}: missing field '{field}'"
            )

    joints = waypoint["joints"]

    if not isinstance(joints, dict):
        raise ValueError(
            f"Line {line_no}: 'joints' must be a dictionary"
        )

    for joint in ARM_JOINTS:
        if joint not in joints:
            raise ValueError(
                f"Line {line_no}: missing joint '{joint}'"
            )

        value = joints[joint]

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Line {line_no}: joint '{joint}' "
                f"is not numeric: {value}"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Line {line_no}: joint '{joint}' "
                f"is not finite: {value}"
            )

    gripper = waypoint["gripper"]

    if gripper not in ("open", "closed"):
        raise ValueError(
            f"Line {line_no}: invalid gripper state "
            f"'{gripper}'"
        )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

class Replayer:
    def __init__(self, cobot, logger):
        self.cobot = cobot
        self.logger = logger

        # MoveIt robot model is needed to construct RobotState goals.
        self.robot_model = cobot.get_robot_model()

        # Planning components
        self.arm = cobot.get_planning_component(
            ARM_GROUP
        )

        self.gripper = cobot.get_planning_component(
            GRIPPER_GROUP
        )

        # We deliberately do NOT assume that the physical gripper
        # starts open or closed.
        self.current_gripper = None

    # ------------------------------------------------------------------
    # Generic planning + execution
    # ------------------------------------------------------------------

    def plan_and_execute_checked(
        self,
        component,
        label,
    ):
        """
        Plan the currently configured goal and execute it.

        Returns True only when both planning and execution succeed.
        """

        self.logger.info(
            f"Planning: {label}"
        )

        try:
            plan_result = component.plan()
        except Exception as e:
            self.logger.error(
                f"Planning exception for '{label}': {e}"
            )
            return False

        if not plan_result:
            self.logger.error(
                f"Planning FAILED: {label}"
            )
            return False

        self.logger.info(
            f"Executing: {label}"
        )

        try:
            # MoveItPy execute() blocks until trajectory execution
            # has completed.
            status = self.cobot.execute(
                plan_result.trajectory,
                controllers=[],
            )
        except Exception as e:
            self.logger.error(
                f"Execution exception for '{label}': {e}"
            )
            return False

        if not status:
            status_text = getattr(
                status,
                "status",
                "unknown",
            )

            self.logger.error(
                f"Execution FAILED: {label} "
                f"(status={status_text})"
            )

            return False

        status_text = getattr(
            status,
            "status",
            "success",
        )

        self.logger.info(
            f"Execution finished: {label} "
            f"(status={status_text})"
        )

        return True

    # ------------------------------------------------------------------
    # Arm
    # ------------------------------------------------------------------

    def move_arm(self, joints):
        """
        Move the arm to one recorded joint configuration.
        """

        # Build a MoveIt RobotState from the recorded joint values.
        goal_state = RobotState(
            self.robot_model
        )

        goal_state.joint_positions = {
            joint: float(joints[joint])
            for joint in ARM_JOINTS
        }

        # Update forward kinematics after changing joint values.
        goal_state.update()

        # IMPORTANT:
        # Start every new segment from the current robot state,
        # not from an assumed previous state.
        self.arm.set_start_state_to_current_state()

        self.arm.set_goal_state(
            robot_state=goal_state
        )

        return self.plan_and_execute_checked(
            self.arm,
            "arm waypoint",
        )

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def set_gripper(self, desired_state):
        """
        Explicitly set the gripper to open or closed.
        """

        if desired_state == "closed":
            moveit_target = CLOSE_STATE

        elif desired_state == "open":
            moveit_target = OPEN_STATE

        else:
            self.logger.error(
                f"Unknown gripper state: "
                f"{desired_state}"
            )

            return False

        self.logger.info(
            f"Gripper: "
            f"{self.current_gripper} "
            f"-> {desired_state}"
        )

        self.gripper.set_start_state_to_current_state()

        self.gripper.set_goal_state(
            configuration_name=moveit_target
        )

        success = self.plan_and_execute_checked(
            self.gripper,
            f"gripper -> {desired_state}",
        )

        # Only change our internal state after MoveIt reports
        # successful execution.
        if success:
            self.current_gripper = desired_state

        return success

    # ------------------------------------------------------------------
    # Complete recording replay
    # ------------------------------------------------------------------

    def replay(
        self,
        waypoints,
        skip_init=False,
    ):
        """
        Replay all recorded waypoints.

        Rule:
            1. Reach arm waypoint.
            2. Only after arm execution finished:
               change gripper if necessary.
            3. Continue with next waypoint.
        """

        playable_waypoints = []

        for waypoint in waypoints:
            # Useful while testing:
            # optionally ignore the synthetic final init target.
            if (
                skip_init
                and waypoint["event"] == "init"
            ):
                continue

            playable_waypoints.append(
                waypoint
            )

        if not playable_waypoints:
            self.logger.error(
                "No playable waypoints."
            )
            return False

        total = len(playable_waypoints)

        self.logger.info(
            "========================================"
        )

        self.logger.info(
            f"REPLAY START: {total} waypoints"
        )

        self.logger.info(
            "========================================"
        )

        for index, waypoint in enumerate(
            playable_waypoints,
            start=1,
        ):
            event = waypoint["event"]
            desired_gripper = waypoint["gripper"]

            self.logger.info(
                "----------------------------------------"
            )

            self.logger.info(
                f"[{index}/{total}] "
                f"event={event}, "
                f"gripper={desired_gripper}"
            )

            # ==========================================================
            # STEP 1:
            # Reach the recorded arm configuration FIRST.
            # ==========================================================

            self.logger.info(
                f"[{index}/{total}] "
                f"Moving arm ..."
            )

            arm_success = self.move_arm(
                waypoint["joints"]
            )

            if not arm_success:
                self.logger.error(
                    f"[{index}/{total}] "
                    f"Replay aborted: "
                    f"arm motion failed."
                )

                return False

            self.logger.info(
                f"[{index}/{total}] "
                f"Arm pose reached."
            )

            # ==========================================================
            # STEP 2:
            # Only AFTER reaching the arm pose,
            # handle the gripper.
            # ==========================================================

            if self.current_gripper is None:
                # At the beginning of replay we don't know
                # the actual physical state of the gripper.
                #
                # Therefore explicitly establish the state
                # stored in the first waypoint.

                self.logger.info(
                    f"[{index}/{total}] "
                    f"Initial gripper state unknown. "
                    f"Setting -> {desired_gripper}"
                )

                if not self.set_gripper(
                    desired_gripper
                ):
                    self.logger.error(
                        f"[{index}/{total}] "
                        f"Replay aborted: "
                        f"initial gripper setup failed."
                    )

                    return False

            elif (
                desired_gripper
                != self.current_gripper
            ):
                # Recorded state changed:
                # NOW actuate the gripper.

                self.logger.info(
                    f"[{index}/{total}] "
                    f"Gripper change required: "
                    f"{self.current_gripper} "
                    f"-> {desired_gripper}"
                )

                if not self.set_gripper(
                    desired_gripper
                ):
                    self.logger.error(
                        f"[{index}/{total}] "
                        f"Replay aborted: "
                        f"gripper motion failed."
                    )

                    return False

            else:
                # No change -> no unnecessary gripper command.
                self.logger.info(
                    f"[{index}/{total}] "
                    f"Gripper unchanged "
                    f"({desired_gripper})."
                )

        self.logger.info(
            "========================================"
        )

        self.logger.info(
            "REPLAY FINISHED SUCCESSFULLY"
        )

        self.logger.info(
            "========================================"
        )

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay Cobot teach waypoints "
            "using MoveIt."
        )
    )

    parser.add_argument(
        "waypoints",
        help=(
            "Path to teach/waypoints.jsonl"
        ),
    )

    parser.add_argument(
        "--skip-init",
        action="store_true",
        help=(
            "Do not execute the final "
            "synthetic init waypoint."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # ROS
    # --------------------------------------------------------------

    rclpy.init()

    logger = get_logger(
        "cobot_replay"
    )

    try:
        # ----------------------------------------------------------
        # Load recording
        # ----------------------------------------------------------

        try:
            waypoints = load_waypoints(
                args.waypoints
            )

        except Exception as e:
            logger.error(
                f"Could not load recording: {e}"
            )
            return

        logger.info(
            f"Loaded {len(waypoints)} waypoints:"
        )

        for index, waypoint in enumerate(
            waypoints,
            start=1,
        ):
            logger.info(
                f"  #{index}: "
                f"event={waypoint['event']}, "
                f"gripper={waypoint['gripper']}"
            )

        # ----------------------------------------------------------
        # Wait until robot / simulation publishes states
        # ----------------------------------------------------------

        logger.info(
            "Waiting for joint states ..."
        )

        wait_for_joint_states(
            logger
        )

        # ----------------------------------------------------------
        # MoveIt
        # ----------------------------------------------------------

        logger.info(
            "Starting MoveIt ..."
        )

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

        cobot = MoveItPy(
            node_name="replay_moveit",
            config_dict=moveit_config,
        )

        logger.info(
            "MoveIt ready."
        )

        logger.info(
            "Waiting for controller discovery ..."
        )

        time.sleep(3.0)

        logger.info(
            "Controller discovery grace period finished."
        )

        # ==========================================================
        # TODO REAL ROBOT:
        #
        # Before replay starts on the physical robot,
        # hand-guiding / compliant mode probably has to be
        # switched to active trajectory control here.
        #
        # This is deliberately NOT implemented yet.
        # ==========================================================

        replayer = Replayer(
            cobot=cobot,
            logger=logger,
        )

        success = replayer.replay(
            waypoints,
            skip_init=args.skip_init,
        )

        if not success:
            logger.error(
                "Replay failed."
            )

    except KeyboardInterrupt:
        logger.warn(
            "Replay interrupted by user."
        )

    except Exception as e:
        logger.error(
            f"Unexpected replay error: {e}"
        )

    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()