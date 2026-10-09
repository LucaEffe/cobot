"""
Arm execution adapter (MoveIt) + current-state reader.

Thin wrapper around MoveItPy so the nodes and the gripper backends share
one plan-and-execute implementation. This is the only place that talks to
MoveIt for arm motion.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from moveit.core.robot_state import RobotState

from .config import ARM_JOINTS, ARM_GROUP, INIT_JOINTS


def plan_and_execute(cobot, component, label: str, logger, plan_params=None) -> bool:
    """
    Plan the currently configured goal of `component` and execute it.

    `plan_params` optionally selects the planner / scaling (see ArmExecutor).
    Returns True only when both planning and execution succeed.
    """

    logger.info(f"Planning: {label}")

    try:
        if plan_params is not None:
            plan_result = component.plan(single_plan_parameters=plan_params)
        else:
            plan_result = component.plan()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Planning exception for '{label}': {e}")
        return False

    if not plan_result:
        logger.error(f"Planning FAILED: {label}")
        return False

    logger.info(f"Executing: {label}")

    try:
        status = cobot.execute(plan_result.trajectory, controllers=[])
    except Exception as e:  # noqa: BLE001
        logger.error(f"Execution exception for '{label}': {e}")
        return False

    if not status:
        logger.error(f"Execution FAILED: {label}")
        return False

    logger.info(f"Execution finished: {label}")
    return True


class ArmExecutor:
    """Moves the arm to joint configurations via MoveIt.

    pipeline / planner_id choose the motion planner:
      OMPL:  pipeline="ompl",
             planner_id="RRTConnectkConfigDefault" | "RRTkConfigDefault"
                        | "RRTstarkConfigDefault" | "PRMkConfigDefault"
      Pilz:  pipeline="pilz_industrial_motion_planner",
             planner_id="LIN" | "PTP" | "CIRC"   (pipeline must be loaded)

    If the params cannot be built (e.g. pipeline not loaded), it falls back
    to the default planner with a warning instead of failing.
    """

    def __init__(self, cobot, logger, speed: float = 1.0,
                 planner_config: str = "ompl_rrtc",
                 pipeline: str = "chomp",
                 planner_id: str = ""):
        self.cobot = cobot
        self.logger = logger
        self.robot_model = cobot.get_robot_model()
        self.arm = cobot.get_planning_component(ARM_GROUP)
        self.speed = max(0.0, min(1.0, speed))
        self.planner_config = planner_config
        self.pipeline = pipeline
        self.planner_id = planner_id
        self._plan_params = self._build_plan_params()

    def _build_plan_params(self):
        try:
            from moveit.planning import PlanRequestParameters
            params = PlanRequestParameters(self.cobot, self.planner_config)
            params.planning_pipeline = self.pipeline
            params.planner_id = self.planner_id
            if self.speed < 1.0:
                params.max_velocity_scaling_factor = self.speed
                params.max_acceleration_scaling_factor = self.speed
            self.logger.info(
                f"Planner: pipeline='{self.pipeline}' id='{self.planner_id}' "
                f"speed={self.speed}"
            )
            return params
        except Exception as e:  # noqa: BLE001
            self.logger.warn(
                f"Could not build plan params ({e}); default planner."
            )
            return None

    def move_to(self, joints: Dict[str, float], label: str = "arm") -> bool:
        goal_state = RobotState(self.robot_model)
        goal_state.joint_positions = {
            j: float(joints[j]) for j in ARM_JOINTS
        }
        goal_state.update()

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(robot_state=goal_state)

        return plan_and_execute(
            self.cobot, self.arm, label, self.logger, self._plan_params
        )

    def move_to_init(self) -> bool:
        return self.move_to(INIT_JOINTS, label="move to init")


def read_current_arm_joints(
    logger,
    timeout: float = 5.0,
) -> Optional[Dict[str, float]]:
    """
    Read the current arm joint positions from /joint_states.

    Uses a short-lived node; call this BEFORE creating MoveItPy to avoid
    executor interference. Returns a dict for all ARM_JOINTS, or None on
    timeout.
    """

    node = Node("replay_init_check")
    got: Dict[str, float] = {}

    def _cb(msg: JointState) -> None:
        got.update(dict(zip(msg.name, msg.position)))

    node.create_subscription(JointState, "/joint_states", _cb, 10)

    start = time.time()
    result: Optional[Dict[str, float]] = None

    try:
        while rclpy.ok() and (time.time() - start) < timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
            if all(j in got for j in ARM_JOINTS):
                result = {j: got[j] for j in ARM_JOINTS}
                break
    finally:
        node.destroy_node()

    return result