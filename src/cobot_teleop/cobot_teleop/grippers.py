"""
Gripper backends.

This is the one place with a real abstraction, because the gripper is the
only thing that genuinely differs between simulation and the real robot:

  * MoveItGripper  -- simulation: drive the gripper group via MoveIt
                      (SRDF configuration_name).
  * OpcUaGripper   -- real robot: standalone OPC-UA commands
                      (cobot_hardware gripper_open / gripper_close), because
                      driving the gripper through MoveIt / ros2_control
                      freezes /joint_states.

Both implement the same tiny interface: set_state("open"|"closed") -> bool.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from typing import Callable, List

from .arm import plan_and_execute
from .config import (
    GRIPPER_CLOSED,
    GRIPPER_GROUP,
    HW_PACKAGE,
    SIM_GRIPPER_CLOSE_STATE,
    SIM_GRIPPER_OPEN_STATE,
)


class GripperBackend(ABC):
    """Set the gripper to an absolute state. Returns success."""

    @abstractmethod
    def set_state(self, state: str) -> bool:
        ...


# --- simulation -----------------------------------------------------------

class MoveItGripper(GripperBackend):
    def __init__(self, cobot, logger):
        self.cobot = cobot
        self.logger = logger
        self.gripper = cobot.get_planning_component(GRIPPER_GROUP)

    def set_state(self, state: str) -> bool:
        target = (
            SIM_GRIPPER_CLOSE_STATE
            if state == GRIPPER_CLOSED
            else SIM_GRIPPER_OPEN_STATE
        )
        self.gripper.set_start_state_to_current_state()
        self.gripper.set_goal_state(configuration_name=target)
        return plan_and_execute(
            self.cobot, self.gripper, f"gripper -> {state}", self.logger
        )


# --- real robot -----------------------------------------------------------

def default_hw_runner(logger=None) -> Callable[[List[str]], bool]:
    """
    Build a runner that invokes `ros2 run cobot_hardware <args...>` in a
    separate process (own OPC-UA connection). Injected into OpcUaGripper so
    it can be swapped out in tests.
    """

    def _run(args: List[str], timeout: float = 30.0) -> bool:
        cmd = ["ros2", "run", HW_PACKAGE] + list(args)
        try:
            subprocess.run(
                cmd,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            return True
        except Exception as e:  # noqa: BLE001
            if logger is not None:
                logger.error(f"HW cmd {args} failed: {e}")
            return False

    return _run


class OpcUaGripper(GripperBackend):
    """
    Real gripper via standalone OPC-UA commands.

    The gripper only actuates when the arm controller is active
    (balancer OFF), so we force balancer off first. Then:

      * teaching (restore_hand_guiding=True): turn balancer back ON so the
        arm stays hand-guidable.
      * replay (restore_hand_guiding=False): leave balancer OFF so the arm
        stays under active trajectory control.
    """

    def __init__(self, restore_hand_guiding: bool, logger=None, runner=None):
        self.restore_hand_guiding = restore_hand_guiding
        self.logger = logger
        self.run = runner if runner is not None else default_hw_runner(logger)

    def set_state(self, state: str) -> bool:
        self.run(["balancer", "off"])

        if state == GRIPPER_CLOSED:
            ok = self.run(["gripper_close"])
        else:
            ok = self.run(["gripper_open"])

        if self.restore_hand_guiding:
            self.run(["balancer", "on"])

        return ok


# --- factory --------------------------------------------------------------

def make_gripper(
    use_real: bool,
    cobot,
    logger,
    restore_hand_guiding: bool,
) -> GripperBackend:
    """Pick the backend for the current controller type."""

    if use_real:
        return OpcUaGripper(
            restore_hand_guiding=restore_hand_guiding, logger=logger
        )
    return MoveItGripper(cobot=cobot, logger=logger)
