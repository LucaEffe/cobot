"""
Pure replay logic: the init safety check and the action plan.

No rclpy, no MoveIt. This is the "intelligence" of the replay and can be
unit-tested without a robot:

  * check_at_init  -> the safety gate ("only replay from init")
  * build_action_plan -> turns recorded waypoints into an ordered list of
    MoveArm / SetGripper actions, encoding the rule
    "reach the arm pose first, only then change the gripper (and only if it
    actually changed)".

The nodes just execute this plan against a robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .config import ARM_JOINTS, INIT_JOINTS
from .waypoints import Waypoint


# ---- init safety check ---------------------------------------------------

def check_at_init(
    current: Dict[str, float],
    tol: float,
) -> Tuple[bool, Optional[str]]:
    """
    Return (True, None) if every arm joint is within `tol` of INIT_JOINTS,
    otherwise (False, first_offending_joint).
    """

    for joint in ARM_JOINTS:
        if joint not in current:
            return False, joint
        if abs(current[joint] - INIT_JOINTS[joint]) > tol:
            return False, joint

    return True, None


# ---- action plan ---------------------------------------------------------

@dataclass
class MoveArm:
    joints: Dict[str, float]
    label: str


@dataclass
class SetGripper:
    state: str


Action = Union[MoveArm, SetGripper]


def build_action_plan(
    waypoints: List[Waypoint],
    skip_init: bool = False,
) -> List[Action]:
    """
    Build the ordered action list for a replay.

    Rule per waypoint:
        1. MoveArm to the recorded configuration.
        2. Afterwards, SetGripper -- but only for the very first waypoint
           (unknown physical state) or when the recorded state changed.
    """

    playable = [
        wp for wp in waypoints
        if not (skip_init and wp.event == "init")
    ]

    actions: List[Action] = []
    current_gripper: Optional[str] = None

    for index, wp in enumerate(playable, start=1):
        actions.append(
            MoveArm(
                joints=wp.joints,
                label=f"[{index}/{len(playable)}] {wp.event}",
            )
        )

        if current_gripper is None or wp.gripper != current_gripper:
            actions.append(SetGripper(state=wp.gripper))
            current_gripper = wp.gripper

    return actions
