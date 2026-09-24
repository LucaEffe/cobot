"""
Waypoint data model + JSONL I/O + validation.

Pure Python: no rclpy, no MoveIt. This is the on-disk format shared by the
recorder (writes) and the replay (reads), and it is unit-testable without a
robot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, TextIO

from .config import ARM_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED


VALID_GRIPPER_STATES = (GRIPPER_OPEN, GRIPPER_CLOSED)


class WaypointError(ValueError):
    """Raised when a waypoint is malformed."""


@dataclass
class Waypoint:
    """One recorded step: an arm configuration plus the gripper state."""

    event: str
    joints: Dict[str, float]
    gripper: str
    t: float = 0.0

    # ---- construction / validation ----

    @classmethod
    def from_dict(cls, data: dict, where: str = "") -> "Waypoint":
        for f in ("event", "joints", "gripper"):
            if f not in data:
                raise WaypointError(f"{where}: missing field '{f}'")

        joints = data["joints"]
        validate_joints(joints, where)

        gripper = data["gripper"]
        if gripper not in VALID_GRIPPER_STATES:
            raise WaypointError(
                f"{where}: invalid gripper state '{gripper}'"
            )

        return cls(
            event=data["event"],
            joints={j: float(joints[j]) for j in ARM_JOINTS},
            gripper=gripper,
            t=float(data.get("t", 0.0)),
        )

    def to_dict(self, decimals: int = 6) -> dict:
        return {
            "t": self.t,
            "event": self.event,
            "joints": {
                j: round(float(self.joints[j]), decimals)
                for j in ARM_JOINTS
            },
            "gripper": self.gripper,
        }


def validate_joints(joints, where: str = "") -> None:
    if not isinstance(joints, dict):
        raise WaypointError(f"{where}: 'joints' must be a dictionary")

    for joint in ARM_JOINTS:
        if joint not in joints:
            raise WaypointError(f"{where}: missing joint '{joint}'")

        value = joints[joint]
        if not isinstance(value, (int, float)):
            raise WaypointError(
                f"{where}: joint '{joint}' is not numeric: {value}"
            )
        if not math.isfinite(value):
            raise WaypointError(
                f"{where}: joint '{joint}' is not finite: {value}"
            )


# ---- I/O -----------------------------------------------------------------

def load_waypoints(path: str) -> List[Waypoint]:
    """Load and validate all waypoints from a JSONL file."""

    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Waypoint file does not exist: {path}")

    waypoints: List[Waypoint] = []

    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise WaypointError(
                    f"Invalid JSON in line {line_no}: {e}"
                ) from e

            waypoints.append(
                Waypoint.from_dict(raw, where=f"Line {line_no}")
            )

    if not waypoints:
        raise WaypointError("Waypoint file contains no waypoints.")

    return waypoints


def dump_waypoint(fileobj: TextIO, waypoint: Waypoint) -> None:
    """Append one waypoint as a JSON line and flush."""

    fileobj.write(json.dumps(waypoint.to_dict()) + "\n")
    fileobj.flush()
