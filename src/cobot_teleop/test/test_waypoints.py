"""
Unit tests for the waypoint data model + I/O.

Pure Python -- no robot, no ROS. Run with: pytest
"""

import json

import pytest

from cobot_teleop.config import ARM_JOINTS
from cobot_teleop.waypoints import (
    Waypoint,
    WaypointError,
    load_waypoints,
)


def _joints(value=0.0):
    return {j: value for j in ARM_JOINTS}


def _raw(**overrides):
    data = {"t": 1.0, "event": "waypoint", "joints": _joints(), "gripper": "open"}
    data.update(overrides)
    return data


def test_from_dict_valid():
    wp = Waypoint.from_dict(_raw())
    assert wp.event == "waypoint"
    assert wp.gripper == "open"
    assert set(wp.joints) == set(ARM_JOINTS)


def test_from_dict_missing_field():
    raw = _raw()
    del raw["gripper"]
    with pytest.raises(WaypointError):
        Waypoint.from_dict(raw)


def test_from_dict_missing_joint():
    raw = _raw()
    del raw["joints"]["joint_3"]
    with pytest.raises(WaypointError):
        Waypoint.from_dict(raw)


def test_from_dict_bad_gripper():
    with pytest.raises(WaypointError):
        Waypoint.from_dict(_raw(gripper="halfopen"))


def test_from_dict_non_finite_joint():
    raw = _raw()
    raw["joints"]["joint_2"] = float("inf")
    with pytest.raises(WaypointError):
        Waypoint.from_dict(raw)


def test_roundtrip():
    wp = Waypoint.from_dict(_raw())
    again = Waypoint.from_dict(wp.to_dict())
    assert again.gripper == wp.gripper
    assert again.joints == wp.joints


def test_load_waypoints(tmp_path):
    path = tmp_path / "waypoints.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(_raw(event="init")) + "\n")
        f.write("\n")  # blank line ignored
        f.write(json.dumps(_raw(gripper="closed")) + "\n")

    wps = load_waypoints(str(path))
    assert len(wps) == 2
    assert wps[0].event == "init"
    assert wps[1].gripper == "closed"


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_waypoints("/does/not/exist.jsonl")


def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(WaypointError):
        load_waypoints(str(path))
