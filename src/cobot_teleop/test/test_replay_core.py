"""
Unit tests for the pure replay logic: init gate + action plan.

Pure Python -- no robot, no ROS. Run with: pytest
"""

from cobot_teleop.config import ARM_JOINTS, INIT_JOINTS
from cobot_teleop.replay_core import (
    MoveArm,
    SetGripper,
    build_action_plan,
    check_at_init,
)
from cobot_teleop.waypoints import Waypoint


def _wp(gripper="open", event="waypoint", **joint_overrides):
    joints = dict(INIT_JOINTS)
    joints.update(joint_overrides)
    return Waypoint(event=event, joints=joints, gripper=gripper)


# ---- init gate ----

def test_at_init_exact():
    ok, bad = check_at_init(dict(INIT_JOINTS), tol=0.05)
    assert ok and bad is None


def test_at_init_within_tol():
    cur = dict(INIT_JOINTS)
    cur["joint_2"] += 0.02
    ok, bad = check_at_init(cur, tol=0.05)
    assert ok and bad is None


def test_not_at_init_revolute():
    cur = dict(INIT_JOINTS)
    cur["joint_4"] += 0.2
    ok, bad = check_at_init(cur, tol=0.05)
    assert not ok and bad == "joint_4"


def test_not_at_init_height():
    cur = dict(INIT_JOINTS)
    cur["joint_0"] = 0.0  # arm lowered, not at 0.3
    ok, bad = check_at_init(cur, tol=0.05)
    assert not ok and bad == "joint_0"


def test_missing_joint_counts_as_not_at_init():
    cur = dict(INIT_JOINTS)
    del cur["joint_1"]
    ok, bad = check_at_init(cur, tol=0.05)
    assert not ok and bad == "joint_1"


# ---- action plan ----

def test_first_waypoint_sets_gripper():
    plan = build_action_plan([_wp(gripper="open")])
    assert isinstance(plan[0], MoveArm)
    assert isinstance(plan[1], SetGripper)
    assert plan[1].state == "open"


def test_arm_before_gripper():
    plan = build_action_plan([_wp("open"), _wp("closed")])
    # every SetGripper must be preceded by a MoveArm
    for i, action in enumerate(plan):
        if isinstance(action, SetGripper):
            assert isinstance(plan[i - 1], MoveArm)


def test_no_gripper_command_without_change():
    plan = build_action_plan([_wp("open"), _wp("open"), _wp("open")])
    gripper_actions = [a for a in plan if isinstance(a, SetGripper)]
    # only the initial one
    assert len(gripper_actions) == 1


def test_gripper_command_on_change():
    plan = build_action_plan(
        [_wp("open"), _wp("closed"), _wp("closed"), _wp("open")]
    )
    states = [a.state for a in plan if isinstance(a, SetGripper)]
    assert states == ["open", "closed", "open"]


def test_skip_init_filters_init_events():
    wps = [_wp(event="init"), _wp(event="waypoint"), _wp(event="init")]
    move_labels = [
        a.label for a in build_action_plan(wps, skip_init=True)
        if isinstance(a, MoveArm)
    ]
    assert len(move_labels) == 1
