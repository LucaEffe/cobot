"""
Teach recorder entry point (thin ROS glue).

Wires together:
    /joint_states   (this node)      -> live arm pose
    PedalListener   (pedal.py)       -> high-level teach events
    TeachRecorder   (recorder.py)    -> waypoints.jsonl + rosbag
    GripperBackend  (grippers.py)    -> open/close the gripper
    ArmExecutor     (arm.py)         -> optional "drive to init"

All the actual logic lives in those modules; this file only connects them.

Pedal:    F13 waypoint | F14 gripper (+waypoint) | F15 finish
Keyboard: r start | 1 waypoint | 2 gripper | 3 finish | i to init | q quit
"""

from __future__ import annotations

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

from moveit.planning import MoveItPy

from py_utils.planner_utils import (
    wait_for_joint_states,
    generate_moveit_config,
)

from .arm import ArmExecutor
from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PEDAL_DEVICE,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    INIT_JOINTS,
)
from .grippers import make_gripper
from .pedal import PedalListener, WAYPOINT, TOGGLE_GRIPPER, FINISH
from .recorder import TeachRecorder


class JointStateHub(Node):
    """Keeps the latest /joint_states message."""

    def __init__(self):
        super().__init__("recorder_node")
        self.latest = None
        self.create_subscription(
            JointState, "/joint_states", self._cb, 10
        )

    def _cb(self, msg: JointState) -> None:
        self.latest = msg

    def snapshot(self):
        """Return (t, {joint: pos}) or (None, None) if nothing yet."""
        msg = self.latest
        if msg is None:
            return None, None
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return t, dict(zip(msg.name, msg.position))


class TeachSession:
    """Bundles the wiring so pedal + keyboard share the same handlers."""

    def __init__(self, hub, recorder, gripper, arm, logger):
        self.hub = hub
        self.recorder = recorder
        self.gripper = gripper
        self.arm = arm
        self.logger = logger
        self.gripper_closed = False

    @property
    def gripper_state(self) -> str:
        return GRIPPER_CLOSED if self.gripper_closed else GRIPPER_OPEN

    def save_waypoint(self, event: str = "waypoint") -> None:
        t, joints = self.hub.snapshot()
        self.recorder.add_waypoint(joints, event, self.gripper_state, t or 0.0)

    def toggle_gripper(self) -> None:
        self.gripper_closed = not self.gripper_closed
        self.logger.info(f"Gripper -> {self.gripper_state}")
        self.gripper.set_state(self.gripper_state)
        self.save_waypoint("grip" if self.gripper_closed else "release")

    def finish_run(self) -> None:
        # end recording WITHOUT moving the real arm
        self.save_waypoint("last")
        self.recorder.add_fixed_pose(
            INIT_JOINTS, "init", self.gripper_state, 0.0
        )
        self.recorder.finish()

    def move_to_init(self) -> None:
        self.logger.info(">> Moving to init ...")
        self.arm.move_to_init()
        self.logger.info(">> Done.")


def _pedal_thread(session: TeachSession, device_path: str, logger) -> None:
    try:
        listener = PedalListener(device_path)
        logger.info(f"Pedal ready: {listener.name}")
    except Exception as e:  # noqa: BLE001
        logger.warn(f"Pedal not available ({e}) - keyboard only.")
        return

    for event in listener.events():
        if event == WAYPOINT:
            session.save_waypoint("waypoint")
        elif event == TOGGLE_GRIPPER:
            session.toggle_gripper()
        elif event == FINISH:
            session.finish_run()


def main():
    rclpy.init()

    # --- parameters ---
    param_node = Node("pedal_params")
    controller_type = (
        param_node.declare_parameter("controller_type", "real")
        .get_parameter_value().string_value
    )
    pedal_device = (
        param_node.declare_parameter("pedal_device", DEFAULT_PEDAL_DEVICE)
        .get_parameter_value().string_value
    )
    data_root = (
        param_node.declare_parameter("data_root", DEFAULT_DATA_ROOT)
        .get_parameter_value().string_value
    )
    param_node.destroy_node()

    use_real = (controller_type == "real")

    # --- joint states first, then MoveIt (order matters on the real robot) ---
    hub = JointStateHub()
    executor = SingleThreadedExecutor()
    executor.add_node(hub)
    threading.Thread(target=executor.spin, daemon=True).start()

    logger = hub.get_logger()
    logger.info(f"Controller type: {controller_type} (use_real={use_real})")

    wait_for_joint_states(logger)
    time.sleep(2.0)

    moveit_config = generate_moveit_config(use_real=use_real)
    cobot = MoveItPy(node_name="recorder_moveit", config_dict=moveit_config)

    arm = ArmExecutor(cobot, logger)
    gripper = make_gripper(
        use_real=use_real,
        cobot=cobot,
        logger=logger,
        restore_hand_guiding=True,   # teaching: stay hand-guidable
    )
    recorder = TeachRecorder(data_root=data_root, logger=logger)

    session = TeachSession(hub, recorder, gripper, arm, logger)

    threading.Thread(
        target=_pedal_thread, args=(session, pedal_device, logger), daemon=True
    ).start()

    logger.info(
        "Keys:  [r]=start  [1]=waypoint  [2]=gripper(+wp)  "
        "[3]=finish (no move)  [i]=drive to init  [q]=quit"
    )

    try:
        while True:
            key = input().strip().lower()
            if key == "r":
                recorder.start()
            elif key == "1":
                session.save_waypoint("waypoint")
            elif key == "2":
                session.toggle_gripper()
            elif key == "3":
                session.finish_run()
            elif key == "i":
                session.move_to_init()
            elif key == "q":
                break
            else:
                logger.info("Keys: r / 1 / 2 / 3 / i / q")
    except KeyboardInterrupt:
        pass
    finally:
        recorder.finish()
        logger.info("Recorder stopped.")


if __name__ == "__main__":
    main()
