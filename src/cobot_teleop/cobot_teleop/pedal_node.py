import os
import json
import signal
import threading
import subprocess
import evdev
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

from py_utils.planner_utils import (
    plan_and_execute, wait_for_joint_states, generate_moveit_config,
)
from moveit.planning import MoveItPy

# --- config ---
ARM_JOINTS = ["joint_0", "joint_1", "joint_2", "joint_3",
              "joint_4", "joint_5", "joint_6"]

INIT_JOINTS = {"joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.0, "joint_3": 0.0,
               "joint_4": 0.0, "joint_5": 0.0, "joint_6": 0.0}

DATA_ROOT = "/workspace/cobot_recordings"

PEDAL_DEVICE = "/dev/input/event27"

# which gripper pedal 2 actuates: finger gripper (visible in sim).
# to use the vacuum instead: GRIPPER_GROUP="vacuum_gripper_group",
#   OPEN_STATE="inactive", CLOSE_STATE="active"
GRIPPER_GROUP = "gripper_group"
OPEN_STATE = "open"
CLOSE_STATE = "close"

# topics recorded continuously by rosbag (the "archive")
BAG_TOPICS = [
    "/joint_states", "/tf", "/tf_static", "/dynamic_joint_states",
    "/arm_group_controller/controller_state",
    "/camera/camera/color/image_raw",
    "/camera/camera/depth/color/points",
]


class Recorder(Node):
    def __init__(self, cobot=None):
        super().__init__("recorder_node")
        self.cobot = cobot

        self.latest_joint_state = None
        self.gripper_closed = False        # finger gripper state we track

        # recording state
        self.recording = False
        self.episode_dir = None
        self.teach_dir = None
        self.traj_file = None
        self.bag_proc = None
        self.count = 0

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    # ---------- joint state ----------
    def _on_joint_state(self, msg):
        self.latest_joint_state = msg

    # ---------- recording lifecycle ----------
    def start_recording(self):
        if self.recording:
            self.get_logger().warn("Already recording.")
            return

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.episode_dir = os.path.join(DATA_ROOT, stamp)
        self.teach_dir = os.path.join(self.episode_dir, "teach")
        os.makedirs(self.teach_dir, exist_ok=True)   # rosbag/ dir must NOT be pre-created

        # open waypoints file
        self.traj_file = open(os.path.join(self.teach_dir, "waypoints.jsonl"), "a")
        self.count = 0

        # start rosbag as its own process group (so we can stop it cleanly later)
        bag_dir = os.path.join(self.teach_dir, "rosbag")
        cmd = ["ros2", "bag", "record", "-o", bag_dir, "--topics"] + BAG_TOPICS
        self.bag_proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, preexec_fn=os.setsid)

        self.recording = True
        self.get_logger().info(f"=== RECORDING STARTED -> {self.teach_dir} ===")

    def save_waypoint(self, event):
        if not self.recording:
            self.get_logger().warn("Not recording. Press 'r' to start.")
            return
        msg = self.latest_joint_state
        if msg is None:
            self.get_logger().warn("No joint state yet - cannot save.")
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        all_pos = dict(zip(msg.name, msg.position))
        arm = {j: round(all_pos.get(j), 6) for j in ARM_JOINTS}

        snapshot = {
            "t": t,
            "event": event,                               # waypoint / grip / release / pre_home / init
            "joints": arm,
            "gripper": "closed" if self.gripper_closed else "open",
        }
        self.traj_file.write(json.dumps(snapshot) + "\n")
        self.traj_file.flush()
        self.count += 1
        self.get_logger().info(f"Waypoint #{self.count} ({event}, gripper={snapshot['gripper']})")

    def append_fixed_pose(self, joints, event):
        if not self.recording:
            return
        msg = self.latest_joint_state
        t = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) if msg else 0.0
        snapshot = {
            "t": t,
            "event": event,
            "joints": {j: round(v, 6) for j, v in joints.items()},
            "gripper": "closed" if self.gripper_closed else "open",
        }
        self.traj_file.write(json.dumps(snapshot) + "\n")
        self.traj_file.flush()
        self.count += 1
        self.get_logger().info(f"Waypoint #{self.count} ({event}, fixed pose)")

    def _stop_bag(self):
        if self.bag_proc is None:
            return
        try:
            # SIGINT = clean Ctrl+C, so rosbag writes metadata.yaml
            os.killpg(os.getpgid(self.bag_proc.pid), signal.SIGINT)
            self.bag_proc.wait(timeout=15)
        except Exception as e:
            self.get_logger().warn(f"Bag stop issue: {e}")
            try:
                self.bag_proc.terminate()
            except Exception:
                pass
        self.bag_proc = None

    def finish_recording(self):
        if not self.recording:
            return
        self._stop_bag()
        if self.traj_file:
            self.traj_file.close()
            self.traj_file = None

        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "mode": "teach",
            "arm_joints": ARM_JOINTS,
            "gripper_group": GRIPPER_GROUP,
            "num_waypoints": self.count,
            "notes": "",
        }
        with open(os.path.join(self.teach_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.get_logger().info(
            f"=== RECORDING FINISHED ({self.count} waypoints) -> {self.episode_dir} ==="
        )
        self.recording = False

    # ---------- gripper ----------
    def _move_group(self, group_name, target):
        if self.cobot is None:
            self.get_logger().warn("MoveIt not ready.")
            return
        comp = self.cobot.get_planning_component(group_name)
        comp.set_start_state_to_current_state()
        comp.set_goal_state(configuration_name=target)
        plan_and_execute(self.cobot, comp, self.get_logger(), sleep_time=1.0)

    def toggle_gripper(self):
        self.gripper_closed = not self.gripper_closed
        target = CLOSE_STATE if self.gripper_closed else OPEN_STATE
        self.get_logger().info(f"Gripper -> {target}")
        self._move_group(GRIPPER_GROUP, target)
        # capture the gripper change as a waypoint
        self.save_waypoint("grip" if self.gripper_closed else "release")

    def close(self):
        self.finish_recording()


def main():
    rclpy.init()

    # recorder first (starts listening to /joint_states)
    recorder = Recorder()
    executor = SingleThreadedExecutor()
    executor.add_node(recorder)
    threading.Thread(target=executor.spin, daemon=True).start()

    # wait for joint states, then set up MoveIt
    wait_for_joint_states(recorder.get_logger())
    time.sleep(2.0)   # give the sim/controllers a moment before MoveIt configures
    moveit_config = generate_moveit_config()
    cobot = MoveItPy(node_name="recorder_moveit", config_dict=moveit_config)
    recorder.cobot = cobot
    arm = cobot.get_planning_component("arm_group")

    def finish_run():
        # Pedal 3: end recording WITHOUT moving the real arm
        recorder.save_waypoint("last")                 # where the arm currently is
        recorder.append_fixed_pose(INIT_JOINTS, "init")  # init as target, no motion
        recorder.finish_recording()

    def move_to_init():
        # separate, deliberate command (key 'i'): actually drive the arm to init
        # TODO (real robot): switch from hand-guiding to active controller BEFORE this
        recorder.get_logger().info(">> Moving to init ...")
        arm.set_start_state_to_current_state()
        arm.set_goal_state(configuration_name="init")
        plan_and_execute(cobot, arm, recorder.get_logger(), sleep_time=3.0)
        recorder.get_logger().info(">> Done.")

    def pedal_loop():
        try:
            dev = evdev.InputDevice(PEDAL_DEVICE)
            recorder.get_logger().info(f"Pedal ready: {dev.name}")
        except Exception as e:
            recorder.get_logger().warn(f"Pedal not available ({e}) - keyboard only.")
            return
        for event in dev.read_loop():
            if event.type == evdev.ecodes.EV_KEY and event.value == 1:
                if event.code == evdev.ecodes.KEY_F13:
                    recorder.save_waypoint("waypoint")
                elif event.code == evdev.ecodes.KEY_F14:
                    recorder.toggle_gripper()
                elif event.code == evdev.ecodes.KEY_F15:
                    finish_run()

    threading.Thread(target=pedal_loop, daemon=True).start()

    recorder.get_logger().info(
        "Keys:  [r]=start   [1]=waypoint   [2]=gripper(+wp)   "
        "[3]=finish (no move)   [i]=drive to init   [q]=quit"
    )
    try:
        while True:
            key = input().strip().lower()
            if key == "r":
                recorder.start_recording()
            elif key == "1":
                recorder.save_waypoint("waypoint")
            elif key == "2":
                recorder.toggle_gripper()
            elif key == "3":
                finish_run()
            elif key == "i":
                move_to_init()
            elif key == "q":
                break
            else:
                recorder.get_logger().info("Keys: r / 1 / 2 / 3 / i / q")
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        recorder.get_logger().info("Recorder stopped.")

if __name__ == "__main__":
    main()