"""
Teach recording core.

Owns the episode folder, waypoints.jsonl, the rosbag process and meta.json.
It does NOT talk to ROS topics itself -- joint snapshots are pushed in by
the node. Every recording starts with an explicit init waypoint so each
episode is a self-contained init -> task -> init trajectory.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import datetime
from typing import Dict, Optional

from .config import (
    ARM_JOINTS,
    BAG_TOPICS,
    DEFAULT_DATA_ROOT,
    FIXED_JOINT0,
    GRIPPER_GROUP,
    GRIPPER_OPEN,
    INIT_JOINTS,
)
from .waypoints import Waypoint, dump_waypoint


class TeachRecorder:
    def __init__(self, data_root: str = DEFAULT_DATA_ROOT, logger=None):
        self.data_root = data_root
        self.logger = logger

        self.recording = False
        self.episode_dir: Optional[str] = None
        self.teach_dir: Optional[str] = None
        self._traj_file = None
        self._bag_proc = None
        self.count = 0

    # ---- logging helper ----
    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)

    # ---- lifecycle ----

    def start(self) -> None:
        if self.recording:
            self._log("Already recording.")
            return

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.episode_dir = os.path.join(self.data_root, stamp)
        self.teach_dir = os.path.join(self.episode_dir, "teach")
        os.makedirs(self.teach_dir, exist_ok=True)

        self._traj_file = open(
            os.path.join(self.teach_dir, "waypoints.jsonl"), "a"
        )
        self.count = 0

        # rosbag as its own process group so we can stop it cleanly.
        bag_dir = os.path.join(self.teach_dir, "rosbag")
        cmd = ["ros2", "bag", "record", "-o", bag_dir, "--topics"] + BAG_TOPICS
        self._bag_proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, preexec_fn=os.setsid
        )

        self.recording = True
        self._log(f"=== RECORDING STARTED -> {self.teach_dir} ===")

        # init as the explicit first waypoint.
        self._write(
            Waypoint(
                event="init",
                joints=dict(INIT_JOINTS),
                gripper=GRIPPER_OPEN,
                t=0.0,
            )
        )

    def add_waypoint(
        self,
        joints: Dict[str, float],
        event: str,
        gripper: str,
        t: float = 0.0,
    ) -> None:
        """Record the live arm pose (joint_0 forced to the constant height)."""

        if not self.recording:
            self._log("Not recording. Press 'r' to start.")
            return
        if joints is None:
            self._log("No joint state yet - cannot save.")
            return

        arm = {j: float(joints.get(j, 0.0)) for j in ARM_JOINTS}
        arm["joint_0"] = FIXED_JOINT0

        self._write(Waypoint(event=event, joints=arm, gripper=gripper, t=t))

    def add_fixed_pose(
        self,
        joints: Dict[str, float],
        event: str,
        gripper: str,
        t: float = 0.0,
    ) -> None:
        """Record a given pose verbatim (e.g. the final init target)."""

        if not self.recording:
            return
        self._write(
            Waypoint(
                event=event,
                joints={j: float(joints[j]) for j in ARM_JOINTS},
                gripper=gripper,
                t=t,
            )
        )

    def finish(self) -> None:
        if not self.recording:
            return

        self._stop_bag()

        if self._traj_file:
            self._traj_file.close()
            self._traj_file = None

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

        self._log(
            f"=== RECORDING FINISHED ({self.count} waypoints) "
            f"-> {self.episode_dir} ==="
        )
        self.recording = False

    # ---- internals ----

    def _write(self, waypoint: Waypoint) -> None:
        dump_waypoint(self._traj_file, waypoint)
        self.count += 1
        self._log(
            f"Waypoint #{self.count} "
            f"({waypoint.event}, gripper={waypoint.gripper})"
        )

    def _stop_bag(self) -> None:
        if self._bag_proc is None:
            return
        try:
            # SIGINT so rosbag writes metadata.yaml cleanly.
            os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGINT)
            self._bag_proc.wait(timeout=15)
        except Exception as e:  # noqa: BLE001
            self._log(f"Bag stop issue: {e}")
            try:
                self._bag_proc.terminate()
            except Exception:
                pass
        self._bag_proc = None
