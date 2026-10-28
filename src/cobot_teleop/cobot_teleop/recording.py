"""
Rosbag episode recording -- the ONE recorder, shared by teach and replay.

BagRecorder    : start/stop a `ros2 bag record` subprocess writing to
                 <dir>/rosbag, stopped cleanly (SIGINT) so metadata.yaml is
                 written. This is the single place that drives rosbag.
EpisodeRecorder: one replay/execution episode under a recording's episodes/
                 folder, with an auto-incremented name and a meta.json.

Teach (recorder.py) and replay (replay_node.py) both go through BagRecorder,
so there is exactly one recording implementation.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime
from typing import List, Optional


class BagRecorder:
    """Records a set of topics to <bag_dir> via `ros2 bag record`."""

    def __init__(self, bag_dir: str, topics: List[str], logger=None):
        self.bag_dir = bag_dir
        self.topics = list(topics)
        self.logger = logger
        self._proc = None

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)

    def start(self, settle: float = 2.0) -> None:
        """Start recording. `settle` waits so rosbag has discovered the topics
        before the robot starts moving (otherwise the first samples are lost)."""
        if self._proc is not None:
            return
        os.makedirs(os.path.dirname(self.bag_dir), exist_ok=True)
        cmd = ["ros2", "bag", "record", "-o", self.bag_dir, "--topics"] + self.topics
        # own process group so we can SIGINT the whole thing cleanly
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, preexec_fn=os.setsid
        )
        self._log(f"Recording bag -> {self.bag_dir}")
        if settle > 0:
            time.sleep(settle)

    def stop(self, timeout: float = 15.0) -> None:
        if self._proc is None:
            return
        try:
            # SIGINT so rosbag writes metadata.yaml cleanly.
            os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
            self._proc.wait(timeout=timeout)
        except Exception as e:  # noqa: BLE001
            self._log(f"Bag stop issue: {e}")
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None
        self._log("Bag recording stopped.")


class EpisodeRecorder:
    """One execution episode under <recording_root>/episodes/<name>_NN/.

    The folder name auto-increments so re-running the same planner never
    overwrites a previous episode.
    """

    def __init__(self, recording_root: str, planner_name: str, role: str,
                 topics: List[str], logger=None):
        self.recording_root = recording_root
        self.planner_name = planner_name
        self.role = role
        self.logger = logger
        self.episodes_root = os.path.join(recording_root, "episodes")
        self.name = self._next_name()
        self.episode_dir = os.path.join(self.episodes_root, self.name)
        self.bag = BagRecorder(
            os.path.join(self.episode_dir, "rosbag"), topics, logger
        )
        self._started: Optional[str] = None

    def _next_name(self) -> str:
        os.makedirs(self.episodes_root, exist_ok=True)
        n = 0
        while os.path.exists(
            os.path.join(self.episodes_root, f"{self.planner_name}_{n:02d}")
        ):
            n += 1
        return f"{self.planner_name}_{n:02d}"

    def start(self) -> None:
        os.makedirs(self.episode_dir, exist_ok=True)
        self._started = datetime.now().isoformat(timespec="seconds")
        self.bag.start()

    def finish(self, extra_meta: Optional[dict] = None) -> None:
        self.bag.stop()
        meta = {
            "created": self._started
            or datetime.now().isoformat(timespec="seconds"),
            "mode": "replay",
            "planner_name": self.planner_name,
            "role": self.role,
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(os.path.join(self.episode_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if self.logger is not None:
            self.logger.info(f"Episode saved -> {self.episode_dir}")