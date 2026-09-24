# cobot_teleop — Teach-in & Replay for the Festo Cobot

Teach-by-demonstration for the Festo cobot: hand-guide the arm, **record** the
motion as a dataset, and **replay** it. Runs both in **simulation** and on the
**real cobot**, switchable via a single parameter.

Part of a master's thesis (teleoperation / teach recorder + replay). Example
task: stacking one cardboard box on another.

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites & setup](#prerequisites--setup)
- [Build](#build)
- [Usage](#usage)
  - [Teach (record)](#teach-record)
  - [Replay](#replay)
- [Data format](#data-format)
- [Safety](#safety)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Open items](#open-items)

---

## Overview

Two nodes in the `cobot_teleop` package:

| Node          | Purpose                                                                 |
|---------------|-------------------------------------------------------------------------|
| `pedal_node`  | Teach recorder: hand-guide the arm, use a foot pedal to save waypoints and toggle the gripper; everything is written to one recording. |
| `replay_node` | Replay: load a recording and drive it with MoveIt (arm + gripper).      |

**Sim vs. real robot** is chosen in one place (`controller_type` /
`--controller-type`):

- `sim`  → simulation controllers + gripper via MoveIt.
- `real` → real controllers + gripper via standalone OPC-UA commands.

---

## Architecture

Pure logic (no ROS, unit-testable) is kept separate from the ROS/hardware
adapters, with **thin nodes** that only wire things together.

```
cobot_teleop/
├── config.py        # constants (joints, INIT, tolerances, gripper states, defaults)
├── waypoints.py     # Waypoint dataclass + JSONL load/save + validation      [pure]
├── replay_core.py   # init check + action plan (arm-then-gripper ordering)   [pure]
├── arm.py           # ArmExecutor (MoveIt), plan_and_execute, read joints
├── grippers.py      # GripperBackend + MoveItGripper (sim) + OpcUaGripper (real)
├── recorder.py      # recording core: waypoints.jsonl + rosbag + meta.json
├── pedal.py         # foot pedal listener (evdev)
├── pedal_node.py    # teach entry point (thin ROS glue)
└── replay_node.py   # replay entry point (thin ROS glue)
test/
├── test_waypoints.py     # format & validation — no robot
└── test_replay_core.py   # init gate & ordering — no robot
```

**Why:** the pure core is testable without a robot; the only real sim/real
difference is the gripper, so there is exactly one abstraction (`GripperBackend`)
instead of scattered `if real:` branches; safety (the init gate) is runtime
logic, not baked into the data format.

---

## Prerequisites & setup

- ROS 2 Jazzy, MoveIt 2 (MoveItPy), ros2_control — provided by the dev container.
- The real cobot speaks OPC UA (`opc.tcp://192.168.4.1:4840`); the standalone
  gripper/balancer commands come from the (private) `cobot_hardware` submodule.

**After every container rebuild** (for the foot pedal), run **inside the
container**:

```bash
pip install evdev --break-system-packages
sudo chmod a+r /dev/input/event27
```

> The pedal must be plugged in **before** the container starts — USB hotplug
> does not reach a running container. If `/dev/input/event27` is missing inside
> the container, plug the pedal in and restart the container
> (`docker restart <id>`).

---

## Build

```bash
cd /workspace
colcon build --packages-select cobot_teleop --symlink-install
source install/setup.bash
```

`.py` changes take effect on node restart (symlink-install); new files need one
`colcon build`.

---

## Usage

Always start the matching **bringup** first (publishes `/joint_states` and the
controllers), then the node.

**Bringup:**
```bash
# real cobot
ros2 launch demo rviz_demo_launch.py controller_type:=real enable_realsense_camera:=false
# simulation
ros2 launch demo rviz_demo_launch.py
```

### Teach (record)

```bash
# real cobot (default controller_type=real)
ros2 run cobot_teleop pedal_node
# simulation
ros2 run cobot_teleop pedal_node --ros-args -p controller_type:=sim
```

| Pedal   | Key | Action                            |
|------  -|-----|-----------------------------------|
| left    | `1` | Save waypoint                     |
| middle  | `2` | Toggle gripper (+ waypoint)       |
| right   | `3` | Finish recording (no arm motion)  |
|         | `r` | Start recording                   |
|         | `i` | Drive to init                     |
|         | `q` | Quit                              |

Each recording **starts with an `init` waypoint** and is written to
`/workspace/cobot_recordings/<timestamp>/teach/`. `joint_0` (height axis) is
held constant while teaching.

**Gripper while teaching:** driven by standalone OPC-UA commands (not MoveIt,
which froze `/joint_states`). Since the gripper only actuates with the
controller active (**balancer OFF**), each toggle does
`balancer off → gripper → balancer on`, so the arm stays hand-guidable.

### Replay

```bash
# simulation, slowed down to check the motion
ros2 run cobot_teleop replay_node <path>/teach/waypoints.jsonl --move-to-init --speed 0.2

# real cobot, slow, only from init (aborts otherwise)
ros2 run cobot_teleop replay_node <path>/teach/waypoints.jsonl --controller-type real --speed 0.1
```

| Flag                | Default     | Meaning                                                     |
|---------------------|-------------|-------------------------------------------------------------|
| `--controller-type` | `sim`       | `sim` or `real` (controllers + gripper backend).            |
| `--move-to-init`    | off         | If not at init, home to init first instead of aborting.     |
| `--speed`           | `1.0`       | Velocity/acceleration scaling `0..1` (e.g. `0.1` = 10 %).   |
| `--init-tol`        | `0.05`      | Init-check tolerance (rad; m for `joint_0`).                |
| `--skip-init`       | off         | Do not execute the synthetic `init` waypoint from the file. |
| `--planner-config`  | `ompl_rrtc` | `plan_request_params` block used for speed scaling.         |

**Replay rule:** per waypoint, reach the **arm** pose first, then change the
**gripper** — only on the first waypoint or on an actual change.

**Gripper during replay:** `balancer off → gripper` (balancer is *not* turned
back on, so the arm stays actively controlled). In sim the gripper goes through
MoveIt.

---

## Data format

Per recording: `/workspace/cobot_recordings/<timestamp>/teach/` with
`waypoints.jsonl` (one line per waypoint), `rosbag/` (raw topic archive) and
`meta.json`.

```json
{"t": 1694.7, "event": "waypoint", "joints": {"joint_0": 0.3, "joint_1": 0.0, "...": 0.0, "joint_6": 0.0}, "gripper": "open"}
```

- `event`: `init` | `waypoint` | `grip` | `release` | `last`
- `joints`: all 7 axes (`joint_0` is the prismatic height axis, kept constant)
- `gripper`: `open` | `closed`

---

## Safety

- **Init gate:** replay only runs when the arm is at the init pose
  (`joint_0=0.3`, rest `0`); otherwise it aborts (no motion from an arbitrary
  pose). `--move-to-init` homes to init first instead.
- Use a small `--speed` for first hardware runs, and keep the **e-stop in hand**.

---

## Tests

```bash
cd /workspace/src/cobot_teleop
python -m pytest test/ -q
```

Covers the waypoint format/validation, the init check, and the replay ordering —
all without a robot.

---

## Troubleshooting

- **`Action client not connected: arm_group_controller/...`** — on the real
  robot without `--controller-type real` (sim controllers loaded).
- **`/joint_states` freezes on gripper** — do not drive the gripper via MoveIt;
  the standalone OPC-UA path avoids this.
- **Gripper hangs (`Waiting for controller status == 1`)** — balancer is on; the
  nodes force it off before gripping.
- **Pedal not detected** — `evdev` installed? `chmod` done inside the container?
  `event27` present (else restart with the pedal plugged in)? Event number
  changed? `grep -A5 -i foot /proc/bus/input/devices`.
- **"Could not apply speed scaling"** — the `plan_request_params` block is named
  differently; pass `--planner-config <name>`.

---

## Open items

- Replay on the **real cobot** not yet fully validated (teach + gripping work).
- Verify the arm stays actively controlled around gripper commands during replay.
- Test two parallel OPC-UA connections (standalone gripper + running real launch).
- Tune the gripper close value (`0.5`) and init values against the real hardware.
- Cross-check `rosbag` ↔ `waypoints.jsonl` timestamps.
- Single-command launch for the whole teach stack.
