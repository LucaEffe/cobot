# cobot_teleop — Teach-in, Replay & Evaluation for the Festo Cobot

Teach-by-demonstration for the Festo cobot: hand-guide the arm, **record** the
motion as a dataset, **replay** it with a motion planner, and **evaluate** the
result (demonstration vs. planned vs. executed path). Runs both in
**simulation** and on the **real cobot**, switchable via a single parameter.

Part of a master's thesis (teleoperation / teach recorder + replay, toward a VLA
training dataset). Example task: stacking one cardboard box on another.

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites & setup](#prerequisites--setup)
- [Build](#build)
- [Usage](#usage)
  - [Teach (record)](#teach-record)
  - [Replay](#replay)
  - [Recording a replay (episodes) — real robot](#recording-a-replay-episodes--real-robot)
- [Evaluation (planner comparison & three-layer plot)](#evaluation-planner-comparison--three-layer-plot)
- [Data format](#data-format)
- [Safety](#safety)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Open items](#open-items)

---

## Overview

Two recording roles write **raw** data to `cobot_recordings/`; the consumers in
`eval/` only read that raw data and write derived results to `eval/results/`.

| Node          | Purpose                                                                 |
|---------------|-------------------------------------------------------------------------|
| `pedal_node`  | Teach recorder: hand-guide the arm, use a foot pedal to save waypoints and toggle the gripper; everything is written to one recording. |
| `replay_node` | Replay: load a recording and drive it with MoveIt (arm + gripper); with `--record` it also saves the executed run as an episode. |

**Sim vs. real robot** is chosen in one place (`controller_type` /
`--controller-type`):

- `sim`  → simulation controllers + gripper via MoveIt.
- `real` → real controllers + gripper via standalone OPC-UA commands.

---

## Architecture

Pure logic (no ROS, unit-testable) is kept separate from the ROS/hardware
adapters, with **thin nodes** that only wire things together. Recording is one
shared component used by both teach and replay; `eval/` holds only consumers.

```
cobot_teleop/
├── config.py            # constants (joints, INIT, tolerances, gripper states, record topics)
├── waypoints.py         # Waypoint dataclass + JSONL load/save + validation      [pure]
├── replay_core.py       # init check + action plan (arm-then-gripper ordering)   [pure]
├── arm.py               # ArmExecutor (MoveIt); plan+execute; waits until the arm arrives
├── grippers.py          # GripperBackend + MoveItGripper (sim) + OpcUaGripper (real)
├── recording.py         # shared rosbag recorder: BagRecorder / EpisodeRecorder
├── recorder.py          # teach recording core: waypoints.jsonl + rosbag + meta.json
├── opcua_joint_reader.py# 2nd OPC-UA connection -> dense /measured_joint_states (also DURING motion)
├── pedal.py             # foot pedal listener (evdev)
├── pedal_node.py        # teach entry point (thin ROS glue)
└── replay_node.py       # replay entry point (thin ROS glue)
eval/                    # consumers: read recordings, write results (never record)
├── planner_logger.py    # plan the route with every planner (no robot) -> tcp_path_<planner>.csv
├── extract_tcp.py       # any rosbag (teach or episode) -> TCP CSV via forward kinematics
├── plot_comparison.py   # overlay all planners + metrics table
├── plot_layers.py       # demonstration vs planned vs executed (+ tracking error)
├── check_timing.py      # timing / stream sync of a teach recording
└── results/<alias>/     # generated CSVs, PNGs, HTML (reproducible)
test/
├── test_waypoints.py    # format & validation — no robot
└── test_replay_core.py  # init gate & ordering — no robot
```

**Why:** the pure core is testable without a robot; the only real sim/real
difference is the gripper, so there is exactly one abstraction (`GripperBackend`)
instead of scattered `if real:` branches; there is exactly **one** way to record
(`recording.py`, used by teach and replay); and `eval/` never touches the robot,
so evaluations are reproducible from the raw recordings.

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
> (`docker restart <id>`). Check ID with (`docker ps`).

**For the dense executed reader** (real robot only), once:

```bash
pip install asyncua --break-system-packages
```

**For plotting**, the eval plotters use an isolated venv `~/plotenv` (numpy +
matplotlib), because the system matplotlib conflicts with NumPy 2; the
interactive Plotly HTML runs in the ROS venv.

---

## Build

```bash
cd /workspace
colcon build --packages-select cobot_teleop --symlink-install --merge-install
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
|---------|-----|-----------------------------------|
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
| `--record`          | off         | Save the executed run as an episode under the recording's `episodes/`. |
| `--role`            | `production`| `comparison` (motion topics only) or `production` (full topics incl. cameras → training data). |
| `--pipeline`        | `ompl`      | MoveIt planning pipeline: `ompl` / `chomp` / `stomp` / `pilz_industrial_motion_planner`. |
| `--planner-id`      | `RRTstarkConfigDefault` | Planner id (OMPL/Pilz); leave empty for CHOMP/STOMP. |

**Replay rule:** per waypoint, reach the **arm** pose first, then change the
**gripper** — only on the first waypoint or on an actual change.

**Gripper during replay:** `balancer off → gripper` (balancer is *not* turned
back on, so the arm stays actively controlled). In sim the gripper goes through
MoveIt.

> **Why replay waits for arrival:** `arm.py` blocks after each waypoint until the
> arm has actually reached it. The `CobotTrajectoryController` acks the
> FollowJointTrajectory action *immediately*, so without waiting the next goal
> would overwrite the controller's single-slot buffer and intermediate waypoints
> get skipped — the arm drives a straight line through them / drags the object.

### Recording a replay (episodes) — real robot

A replay can be recorded as an **episode** next to the teach recording:

```
cobot_recordings/<timestamp>/episodes/<planner>_NN/   # rosbag/ + meta.json (planner, role, speed, source)
```

On the **real** cobot two extra steps are required, otherwise the executed path
is wrong:

1. **Full-trajectory mode.** Otherwise the SPS plans point-to-point itself
   (straight lines). The controller re-reads the parameter live, no restart:
   ```bash
   ros2 param set /cobot_arm_group_controller execution_mode full_trajectory
   ros2 param set /cobot_arm_group_controller resampling_delta 0.05   # smaller = finer path
   ```
2. **Dense joint capture.** `/joint_states` freezes while the SPS executes a move
   (the hardware `write()` blocks in `ExecuteInstructionList`). A second OPC-UA
   connection reads the joints independently and publishes them at ~50 Hz on
   `/measured_joint_states`, which is recorded with the episode:
   ```bash
   python3 src/cobot_teleop/cobot_teleop/opcua_joint_reader.py   # run alongside the replay
   ```
   `extract_tcp.py` uses this dense topic automatically when present; teach bags
   (no blocking SPS) stay on `/joint_states`.

Full recorded replay on the real robot:
```bash
# Terminal A — dense reader
python3 src/cobot_teleop/cobot_teleop/opcua_joint_reader.py
# Terminal B — mode + recorded replay
ros2 param set /cobot_arm_group_controller execution_mode full_trajectory
ros2 run cobot_teleop replay_node <path>/teach/waypoints.jsonl \
    --controller-type real --speed 0.1 --move-to-init \
    --record --role comparison --pipeline ompl --planner-id RRTConnectkConfigDefault
# note the episode number:  ls <path>/episodes/
```

---

## Evaluation (planner comparison & three-layer plot)

Offline consumers in `eval/`. `<alias>` is a short name; results land in
`eval/results/<alias>/` — use the **same alias** for every step. The ROS steps
need a bringup (sim is fine) so the robot model loads for forward kinematics; the
plotters run in `~/plotenv`.

```bash
REC=/workspace/cobot_recordings/<timestamp>
EVAL=/workspace/src/cobot_teleop/eval
ALIAS=pickplace

# planned paths — all planners, no robot motion
python3 $EVAL/planner_logger.py $REC $ALIAS

# demonstration + executed layers (rosbag -> TCP CSV)
python3 $EVAL/extract_tcp.py $REC/teach                   $EVAL/results/$ALIAS/demo.csv
python3 $EVAL/extract_tcp.py $REC/episodes/rrtconnect_00  $EVAL/results/$ALIAS/executed_rrtconnect.csv

# plots  (HTML in ROS venv; PNG via ~/plotenv)
python3 $EVAL/plot_comparison.py $ALIAS                   # planner overlay + metrics.csv
python3 $EVAL/plot_layers.py $ALIAS --planner rrtconnect  # demo/planned/executed + tracking error
firefox $EVAL/results/$ALIAS/three_layers_rrtconnect.html
```

| Script | Purpose |
|---|---|
| `planner_logger.py` | plan the route with every planner (no robot) → `tcp_path_<planner>.csv` |
| `extract_tcp.py` | rosbag (teach/episode) → `x,y,z + joints` CSV via FK (prefers `/measured_joint_states`) |
| `plot_comparison.py` | all planners overlaid (3D + 2D projections) + metrics table |
| `plot_layers.py` | demonstration vs planned vs executed (3D + projections + tracking error) |
| `check_timing.py` | per-topic rate/jitter, stream sync, waypoints ↔ rosbag match |

> The VNC/docker browser lacks WebGL, so open the Plotly `*.html` with `firefox`.

---

## Data format

Per recording under `/workspace/cobot_recordings/<timestamp>/`:

```
teach/        waypoints.jsonl, rosbag/, meta.json           # teleop demonstration (raw)
episodes/     <planner>_NN/ : rosbag/ + meta.json           # one folder per recorded replay
```

`teach/waypoints.jsonl` — one line per waypoint:

```json
{"t": 1694.7, "event": "waypoint", "joints": {"joint_0": 0.3, "joint_1": 0.0, "...": 0.0, "joint_6": 0.0}, "gripper": "open"}
```

- `event`: `init` | `waypoint` | `grip` | `release` | `last`
- `joints`: all 7 axes (`joint_0` is the prismatic height axis, kept constant)
- `gripper`: `open` | `closed`

Recorded topics are role-dependent (`config.bag_topics_for_role`): `comparison`
records motion only; `production` adds the cameras and becomes training data.
`/measured_joint_states` (the dense OPC-UA stream) is recorded in both.

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
  robot without `--controller-type real` (sim controllers loaded), or the arm is
  not at init (init gate; add `--move-to-init`).
- **Real arm drives a straight line / skips waypoints / drags the object** — set
  `execution_mode=full_trajectory` on `cobot_arm_group_controller`, and make sure
  the current `arm.py` (waits for arrival) is installed.
- **`executed_*.csv` has very few points / looks straight** — `/joint_states`
  freezes during SPS motion; run `opcua_joint_reader.py` alongside the replay so
  the dense `/measured_joint_states` is recorded.
- **`/joint_states` freezes on gripper** — do not drive the gripper via MoveIt;
  the standalone OPC-UA path avoids this.
- **Gripper hangs (`Waiting for controller status == 1`)** — balancer is on; the
  nodes force it off before gripping.
- **Pedal not detected** — `evdev` installed? `chmod` done inside the container?
  `event27` present (else restart with the pedal plugged in)? Event number
  changed? `grep -A5 -i foot /proc/bus/input/devices`.
- **"Could not apply speed scaling"** — the `plan_request_params` block is named
  differently; pass `--planner-config <name>`.
- **matplotlib / NumPy 2 conflict or "WebGL not supported"** — run the plotters
  via `~/plotenv`, and open Plotly HTML with `firefox`.

---

## Open items

- Record episodes for the other planners (reader + replay per planner) to get a
  real *executed* planner comparison, not just planned paths.
- Multiple runs per planner (`RUNS > 1`) for boxplots / variance.
- Tune the gripper close value (`0.5`) and init values against the real hardware.
- Single-command launch for the whole teach stack.