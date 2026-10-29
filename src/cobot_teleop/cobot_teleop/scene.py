#!/usr/bin/env python3
"""
scene.py -- shared planning-scene helpers (one source of truth).

Collision objects for the planning scene, reused by the planner comparison
(eval/planner_logger.py) and later by the replay (replay_node.py), so the
PLANNED and the EXECUTED runs see exactly the same obstacles.

The star here is a *virtual barrier*: a thin, fictitious wall placed between the
grasp and the place point. It exists only in MoveIt's planning scene (nothing is
physically there), so on the real robot the arm simply arcs a bit higher -- it
forces the planners to find a way OVER/AROUND instead of the trivial straight
line, which is what makes different planners produce visibly different paths.

The geometry is derived from the two end-effector points you pass in (grasp and
place in world coordinates), so there are no hard-coded positions -- move the
task and the wall moves with it.
"""

import math

import numpy as np

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA

# Planning frame (matches pick_place_demo.py / the SRDF).
BASE_FRAME = "world"

BARRIER_ID = "barrier"


def make_box(obj_id, position, size, frame=BASE_FRAME, quat=(0.0, 0.0, 0.0, 1.0)):
    """A moveit_msgs/CollisionObject BOX at `position` with `size` (x,y,z)."""
    obj = CollisionObject()
    obj.header.frame_id = frame
    obj.id = obj_id
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [float(size[0]), float(size[1]), float(size[2])]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = (float(q) for q in quat)
    obj.primitives.append(box)
    obj.primitive_poses.append(pose)
    obj.operation = CollisionObject.ADD
    return obj


def color_for(obj_id, rgba):
    c = ObjectColor()
    c.id = obj_id
    c.color = ColorRGBA(r=float(rgba[0]), g=float(rgba[1]),
                        b=float(rgba[2]), a=float(rgba[3]))
    return c


def barrier_geometry(grasp_xyz, place_xyz,
                     thickness=0.02, width=0.30,
                     clearance=0.05, drop_below=0.15, ref_top_z=None):
    """Geometry of the wall that blocks the direct grasp->place line.

    The wall is placed at the midpoint of the two points, turned so that it
    stands *across* the travel direction (thin in the direction of motion), and
    tall enough that the travel path would hit it:

      * top    = `ref_top_z` (or highest of the two points) + `clearance`
      * bottom = lowest of the two points - `drop_below`   (reaches down so
                 going under is not a cheap option)

    Pass `ref_top_z` = the height the arm actually travels at (the lifted
    transport height, i.e. the highest waypoint between grasp and place). The
    grasp/place points themselves are lower (the arm lifts before carrying), so
    keying the top off them makes the wall too short to block anything.

    Returns (position xyz, quaternion xyzw, dims xyz).
    """
    g = np.asarray(grasp_xyz, dtype=float)
    p = np.asarray(place_xyz, dtype=float)
    mid = (g + p) / 2.0

    # yaw so the wall's thin axis (local x) aligns with the travel direction
    yaw = math.atan2(float(p[1] - g[1]), float(p[0] - g[0]))
    quat = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    base_top = float(ref_top_z) if ref_top_z is not None else max(float(g[2]), float(p[2]))
    top = base_top + clearance
    bottom = min(float(g[2]), float(p[2])) - drop_below
    height = max(top - bottom, 0.01)
    cz = (top + bottom) / 2.0

    position = (float(mid[0]), float(mid[1]), cz)
    dims = (float(thickness), float(width), float(height))
    return position, quat, dims


def add_barrier(cobot, grasp_xyz, place_xyz, logger,
                thickness=0.02, width=0.30, clearance=0.05, drop_below=0.15,
                ref_top_z=None, color=(0.95, 0.85, 0.1, 0.6)):
    """Insert the virtual barrier into the planning scene. Returns its geometry."""
    position, quat, dims = barrier_geometry(
        grasp_xyz, place_xyz, thickness, width, clearance, drop_below, ref_top_z)
    psm = cobot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        obj = make_box(BARRIER_ID, position, dims, BASE_FRAME, quat)
        scene.apply_collision_object(obj, color_for(BARRIER_ID, color))
        scene.current_state.update()
    if logger is not None:
        logger.info(
            f"Barrier '{BARRIER_ID}' at {np.round(position, 3).tolist()} "
            f"dims {np.round(dims, 3).tolist()} (thin axis across travel).")
    return position, quat, dims


def remove_barrier(cobot, logger=None):
    """Remove the virtual barrier from the planning scene."""
    psm = cobot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        obj = CollisionObject()
        obj.id = BARRIER_ID
        obj.header.frame_id = BASE_FRAME
        obj.operation = CollisionObject.REMOVE
        scene.apply_collision_object(obj)
        scene.current_state.update()
    if logger is not None:
        logger.info(f"Barrier '{BARRIER_ID}' removed.")