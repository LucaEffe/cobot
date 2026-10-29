#!/usr/bin/env python3
"""
opcua_joint_reader.py -- dense measured joint states from the Festo SPS.

Runs ALONGSIDE the replay. Opens its OWN OPC-UA connection to the cobot (a
second client, exactly like the gripper uses a separate connection), polls the
current joint positions at ~RATE Hz and publishes them as sensor_msgs/JointState
on /measured_joint_states.

This bypasses the ros2_control loop that the SPS blocks during a motion
(cobot_hardware write() blocks in ExecuteInstructionList), so the executed
trajectory is actually sampled WHILE the arm moves -- unlike /joint_states,
which freezes during every motion.

Record /measured_joint_states during the replay; extract_tcp then reads it.

Reads two OPC-UA nodes (from cobot_hardware/config/festo_node_config.ini,
namespace 4):
    dCurrentPoseJ  -> 6 revolute joints, degrees  -> joint_1..6 (rad)
    dCurrentZAxis  -> height axis, mm              -> joint_0 (m)

Run (workspace sourced; `pip install asyncua` done; robot bringup up):
    python3 opcua_joint_reader.py
"""

import math
import time

import rclpy
from sensor_msgs.msg import JointState

from asyncua.sync import Client
from asyncua import ua

# --- OPC-UA server + nodes (see festo_node_config.ini) --------------------
OPCUA_URL = "opc.tcp://192.168.4.1:4840"
NS = 4
NODE_J = "|var|CPX-E-CEC-M1-PN.Application.OPC_UA_Variables_Robot.dCurrentPoseJ"
NODE_Z = "|var|CPX-E-CEC-M1-PN.Application.OPC_UA_Variables_Robot.dCurrentZAxis"

ARM_JOINTS = ["joint_0", "joint_1", "joint_2", "joint_3",
              "joint_4", "joint_5", "joint_6"]
TOPIC = "/measured_joint_states"
RATE = 50.0   # Hz target (actual rate is limited by OPC-UA round-trip time)


def main():
    rclpy.init()
    node = rclpy.create_node("opcua_joint_reader")
    pub = node.create_publisher(JointState, TOPIC, 10)
    log = node.get_logger()

    log.info(f"Connecting to {OPCUA_URL} ...")
    client = Client(OPCUA_URL)
    client.connect()
    log.info(f"Connected. Publishing {TOPIC} at up to {RATE:.0f} Hz ...")

    j_node = client.get_node(ua.NodeId(NODE_J, NS))
    z_node = client.get_node(ua.NodeId(NODE_Z, NS))

    period = 1.0 / RATE
    count = 0
    last_log = time.time()
    try:
        while rclpy.ok():
            t0 = time.time()
            try:
                j = j_node.read_value()   # list of 6 floats (degrees)
                z = z_node.read_value()   # float (mm)
            except Exception as e:  # noqa: BLE001
                log.warn(f"OPC-UA read failed: {e}")
                time.sleep(0.1)
                continue

            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = list(ARM_JOINTS)
            msg.position = [float(z) / 1000.0] + [math.radians(float(v)) for v in j]
            pub.publish(msg)

            count += 1
            if time.time() - last_log >= 2.0:
                log.info(f"{TOPIC}: ~{count / (time.time() - last_log):.0f} Hz")
                count = 0
                last_log = time.time()

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()