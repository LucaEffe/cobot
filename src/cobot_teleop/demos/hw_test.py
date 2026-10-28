"""
Minimal hardware test for the REAL cobot.
First run: gripper only (arm move is commented out).
Run ONLY when: arm in active mode, workspace clear, e-stop in hand,
and gripper init done (ros2 run cobot_hardware gripper_off).
"""
import time
import time as _t
import rclpy
from rclpy.logging import get_logger
from py_utils.planner_utils import (
    plan_and_execute, wait_for_joint_states, generate_moveit_config,
)
from moveit.planning import MoveItPy

logger = get_logger("hw_test")


def main():
    rclpy.init()

    moveit_config = generate_moveit_config(use_real=True)
    cobot = MoveItPy(node_name="hw_test_moveit", config_dict=moveit_config)

    time.sleep(8.0)
    # give the real robot time to publish a fresh state before MoveIt starts
    wait_for_joint_states(logger)
    time.sleep(3.0)

    arm = cobot.get_planning_component("arm_group")
    gripper = cobot.get_planning_component("gripper_group")

    def move_arm(target):
        arm.set_start_state_to_current_state()
        arm.set_goal_state(configuration_name=target)
        plan_and_execute(cobot, arm, logger, sleep_time=3.0)

    def move_gripper(target):
        gripper.set_start_state_to_current_state()
        gripper.set_goal_state(configuration_name=target)
        plan_and_execute(cobot, gripper, logger, sleep_time=2.0)

    # --- STEP 1: arm move (DISABLED for first run) ---
    logger.info(">>> Step 1: move arm to init")
    move_arm("init")
    time.sleep(2.0)

    # --- STEP 2: gripper close ---
    logger.info(">>> Step 2: close gripper")
    move_gripper("close")
    time.sleep(2.0)

    # --- STEP 3: gripper open ---
    logger.info(">>> Step 3: open gripper")
    move_gripper("open")

    logger.info(">>> Done.")


if __name__ == "__main__":
    main()