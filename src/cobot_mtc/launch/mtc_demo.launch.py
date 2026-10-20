from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Mirror py_utils.generate_moveit_config() exactly, so the MTC node starts
    # with the same MoveIt config as your other nodes.
    moveit_config = (
        MoveItConfigsBuilder(robot_name="cobot")
        .robot_description(
            file_path=get_package_share_directory("cobot_model")
            + "/urdf/festo_cobot_model.urdf.xacro"
        )
        .robot_description_semantic(
            file_path=get_package_share_directory("cobot_moveit_config")
            + "/config/festo_cobot_model.srdf"
        )
        .robot_description_kinematics(
            file_path=get_package_share_directory("cobot_moveit_config")
            + "/config/kinematics.yaml"
        )
        .joint_limits(
            file_path=get_package_share_directory("cobot_moveit_config")
            + "/config/joint_limits.yaml"
        )
        .trajectory_execution(
            file_path=get_package_share_directory("cobot_moveit_config")
            + "/config/moveit_controllers.yaml"
        )
        .planning_pipelines(
            pipelines=["ompl", "chomp", "stomp", "pilz_industrial_motion_planner"],
            default_planning_pipeline="ompl",
        )
        .pilz_cartesian_limits(
            file_path=get_package_share_directory("py_demo")
            + "/config/pilz_cartesian_limits.yaml"
        )
        .moveit_cpp(
            file_path=get_package_share_directory("py_demo") + "/config/moveit_cpp.yaml"
        )
        .to_moveit_configs()
    )

    mtc_node = Node(
        package="cobot_mtc",
        executable="mtc_demo",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription([mtc_node])