// MoveIt Task Constructor pick-and-place for the Festo cobot.
//
// Mirrors the reachable logic from the Python pick_place_demo:
//   * joint_0 (height) is set once (move to init) and then held ~fixed via a
//     path constraint on every arm stage.
//   * a cube is spawned on the table; the gripper/table are allowed to touch
//     it; it is attached on close and detached on open.
//   * motions use fixed, reachable poses (no GenerateGraspPose sampling, which
//     fights the fixed-height arm).
//
// The task is planned and then executed (move_group needs the
// ExecuteTaskSolutionCapability -- your rviz_demo_launch.py loads it).

#include <memory>
#include <thread>
#include <vector>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>

namespace mtc = moveit::task_constructor;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("cobot_mtc");

// --- names (SRDF / config.py) ---
static const std::string ARM_GROUP = "arm_group";
static const std::string GRIPPER_GROUP = "gripper_group";
static const std::string EEF = "end_effector_cobot";
static const std::string IK_FRAME = "gripper_tcp";
static const std::string GRIPPER_OPEN = "open";
static const std::string GRIPPER_CLOSE = "close";
static const std::string HEIGHT_JOINT = "joint_0";
static const double HEIGHT_VALUE = 0.3;
static const double HEIGHT_TOL = 0.01;

static const std::string CUBE_ID = "cube";
static const double CUBE_SIZE = 0.04;

// everything the cube may overlap (gripper head + the table = lin_axis mesh)
static const std::vector<std::string> CUBE_ALLOWED_LINKS = {
    "finger1", "finger2", "gripper_tcp", "segment7",
    "vacuum_gripper_up", "vacuum_gripper_down",
    "lin_axis", "lin_axis_carriage"};

// reachable grasp orientation from the working Python demo
static geometry_msgs::msg::Quaternion graspQuat()
{
  geometry_msgs::msg::Quaternion q;
  q.x = 0.673105;
  q.y = 0.738588;
  q.z = 0.031103;
  q.w = -0.021228;
  return q;
}

static geometry_msgs::msg::PoseStamped toolPose(double x, double y, double z)
{
  geometry_msgs::msg::PoseStamped p;
  p.header.frame_id = "world";
  p.pose.position.x = x;
  p.pose.position.y = y;
  p.pose.position.z = z;
  p.pose.orientation = graspQuat();
  return p;
}

// cube on the table and the tool poses around it / the drop spot
static const double CUBE_X = 0.534, CUBE_Y = -0.285, CUBE_Z = 0.773;
static const double DROP_X = 0.534, DROP_Y = -0.130, DROP_Z = 0.773;
static const double CUBE_DROP = 0.05;   // tool point sits this far above cube
static const double LIFT = 0.10;

static moveit_msgs::msg::Constraints lockHeight()
{
  moveit_msgs::msg::Constraints c;
  c.name = "lock_joint_0";
  moveit_msgs::msg::JointConstraint jc;
  jc.joint_name = HEIGHT_JOINT;
  jc.position = HEIGHT_VALUE;
  jc.tolerance_above = HEIGHT_TOL;
  jc.tolerance_below = HEIGHT_TOL;
  jc.weight = 1.0;
  c.joint_constraints.push_back(jc);
  return c;
}

static void spawnCube(const rclcpp::Logger& logger)
{
  moveit::planning_interface::PlanningSceneInterface psi;
  moveit_msgs::msg::CollisionObject obj;
  obj.id = CUBE_ID;
  obj.header.frame_id = "world";
  shape_msgs::msg::SolidPrimitive box;
  box.type = box.BOX;
  box.dimensions = {CUBE_SIZE, CUBE_SIZE, CUBE_SIZE};
  geometry_msgs::msg::Pose pose;
  pose.position.x = CUBE_X;
  pose.position.y = CUBE_Y;
  pose.position.z = CUBE_Z;
  pose.orientation.w = 1.0;
  obj.primitives.push_back(box);
  obj.primitive_poses.push_back(pose);
  obj.operation = obj.ADD;
  psi.applyCollisionObject(obj);
  RCLCPP_INFO(logger, "Spawned cube on the table.");
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("cobot_mtc", options);

  rclcpp::executors::MultiThreadedExecutor executor;
  auto spin_thread = std::make_unique<std::thread>(
      [&executor, &node]()
      {
        executor.add_node(node);
        executor.spin();
        executor.remove_node(node);
      });

  // put the cube in the scene, then give move_group a moment to see it
  spawnCube(LOGGER);
  rclcpp::sleep_for(std::chrono::seconds(1));

  // --- solvers ---
  auto interpolation = std::make_shared<mtc::solvers::JointInterpolationPlanner>();
  auto sampling = std::make_shared<mtc::solvers::PipelinePlanner>(node);

  // --- build task ---
  mtc::Task task;
  task.stages()->setName("cobot pick and place");
  task.loadRobotModel(node);
  task.setProperty("group", ARM_GROUP);
  task.setProperty("eef", EEF);
  task.setProperty("ik_frame", IK_FRAME);

  // helper to add an arm MoveTo to a tool pose with the height lock
  auto addArmPose = [&](const std::string& name, geometry_msgs::msg::PoseStamped pose)
  {
    auto s = std::make_unique<mtc::stages::MoveTo>(name, sampling);
    s->setGroup(ARM_GROUP);
    s->setIKFrame(IK_FRAME);
    s->setGoal(pose);
    s->setPathConstraints(lockHeight());
    task.add(std::move(s));
  };
  auto addGripper = [&](const std::string& name, const std::string& goal)
  {
    auto s = std::make_unique<mtc::stages::MoveTo>(name, interpolation);
    s->setGroup(GRIPPER_GROUP);
    s->setGoal(goal);
    task.add(std::move(s));
  };

  task.add(std::make_unique<mtc::stages::CurrentState>("current"));

  // allow the cube to touch the gripper head and the table
  {
    auto s = std::make_unique<mtc::stages::ModifyPlanningScene>("allow cube contact");
    s->allowCollisions(CUBE_ID, CUBE_ALLOWED_LINKS, true);
    task.add(std::move(s));
  }

  addGripper("open hand", GRIPPER_OPEN);

  // move once to init (establishes joint_0 = 0.3); no height lock here
  {
    auto s = std::make_unique<mtc::stages::MoveTo>("move to init", sampling);
    s->setGroup(ARM_GROUP);
    std::map<std::string, double> init{
        {"joint_0", 0.3}, {"joint_1", 0.0}, {"joint_2", 0.0}, {"joint_3", 0.0},
        {"joint_4", 0.0}, {"joint_5", 0.0}, {"joint_6", 0.0}};
    s->setGoal(init);
    task.add(std::move(s));
  }

  addArmPose("approach cube", toolPose(CUBE_X, CUBE_Y, CUBE_Z + CUBE_DROP));
  addGripper("close hand", GRIPPER_CLOSE);

  // attach the cube to the gripper
  {
    auto s = std::make_unique<mtc::stages::ModifyPlanningScene>("attach cube");
    s->attachObject(CUBE_ID, IK_FRAME);
    task.add(std::move(s));
  }

  addArmPose("lift", toolPose(CUBE_X, CUBE_Y, CUBE_Z + CUBE_DROP + LIFT));
  addArmPose("carry over drop", toolPose(DROP_X, DROP_Y, DROP_Z + CUBE_DROP + LIFT));
  addArmPose("lower to drop", toolPose(DROP_X, DROP_Y, DROP_Z + CUBE_DROP));
  addGripper("open hand (release)", GRIPPER_OPEN);

  // detach the cube (stays at the drop spot)
  {
    auto s = std::make_unique<mtc::stages::ModifyPlanningScene>("detach cube");
    s->detachObject(CUBE_ID, IK_FRAME);
    task.add(std::move(s));
  }

  addArmPose("lift off", toolPose(DROP_X, DROP_Y, DROP_Z + CUBE_DROP + LIFT));

  // back to init
  {
    auto s = std::make_unique<mtc::stages::MoveTo>("return to init", sampling);
    s->setGroup(ARM_GROUP);
    std::map<std::string, double> init{
        {"joint_0", 0.3}, {"joint_1", 0.0}, {"joint_2", 0.0}, {"joint_3", 0.0},
        {"joint_4", 0.0}, {"joint_5", 0.0}, {"joint_6", 0.0}};
    s->setGoal(init);
    task.add(std::move(s));
  }

  // --- init + plan + execute ---
  try
  {
    task.init();
  }
  catch (mtc::InitStageException& e)
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Task init failed: " << e);
    rclcpp::shutdown();
    if (spin_thread->joinable())
      spin_thread->join();
    return 1;
  }

  if (!task.plan(5))
  {
    RCLCPP_ERROR(LOGGER, "Task planning failed");
  }
  else
  {
    RCLCPP_INFO(LOGGER, "Task planning succeeded - executing ...");
    task.introspection().publishSolution(*task.solutions().front());
    auto result = task.execute(*task.solutions().front());
    if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
      RCLCPP_ERROR(LOGGER, "Execution failed (code %d)", result.val);
    else
      RCLCPP_INFO(LOGGER, "Execution succeeded.");
  }

  spin_thread->join();
  rclcpp::shutdown();
  return 0;
}