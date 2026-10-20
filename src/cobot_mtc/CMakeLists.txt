cmake_minimum_required(VERSION 3.8)
project(cobot_mtc)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(mtc_demo src/mtc_demo.cpp)
ament_target_dependencies(mtc_demo
  rclcpp
  moveit_task_constructor_core
  moveit_ros_planning_interface
)

install(TARGETS mtc_demo
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()