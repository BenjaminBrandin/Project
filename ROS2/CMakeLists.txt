cmake_minimum_required(VERSION 3.5)
project(sml_nexus_tutorials)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sml_nexus_gazebo REQUIRED)
find_package(custom_msg REQUIRED)

# Uncomment this if you have Python scripts and need to install them
# find_package(python_cmake_module REQUIRED)
# find_package(Python3 REQUIRED COMPONENTS Interpreter)

################################################
## Declare ROS messages, services and actions ##
################################################

# Uncomment and adapt these lines if you have custom messages/services/actions
# find_package(rosidl_default_generators REQUIRED)
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/Message1.msg"
#   "srv/Service1.srv"
#   "action/Action1.action"
#   DEPENDENCIES geometry_msgs
# )

###################################
## ament specific configuration  ##
###################################

ament_package()

###########
## Build ##
###########

# Specify additional locations of header files
# Your package locations should be listed before other locations
include_directories(
  # include
  ${rclcpp_INCLUDE_DIRS}
  ${gazebo_ros_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${tf2_geometry_msgs_INCLUDE_DIRS}
  ${sml_nexus_gazebo_INCLUDE_DIRS}
  ${custom_msg_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/sml_nexus_tutorial.cpp
# )

# Add cmake target dependencies of the library
# as an example, code may need to be generated before libraries
# either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${rclcpp_EXPORTED_TARGETS})

## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/sml_nexus_tutorial_node.cpp)

# Rename C++ executable without prefix
set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

# Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
  ${rclcpp_LIBRARIES}
  ${gazebo_ros_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${tf2_geometry_msgs_LIBRARIES}
  ${sml_nexus_gazebo_LIBRARIES}
  ${custom_msg_LIBRARIES}
)

#############
## Install ##
#############

# Mark executables for installation
install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# Mark other files for installation (e.g. launch and config files)
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

#############
## Testing ##
#############

# Add gtest based cpp test target and link libraries
# ament_add_gtest(${PROJECT_NAME}-test test/test_sml_nexus_tutorial.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

# Add folders to be run by python nosetests
# ament_add_nose_test(test)
