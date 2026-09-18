cmake_minimum_required(VERSION 2.8.3)
project(aloha)

## 编译为 C++11，ROS Kinetic 及更新版本支持
add_compile_options(-std=c++11)

## 查找 catkin 宏和库
## 如果使用了类似 find_package(catkin REQUIRED COMPONENTS xyz) 的 COMPONENTS 列表
## 也会查找其他 catkin 包
find_package(catkin REQUIRED COMPONENTS
  interbotix_xs_msgs
  interbotix_xsarm_control
  rosbag
  roscpp
  tf2_ros
  rviz
  sensor_msgs
  std_msgs
  std_srvs
)

add_message_files(
  FILES
  RGBGrayscaleImage.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

###################################
## catkin 特定配置 ##
###################################
## catkin_package 宏会为包生成 cmake 配置文件
## 声明要传递给依赖项目的各项内容
catkin_package(
 CATKIN_DEPENDS interbotix_xs_msgs interbotix_xsarm_control rosbag roscpp tf2_ros rviz sensor_msgs std_msgs std_srvs
)

###########
## 构建 ##
###########

## 指定头文件的额外位置
## 你的包位置应列在其他位置之前
include_directories(
  ${catkin_INCLUDE_DIRS}
)


