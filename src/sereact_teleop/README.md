# sereact_teleop

Dual-arm teleoperation system using MediaPipe hand tracking to control two UR5e robot arms with Robotiq 2F-85 grippers in RViz simulation.

## Overview

The operator controls both arms simultaneously using hand gestures in front of a webcam. The system maps 2D hand position in the camera frame to 3D end-effector targets, computes inverse kinematics, and visualizes the result in RViz in real time.

Key features:
- **Dead-man switch** — arms only move when the middle finger is extended (3D world landmark distance). Fist closed = arms frozen
- **Gripper control** — thumb-index pinch distance maps to gripper opening
- **Wrist rotation** — hand rotation about the camera axis rotates the end-effector
- **Collision avoidance** — IK solution is rejected before applying if it would bring end-effectors within 35 cm of each other
- **Per-hand wrist recalibration** — wrist neutral angle resets each time a hand re-enters the camera frame
- **Camera overlay** — live webcam feed shows ACTIVE/HOLD status per hand and a red border on collision

## Dependencies

**ROS 2 Humble** and the following packages:

```bash
# ROS packages
sudo apt install ros-humble-ur-description ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher-gui ros-humble-rviz2 ros-humble-xacro

# Python packages
pip install mediapipe ikpy opencv-python numpy
```

## Build

```bash
cd ~/sereact_ws
colcon build
source install/setup.bash
```

## Run

```bash
ros2 launch sereact_teleop teleop.launch.py
```

This starts:
- `robot_state_publisher` — publishes robot URDF to TF
- `rviz2` — visualizes both arms
- `ik_node` — computes IK from target poses and publishes joint states
- `teleop_node` — maps hand state to end-effector targets
- `mediapipe_node` — captures webcam, runs hand tracking, opens camera window

## Controls

| Gesture | Action |
|---|---|
| Middle finger extended | Arm active — follows hand position |
| Fist closed | Arm frozen (dead-man switch) |
| Thumb-index pinch (close) | Close gripper |
| Thumb-index pinch (open) | Open gripper |
| Rotate hand | Rotate end-effector wrist |
| Remove hand from frame | Arm freezes, recalibrates wrist on return |

Left hand controls the left arm, right hand controls the right arm.

## Architecture

```
mediapipe_node  →  /hand_state (Float32MultiArray, 10 values)
                         
teleop_node     →  /left_arm/target_pose  (geometry_msgs/Pose)
                →  /right_arm/target_pose (geometry_msgs/Pose)
                         
ik_node         →  /joint_states (sensor_msgs/JointState)
                →  /collision_state (std_msgs/Bool)
                         
robot_state_publisher → RViz
```

`/hand_state` layout: `[lx, ly, l_pinch, l_angle, rx, ry, r_pinch, r_angle, left_active, right_active]`

## Open-source references

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — Google, Apache 2.0
- [ikpy](https://github.com/Phylliade/ikpy) — Pierre Manceron, MIT
- [ur_description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) — Universal Robots, BSD-3
- [robotiq_description](https://github.com/ros-industrial/robotiq) — ROS-Industrial, Apache 2.0
- [OpenCV](https://opencv.org/) — OpenCV team, Apache 2.0
