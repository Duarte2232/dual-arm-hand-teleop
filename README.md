# Dual-arm hand teleoperation (ROS 2)

Control two UR5e arms with two Robotiq 2F-85 grippers using only your hands and a webcam. MediaPipe tracks the hands, a small ROS 2 node graph turns them into end-effector targets, inverse kinematics (ikpy) solves the joints at 20 Hz, and the result is shown live in RViz. Simulation only (ROS 2 Humble, Python).

## Results

Measured on a laptop CPU, in simulation, without a camera or RViz (scripts in `scripts/`):

| check | result |
|---|---|
| Headless pipeline test (`headless_pipeline_test.py`): fake hand to `/hand_state`, real `teleop_node` and `ik_node` | 9 of 9 checks pass: arms follow the hand, grippers close with the pinch, the dead-man switch freezes the inactive arm (0.0000 rad of motion), the collision guard triggers when the end-effectors meet |
| IK time per tick (`ik_workspace_check.py`, same chain, home pose and 10 warm-started iterations as `ik_node`) | median 14 to 20 ms, 95th percentile 20 to 51 ms over three runs (it varies with CPU load), against a 50 ms tick at 20 Hz; the slowest ticks can therefore occasionally miss a tick |
| IK reachability over the teleop workspace (143 targets per arm, x = 0.4 m, y and z swept) | every target reached within 1 cm; median error below 0.01 mm, worst case 9.1 mm |

**Not measured:** camera-to-joint-state latency, tracking accuracy of the hands, behaviour on a real robot. IK error is position only; the tool's Z axis is constrained to point down, and the full orientation is not.

## How it works

```
mediapipe_node  ->  /hand_state (Float32MultiArray, 10 values)
                         |
teleop_node     ->  /left_arm/target_pose, /right_arm/target_pose (geometry_msgs/Pose)
                         |
ik_node         ->  /joint_states (sensor_msgs/JointState), /collision_state (std_msgs/Bool)
                         |
robot_state_publisher -> RViz
```

`/hand_state` layout: `[lx, ly, l_pinch, l_angle, rx, ry, r_pinch, r_angle, left_active, right_active]`.

- **Hand tracking.** The camera runs in a background thread with MediaPipe Hand Landmarker (two hands). Wrist position, thumb-index pinch distance and a wrist rotation angle are low-pass filtered; the left and right labels are swapped because MediaPipe reports them from the camera's point of view.
- **Dead-man switch.** An arm only moves while that hand's middle finger is extended (3D world-landmark distance between fingertip and knuckle above 6 cm). A closed fist freezes the arm; a hand that leaves the frame also freezes it.
- **Targets.** The hand's position in the image maps to the target's `y` and `z` inside a per-arm workspace (left arm: left half, right arm: right half); depth is fixed at `x = 0.4 m`. Thumb-index pinch maps to gripper opening, hand rotation to the wrist joint. A small deadzone ignores tremor, and the wrist neutral angle is recalibrated each time a hand re-enters the frame.
- **Inverse kinematics.** One ikpy chain per arm, built from the same URDF. Each tick the solver starts from 85% of the current joints plus 15% of a home pose (this keeps the elbow up), runs 10 iterations, and joint speed is limited to 0.12 rad per tick.
- **Collision guard.** A new solution is rejected if it would bring the two end-effectors within 35 cm of each other; the camera window then shows a red border.

## Run

Requires ROS 2 Humble.

```bash
# ROS packages
sudo apt install ros-humble-ur-description ros-humble-robotiq-description \
  ros-humble-robot-state-publisher ros-humble-joint-state-publisher-gui \
  ros-humble-rviz2 ros-humble-xacro

# Python packages
pip install mediapipe ikpy opencv-python numpy

# build and launch (this repository is the workspace; it has a src/ folder)
colcon build
source install/setup.bash
ros2 launch dual_arm_hand_teleop teleop.launch.py
```

The launch file starts `robot_state_publisher`, `rviz2`, `ik_node`, `teleop_node` and `mediapipe_node` (it opens the webcam window).

| Gesture | Action |
|---|---|
| Middle finger extended | Arm active, follows the hand |
| Fist closed | Arm frozen (dead-man switch) |
| Thumb-index pinch | Gripper closes; open the pinch to open it |
| Rotate hand | Rotates the end-effector wrist |
| Remove hand from frame | Arm freezes, wrist recalibrates on return |

Left hand controls the left arm, right hand the right arm.

## Tests and checks

```bash
# no camera, no RViz: IK reachability and timing (needs only ikpy and numpy)
python3 scripts/ik_workspace_check.py

# no camera, no RViz: full node graph against a fake hand (needs the built package)
source install/setup.bash
ros2 run dual_arm_hand_teleop ik_node.py &
ros2 run dual_arm_hand_teleop teleop_node.py &
python3 scripts/headless_pipeline_test.py

# webcam only, no ROS: look at what the hand tracker sees
python3 scripts/check_hand_tracking.py
```

## Limitations

- **Depth is not controlled.** Hand position is 2D, so the end-effector moves in a vertical plane at a fixed `x = 0.4 m`.
- **`geometry_msgs/Pose` is overloaded.** `orientation.z` carries the wrist angle and `orientation.w` the gripper opening; it is not a real quaternion. A custom message would be cleaner.
- **Collision check is coarse.** It only compares the two end-effector positions (35 cm); it does not check the arms' links, the table or joint limits beyond clipping to plus or minus 2 pi.
- **Calibration is hand-tuned.** The pinch range (0.056 to 0.36, normalised image units) was set by trial for one operator and one camera.
- **Generated file.** `urdf/dual_arm.urdf` is the output of `xacro urdf/dual_arm.urdf.xacro` and is checked in so that `ik_node` runs without the description packages; regenerate it if you change the xacro.
- **Simulation only.** There is no robot driver, and no latency or accuracy measurement of the full loop yet.

## Open-source references

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker): Google, Apache 2.0 (the model file `hand_landmarker.task` is from here)
- [ikpy](https://github.com/Phylliade/ikpy): Pierre Manceron, MIT
- [ur_description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description): Universal Robots, BSD-3
- [robotiq_description](https://github.com/ros-industrial/robotiq): ROS-Industrial, Apache 2.0
- [OpenCV](https://opencv.org/): OpenCV team, Apache 2.0

## Licence

MIT, see `LICENSE`. The third-party components above keep their own licences.
