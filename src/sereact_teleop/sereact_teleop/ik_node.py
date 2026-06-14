#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
import ikpy.chain
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

class IKNode(Node):
    def __init__(self):
        super().__init__('ik_node')

        self.publisher     = self.create_publisher(JointState, '/joint_states', 10)
        self.collision_pub = self.create_publisher(Bool, '/collision_state', 10)
        self.create_subscription(Pose, '/left_arm/target_pose',  self.left_target_callback,  10)
        self.create_subscription(Pose, '/right_arm/target_pose', self.right_target_callback, 10)

        urdf_path = os.path.join(
            get_package_share_directory('sereact_teleop'),
            'urdf', 'dual_arm.urdf'
        )

        # load the kinematic chain for each arm separately from the same URDF
        # active_links_mask skips the fixed world/base links and the tool frame at the end
        self.left_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=['world', 'left_base_joint', 'left_base_link'],
            active_links_mask=[False, False, False, True, True, True, True, True, True, False]
        )
        self.right_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=['world', 'right_base_joint', 'right_base_link'],
            active_links_mask=[False, False, False, True, True, True, True, True, True, False]
        )

        self.joint_names_left = [
            'left_shoulder_pan_joint',
            'left_shoulder_lift_joint',
            'left_elbow_joint',
            'left_wrist_1_joint',
            'left_wrist_2_joint',
            'left_wrist_3_joint',
            'left_robotiq_85_left_knuckle_joint',
            'left_robotiq_85_right_knuckle_joint',
            'left_robotiq_85_left_inner_knuckle_joint',
            'left_robotiq_85_right_inner_knuckle_joint',
            'left_robotiq_85_left_finger_tip_joint',
            'left_robotiq_85_right_finger_tip_joint'
        ]
        self.joint_names_right = [
            'right_shoulder_pan_joint',
            'right_shoulder_lift_joint',
            'right_elbow_joint',
            'right_wrist_1_joint',
            'right_wrist_2_joint',
            'right_wrist_3_joint',
            'right_robotiq_85_left_knuckle_joint',
            'right_robotiq_85_right_knuckle_joint',
            'right_robotiq_85_left_inner_knuckle_joint',
            'right_robotiq_85_right_inner_knuckle_joint',
            'right_robotiq_85_left_finger_tip_joint',
            'right_robotiq_85_right_finger_tip_joint'
        ]

        # home angles bias the IK solution toward elbow-up configuration
        # without this the solver tends to find elbow-down solutions which look bad
        self.home_angles = np.array([0, 0, 0, 0.0, -1.0, 1.4, -1.7, 0.0, 0.0, 0.0])

        # solve startup position with many iterations to get a clean initial pose
        initial_target_left  = [0.3, -0.5, 0.5]
        initial_target_right = [0.3,  0.5, 0.5]
        self.joint_angles_left  = np.clip(self.left_chain.inverse_kinematics(
            initial_target_left,
            target_orientation=[0, 0, 1],
            orientation_mode="Z",
            initial_position=self.home_angles,
            max_iter=200), -2*np.pi, 2*np.pi)
        self.joint_angles_right = np.clip(self.right_chain.inverse_kinematics(
            initial_target_right,
            target_orientation=[0, 0, 1],
            orientation_mode="Z",
            initial_position=self.home_angles,
            max_iter=200), -2*np.pi, 2*np.pi)

        self.left_gripper  = 0.0
        self.right_gripper = 0.0

        # callbacks just store the latest pose — IK is solved in the timer to avoid
        # both arms blocking each other when running simultaneously
        self.left_pose  = None
        self.right_pose = None

        self.last_published_left  = self.joint_angles_left.copy()
        self.last_published_right = self.joint_angles_right.copy()

        # skip the speed limiter for the first 100 ticks so the arms reach
        # their starting position quickly on launch
        self.startup_ticks  = 100
        self.collision_ticks = 0

        self.create_timer(0.05, self.timer_callback)

    def left_target_callback(self, msg):
        self.left_pose = msg

    def right_target_callback(self, msg):
        self.right_pose = msg

    def timer_callback(self):
        # solve IK at 20Hz — both arms solved sequentially in one tick
        if self.left_pose is not None:
            msg = self.left_pose
            self.left_pose = None
            target = [msg.position.x, msg.position.y, msg.position.z]
            # blend current angles with home to keep elbow-up tendency
            initial = 0.85 * self.joint_angles_left + 0.15 * self.home_angles
            new_angles = self.left_chain.inverse_kinematics(
                target,
                target_orientation=[0, 0, 1],
                orientation_mode="Z",
                initial_position=initial,
                max_iter=10)
            new_angles = np.clip(new_angles, -2*np.pi, 2*np.pi)
            new_angles[8] = float(np.clip(msg.orientation.z, -np.pi, np.pi))

            # check collision before applying — reject if arms would be closer than 35cm
            ee_left  = self.left_chain.forward_kinematics(new_angles)[:3, 3]
            ee_right = self.right_chain.forward_kinematics(self.joint_angles_right)[:3, 3]
            if np.linalg.norm(ee_left - ee_right) > 0.35:
                self.joint_angles_left = new_angles
                self.left_gripper = msg.orientation.w
            else:
                self.get_logger().warn('Collision avoidance triggered! (left arm blocked)')
                self.collision_ticks = 25

        if self.right_pose is not None:
            msg = self.right_pose
            self.right_pose = None
            target = [msg.position.x, msg.position.y, msg.position.z]
            initial = 0.85 * self.joint_angles_right + 0.15 * self.home_angles
            new_angles = self.right_chain.inverse_kinematics(
                target,
                target_orientation=[0, 0, 1],
                orientation_mode="Z",
                initial_position=initial,
                max_iter=10)
            new_angles = np.clip(new_angles, -2*np.pi, 2*np.pi)
            new_angles[8] = float(np.clip(msg.orientation.z, -np.pi, np.pi))

            ee_left  = self.left_chain.forward_kinematics(self.joint_angles_left)[:3, 3]
            ee_right = self.right_chain.forward_kinematics(new_angles)[:3, 3]
            if np.linalg.norm(ee_left - ee_right) > 0.35:
                self.joint_angles_right = new_angles
                self.right_gripper = msg.orientation.w
            else:
                self.get_logger().warn('Collision avoidance triggered! (right arm blocked)')
                self.collision_ticks = 25

        # speed limiter — clamp how much each joint can move per tick to avoid jerky motion
        if self.startup_ticks > 0:
            self.startup_ticks -= 1
        else:
            max_speed = 0.12  # rad per tick (~2.4 rad/s at 20Hz)
            self.joint_angles_left[3:9] = np.clip(
                self.joint_angles_left[3:9],
                self.last_published_left[3:9]  - max_speed,
                self.last_published_left[3:9]  + max_speed
            )
            self.joint_angles_right[3:9] = np.clip(
                self.joint_angles_right[3:9],
                self.last_published_right[3:9] - max_speed,
                self.last_published_right[3:9] + max_speed
            )

        self.last_published_left  = self.joint_angles_left.copy()
        self.last_published_right = self.joint_angles_right.copy()

        if self.collision_ticks > 0:
            self.collision_ticks -= 1
        collision_msg = Bool()
        collision_msg.data = self.collision_ticks > 0
        self.collision_pub.publish(collision_msg)

        g_l = self.left_gripper
        g_r = self.right_gripper

        # gripper mimic joints — the robotiq 2F-85 has several coupled joints
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names_left + self.joint_names_right
        msg.position = (list(self.joint_angles_left[3:9]) +
                        [g_l, -g_l, g_l, -g_l, -g_l, g_l] +
                        list(self.joint_angles_right[3:9]) +
                        [g_r, -g_r, g_r, -g_r, -g_r, g_r])
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = IKNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
