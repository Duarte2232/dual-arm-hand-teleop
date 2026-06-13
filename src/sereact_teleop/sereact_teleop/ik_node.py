#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import ikpy.chain
import numpy as np
import time
import os
from ament_index_python.packages import get_package_share_directory

class IKNode(Node):
    def __init__(self):
        super().__init__('ik_node')
        
        # Publisher
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(Pose, '/left_arm/target_pose', self.left_target_callback, 10)
        self.create_subscription(Pose, '/right_arm/target_pose', self.right_target_callback, 10)
        
        
        # Load ikpy chain
        urdf_path = os.path.join(
            get_package_share_directory('sereact_teleop'),
            'urdf', 'dual_arm.urdf'
        )
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
        
        self.joint_names_left = ['left_shoulder_pan_joint', 
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

        
        initial_target_left = [0.3, -0.5, 0.5]
        initial_target_right = [0.3, 0.5, 0.5]
        self.joint_angles_left = self.left_chain.inverse_kinematics(initial_target_left)
        self.joint_angles_left = np.clip(
            self.joint_angles_left, 
            -2*np.pi, 
            2*np.pi
            )
        self.joint_angles_right = self.right_chain.inverse_kinematics(initial_target_right)
        self.joint_angles_right = np.clip(
            self.joint_angles_right, 
            -2*np.pi, 
            2*np.pi
        )
        self.prev_joint_angles_left = self.joint_angles_left.copy()
        self.prev_joint_angles_right = self.joint_angles_right.copy()

        self.left_gripper = 0.0
        self.right_gripper = 0.0

        self.last_published_left = self.joint_angles_left.copy()
        self.last_published_right = self.joint_angles_right.copy()  
         # Timer — solve and publish at 50Hz
        self.create_timer(0.05, self.timer_callback)


    def left_target_callback(self, msg):
        self.prev_joint_angles_left = self.joint_angles_left.copy()
        self.joint_angles_left = np.clip(self.joint_angles_left, -2*np.pi, 2*np.pi)
        target = [msg.position.x, msg.position.y, msg.position.z]
        
      
        self.joint_angles_left = self.left_chain.inverse_kinematics(
            target, 
            target_orientation=[0, 0, 1],
            orientation_mode="Z",
            initial_position=self.joint_angles_left,
            max_iter= 10)
        wrist_angle = float(np.clip(msg.orientation.z, -np.pi, np.pi))
        self.joint_angles_left[8] = wrist_angle
        self.left_gripper = msg.orientation.w
       
    

    def right_target_callback(self, msg):
        self.prev_joint_angles_right = self.joint_angles_right.copy()
        self.joint_angles_right = np.clip(self.joint_angles_right, -2*np.pi, 2*np.pi)
        target = [msg.position.x, msg.position.y, msg.position.z]
        self.joint_angles_right = self.right_chain.inverse_kinematics(
            target, 
            target_orientation=[0, 0, 1],
            orientation_mode="Z",
            initial_position=self.joint_angles_right,
            max_iter=10)
        wrist_angle = float(np.clip(msg.orientation.z, -np.pi, np.pi))
        self.joint_angles_right[8] = wrist_angle
        self.right_gripper = msg.orientation.w  # right_target_callback
    
    def timer_callback(self):
        max_speed = 0.1  # radians per callback

        self.joint_angles_left[3:9] = np.clip(
            self.joint_angles_left[3:9],
            self.last_published_left[3:9] - max_speed,
            self.last_published_left[3:9] + max_speed
        )
        self.joint_angles_right[3:9] = np.clip(
            self.joint_angles_right[3:9],
            self.last_published_right[3:9] - max_speed,
            self.last_published_right[3:9] + max_speed
        )

        # Update last published
        self.last_published_left = self.joint_angles_left.copy()
        self.last_published_right = self.joint_angles_right.copy()

    

        ee_left_position = self.left_chain.forward_kinematics(self.joint_angles_left)[:3, 3]
        ee_right_position = self.right_chain.forward_kinematics(self.joint_angles_right)[:3, 3]

        distance = np.linalg.norm(ee_left_position - ee_right_position)

        g_l = self.left_gripper
        g_r = self.right_gripper

        if distance < 0.35:
            self.get_logger().warn('Collision avoidance triggered!')
            self.joint_angles_left = self.prev_joint_angles_left
            self.joint_angles_right = self.prev_joint_angles_right 
        
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