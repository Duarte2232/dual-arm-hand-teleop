#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
import numpy as np

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')
        
        # Subscribers
        self.subscription = self.create_subscription(
            Float32MultiArray, '/hand_state', self.hand_callback, 10)
        
        # Publishers
        self.left_pub = self.create_publisher(Pose, '/left_arm/target_pose', 10)
        self.right_pub = self.create_publisher(Pose, '/right_arm/target_pose', 10)
        
        # Workspace bounds
        self.y_min_left, self.y_max_left = -0.5, 0.0
        self.y_min_right, self.y_max_right = 0.0, 0.5
        self.z_min, self.z_max = 0.1, 0.7
        self.left_x  = 0.4
        self.right_x = 0.4

        

        

        
        # Deadzone
        self.deadzone = 0.01
        self.angle_deadzone = 0.02
        
        # Previous hand positions for deadzone check
        self.prev_left  = [0.5, 0.5]
        self.prev_right = [0.5, 0.5]
        self.prev_left_angle = 0.0
        self.prev_right_angle = 0.0
        
        # Current targets
        self.left_target  = [self.left_x, -0.5, 0.4]
        self.right_target = [self.right_x,  0.5, 0.4]

        
        self.left_wrist_neutral = None
        self.right_wrist_neutral = None

        self.prev_gripper_left  = 0.0
        self.prev_gripper_right = 0.0

        self.left_hand_present  = False
        self.right_hand_present = False

    def hand_callback(self, msg):
        data = msg.data
        
        # Left hand
        lx, ly, l_pinch, l_angle = data[0], data[1], data[2], data[3]
        # Right hand
        rx, ry, r_pinch, r_angle = data[4], data[5], data[6], data[7]
        # Dead-man switch: middle finger extended
        left_active  = data[8] > 0.5
        right_active = data[9] > 0.5

        pinch_min, pinch_max = 0.056, 0.36

        left_detected  = lx != 0.0
        right_detected = rx != 0.0

        # Recalibrate wrist neutral each time a hand re-enters the frame
        if left_detected and not self.left_hand_present:
            self.left_wrist_neutral = l_angle
        if right_detected and not self.right_hand_present:
            self.right_wrist_neutral = r_angle

        self.left_hand_present  = left_detected
        self.right_hand_present = right_detected

        if left_active:
            gripper_left = 0.8 * (1.0 - ((l_pinch - pinch_min) / (pinch_max - pinch_min)))
            gripper_left = float(np.clip(gripper_left, 0.0, 0.8))

            if abs(lx - self.prev_left[0]) > self.deadzone or \
               abs(ly - self.prev_left[1]) > self.deadzone:
                self.left_target[1] = self.y_min_left + lx * (self.y_max_left - self.y_min_left)
                self.left_target[2] = self.z_max - ly * (self.z_max - self.z_min)
                self.prev_left = [lx, ly]

            if abs(l_angle - self.prev_left_angle) < self.angle_deadzone:
                l_angle = self.prev_left_angle
            else:
                self.prev_left_angle = l_angle
        else:
            gripper_left = self.prev_gripper_left
            l_angle = self.prev_left_angle

        if right_active:
            gripper_right = 0.8 * (1.0 - (r_pinch - pinch_min) / (pinch_max - pinch_min))
            gripper_right = float(np.clip(gripper_right, 0.0, 0.8))

            if abs(rx - self.prev_right[0]) > self.deadzone or \
               abs(ry - self.prev_right[1]) > self.deadzone:
                self.right_target[1] = self.y_min_right + rx * (self.y_max_right - self.y_min_right)
                self.right_target[2] = self.z_max - ry * (self.z_max - self.z_min)
                self.prev_right = [rx, ry]

            if abs(r_angle - self.prev_right_angle) < self.angle_deadzone:
                r_angle = self.prev_right_angle
            else:
                self.prev_right_angle = r_angle
        else:
            gripper_right = self.prev_gripper_right
            r_angle = self.prev_right_angle

        self.prev_gripper_left  = gripper_left
        self.prev_gripper_right = gripper_right

        # Only publish when active — prevents IK from re-solving with home bias and drifting
        if left_active:
            left_msg = Pose()
            left_msg.position.x = self.left_target[0]
            left_msg.position.y = self.left_target[1]
            left_msg.position.z = self.left_target[2]
            if self.left_wrist_neutral is not None:
                left_msg.orientation.z = float(np.clip(
                    (l_angle - self.left_wrist_neutral) * 2.0, -np.pi, np.pi))
            left_msg.orientation.w = gripper_left
            self.left_pub.publish(left_msg)

        if right_active:
            right_msg = Pose()
            right_msg.position.x = self.right_target[0]
            right_msg.position.y = self.right_target[1]
            right_msg.position.z = self.right_target[2]
            if self.right_wrist_neutral is not None:
                right_msg.orientation.z = float(np.clip(
                    (r_angle - self.right_wrist_neutral) * 2.0, -np.pi, np.pi))
            right_msg.orientation.w = gripper_right
            self.right_pub.publish(right_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()