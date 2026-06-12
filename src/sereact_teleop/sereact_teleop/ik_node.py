import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import ikpy.chain
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

class IKNode(Node):
    def __init__(self):
        super().__init__('ik_node')
        
        # Publisher
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        
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
                                 'left_robotiq_85_right_knuckle_joint']
        
        self.joint_names_right = [
            'right_shoulder_pan_joint',
            'right_shoulder_lift_joint',
            'right_elbow_joint',
            'right_wrist_1_joint',
            'right_wrist_2_joint',
            'right_wrist_3_joint',
            'right_robotiq_85_left_knuckle_joint',
            'right_robotiq_85_right_knuckle_joint'
        ]
        
        # Timer — solve and publish at 50Hz
        self.create_timer(0.05, self.timer_callback)
        # In __init__, after loading the chain
        initial_target_left = [0.3, -0.5, 0.5]
        initial_target_right = [0.3, 0.5, 0.5]
        self.joint_angles_left = self.left_chain.inverse_kinematics(initial_target_left)
        self.joint_angles_right = self.right_chain.inverse_kinematics(initial_target_right)
    
    def timer_callback(self):
        t = self.get_clock().now().nanoseconds / 1e9
        target_left = [0.3 + 0.05 * np.sin(t), -0.5, 0.5]
        target_right = [0.3 + 0.05 * np.sin(t), 0.5, 0.5]

        prev_left = self.joint_angles_left.copy()
        prev_right = self.joint_angles_right.copy()
        
        self.joint_angles_left = self.left_chain.inverse_kinematics(
            target_left,
            initial_position=self.joint_angles_left
        )

        self.joint_angles_right = self.right_chain.inverse_kinematics(
            target_right,
            initial_position=self.joint_angles_right
        )

        ee_left_position = self.left_chain.forward_kinematics(self.joint_angles_left)[:3, 3]
        ee_right_position = self.right_chain.forward_kinematics(self.joint_angles_right)[:3, 3]

        distance = np.linalg.norm(ee_left_position - ee_right_position)

        if distance < 0.35:
            self.get_logger().warn('Collision avoidance triggered!')
            self.joint_angles_left = prev_left
            self.joint_angles_right = prev_right
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names_left + self.joint_names_right
        msg.position = list(self.joint_angles_left[3:9]) + [0.0, 0.0] + list(self.joint_angles_right[3:9]) + [0.0, 0.0]
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = IKNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()