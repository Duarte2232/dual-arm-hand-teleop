import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg = get_package_share_directory('sereact_teleop')
    xacro_file = os.path.join(pkg, 'urdf', 'dual_arm.urdf.xacro')
    robot_description = subprocess.check_output(['xacro', xacro_file]).decode()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(pkg, 'config', 'display.rviz')]
        ),
        Node(
            package='sereact_teleop',
            executable='ik_node.py',
            name='ik_node',
            output='screen'
        ),
        Node(
            package='sereact_teleop',
            executable='teleop_node.py',
            name='teleop_node',
            output='screen'
        ),
        Node(
            package='sereact_teleop',
            executable='mediapipe_node.py',
            name='mediapipe_node',
            output='screen'
        ),
    ])