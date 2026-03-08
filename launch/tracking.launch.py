"""
tracking.launch.py
Launches the follow_trajectory_server node from the trajectory_tracking package.
Turtlebot4 Ignition bringup is commented out. Uncomment to launch with simulator.
"""

from launch import LaunchDescription
from launch_ros.actions import Node                              # Node action was missing


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='trajectory_tracking',
            executable='follow_trajectory_server',              # tracker_node was changed
            name='follow_trajectory_server',
            output='screen'
        )
    ])
