from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node                              # FIX: was missing
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    turtlebot4_dir = get_package_share_directory('turtlebot4_ignition_bringup')
    
    return LaunchDescription([
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         os.path.join(turtlebot4_dir, 'launch', 'turtlebot4_ignition.launch.py')
        #     ),
        #     launch_arguments={'world': 'empty'}.items()
        # ),
        Node(
            package='trajectory_tracking',
            executable='follow_trajectory_server',              # FIX: was 'tracker_node'
            name='follow_trajectory_server',
            output='screen'
        )
    ])
