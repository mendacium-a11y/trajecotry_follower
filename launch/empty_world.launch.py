from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    turtlebot4_dir = get_package_share_directory('turtlebot4_ignition_bringup')

    return LaunchDescription([
        # Empty world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(turtlebot4_dir, 'launch', 'turtlebot4_ignition.launch.py')
            ),
            launch_arguments={'world': 'empty'}.items()
        ),

        # Ground plane — fixes Transform Control gizmo ray-picking in empty worlds
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_sim', 'create',
                '-world', 'empty',
                '-name', 'ground_plane',
                '-x', '0.0', '-y', '0.0', '-z', '0.0',
                '-string', GROUND_SDF
            ],
            output='screen'
        ),

        # Spawn box 1 — directly ahead of robot start
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_sim', 'create',
                '-world', 'empty',
                '-name', 'box1',
                '-x', '3.0', '-y', '0.0', '-z', '0.25',
                '-string', BOX_SDF
            ],
            output='screen'
        ),

        # Spawn box 2 — off to the side
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_sim', 'create',
                '-world', 'empty',
                '-name', 'box2',
                '-x', '3.0', '-y', '1.5', '-z', '0.25',
                '-string', BOX_SDF
            ],
            output='screen'
        ),

        # Action server node
        Node(
            package='trajectory_tracking',
            executable='follow_trajectory_server',
            name='follow_trajectory_server',
            output='screen'
        ),
    ])


GROUND_SDF = """
<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='ground_plane'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>100 100</size>
          </plane>
        </geometry>
      </collision>
      <visual name='visual'>
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>100 100</size>
          </plane>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


BOX_SDF = """
<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='box'>
    <link name='link'>
      <gravity>false</gravity>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.0208</ixx>
          <iyy>0.0208</iyy>
          <izz>0.0208</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name='collision'>
        <geometry><box><size>0.5 0.5 0.5</size></box></geometry>
      </collision>
      <visual name='visual'>
        <geometry><box><size>0.5 0.5 0.5</size></box></geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
