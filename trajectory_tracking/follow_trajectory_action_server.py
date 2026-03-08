"""
follow_trajectory_action_server.py
ROS 2 action server for trajectory-following with geometric obstacle avoidance.

Architecture:
- Accepts a list of 2D waypoints via the FollowTrajectory action.
- Smooths waypoints with a B-spline, generates a trapezoidal-profile trajectory.
- Runs Pure Pursuit at control_rate Hz to produce /cmd_vel commands.
- PathReplanner monitors /scan every cycle; on obstacle detection it generates
  a geometric bypass and injects a new trajectory seamlessly.
- All visualization topics are published for live inspection in RViz.
"""

import rclpy
import numpy as np
import time
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, ColorRGBA
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from turtlebot4_msgs.action import FollowTrajectory
from trajectory_tracking.path_smoother import smooth_path
from trajectory_tracking.trajectory_generator import generate_trajectory
from trajectory_tracking.pure_pursuit import PurePursuit
from trajectory_tracking.path_replanner import PathReplanner


class FollowTrajectoryActionServer(Node):

    def __init__(self):
        super().__init__('follow_trajectory_server')

        self.declare_parameter('total_time',     60.0)
        self.declare_parameter('max_vel',         0.3)
        self.declare_parameter('lookahead',       0.5)
        self.declare_parameter('control_rate',   20.0)
        self.declare_parameter('goal_tolerance',  0.25)
        self.declare_parameter('viz_frame',     'odom')

        self.cb_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            FollowTrajectory,
            'follow_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group,
        )

        self.cmd_vel_pub   = self.create_publisher(Twist,       '/cmd_vel',              10)
        self.traj_pub      = self.create_publisher(Path,        '/trajectory_path',      10)
        self.replan_pub    = self.create_publisher(Path,        '/replanned_trajectory', 10)
        self.waypoints_pub = self.create_publisher(MarkerArray, '/input_waypoints',      10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10,
            callback_group=self.cb_group,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10,
            callback_group=self.cb_group,
        )

        self.replanner = PathReplanner(
            robot_radius        = 0.22,
            corridor_half_width = 0.50,
            lookahead_check     = 3.0,
            stop_distance       = 0.35,
            stop_arc_deg        = 50.0,
            replan_cooldown     = 2.0,
            bypass_timeout      = 10.0,
            lateral_clearance   = 0.18,
        )

        self.robot_pose = None
        self.get_logger().info('Follow Trajectory Action Server ready')

    # ── Visualization helpers ──────────────────────────────────────────────

    def _make_path_msg(self, trajectory) -> Path:
        frame = self.get_parameter('viz_frame').value
        msg   = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        for (x, y, _t) in trajectory:
            ps = PoseStamped()
            ps.header.frame_id    = frame
            ps.header.stamp       = msg.header.stamp
            ps.pose.position.x    = x
            ps.pose.position.y    = y
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def _make_waypoint_markers(self, waypoints) -> MarkerArray:
        frame = self.get_parameter('viz_frame').value
        now   = self.get_clock().now().to_msg()
        array = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = frame
        delete_all.header.stamp    = now
        delete_all.ns              = 'input_waypoints'
        delete_all.action          = Marker.DELETEALL
        array.markers.append(delete_all)

        for i, (x, y) in enumerate(waypoints):
            m = Marker()
            m.header.frame_id    = frame
            m.header.stamp       = now
            m.ns                 = 'input_waypoints'
            m.id                 = i
            m.type               = Marker.SPHERE
            m.action             = Marker.ADD
            m.pose.position.x    = x
            m.pose.position.y    = y
            m.pose.position.z    = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            if i == 0:
                m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            elif i == len(waypoints) - 1:
                m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            else:
                m.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0)
            m.lifetime = Duration(sec=0)
            array.markers.append(m)

        return array

    # ── Callbacks ──────────────────────────────────────────────────────────

    def goal_callback(self, goal_request):
        if len(goal_request.waypoints) < 2:
            self.get_logger().warn('Goal rejected: need at least 2 waypoints')
            return GoalResponse.REJECT
        self.get_logger().info(
            f'Goal accepted: {len(goal_request.waypoints)} waypoints'
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel requested')
        return CancelResponse.ACCEPT

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.robot_pose = (pos.x, pos.y, 2.0 * np.arctan2(ori.z, ori.w))

    def scan_callback(self, msg: LaserScan):
        # Guard: scan can arrive before first odometry, pose would be None
        if self.robot_pose is not None:
            self.replanner.update_scan(msg, self.robot_pose)

    # ── Execution loop ─────────────────────────────────────────────────────

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        total_time = self.get_parameter('total_time').value
        max_vel    = self.get_parameter('max_vel').value
        lookahead  = self.get_parameter('lookahead').value
        rate_hz    = self.get_parameter('control_rate').value
        tolerance  = self.get_parameter('goal_tolerance').value

        feedback_msg   = FollowTrajectory.Feedback()
        control_period = 1.0 / rate_hz

        # Wait up to 5s for first odometry
        deadline = time.time() + 5.0
        while self.robot_pose is None and time.time() < deadline:
            time.sleep(0.05)
        if self.robot_pose is None:
            self.get_logger().error('No odometry received, aborting')
            goal_handle.abort()
            return FollowTrajectory.Result()

        self.get_logger().info(
            f'Start: x={self.robot_pose[0]:.2f} '
            f'y={self.robot_pose[1]:.2f} '
            f'θ={np.degrees(self.robot_pose[2]):.1f}°'
        )

        waypoints           = [(wp.x, wp.y) for wp in goal_handle.request.waypoints]
        smoothed            = smooth_path(waypoints, num_points=200)
        original_trajectory = generate_trajectory(smoothed, total_time, max_vel)
        trajectory          = original_trajectory
        final_goal          = original_trajectory[-1][:2]

        self.waypoints_pub.publish(self._make_waypoint_markers(waypoints))
        self.traj_pub.publish(self._make_path_msg(original_trajectory))
        self.get_logger().info('Published trajectory and waypoints for RViz')

        pp = PurePursuit(lookahead=lookahead, max_lin_vel=max_vel)
        pp.set_trajectory(trajectory)

        start_time    = time.time()
        result        = FollowTrajectory.Result()
        _estop_logged = False

        while rclpy.ok():

            # ── Cancel check ───────────────────────────────────────────────
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Cancelled')
                self.cmd_vel_pub.publish(Twist())
                goal_handle.canceled()
                result.success        = False
                result.final_error    = 0.0
                result.execution_time = float(time.time() - start_time)
                return result

            elapsed        = time.time() - start_time
            remaining_time = max(total_time - elapsed, 5.0)

            # Capture snapshot once, e-stop and replan use identical data
            snap = self.replanner._snapshot

            # ── Emergency stop - does NOT skip replan ──────────────────────
            if self.replanner.emergency_stop_needed(snap):
                self.cmd_vel_pub.publish(Twist())
                if not _estop_logged:
                    self.get_logger().warn('Emergency stop triggered, obstacle too close')
                    _estop_logged = True
            else:
                _estop_logged = False

            # ── Replan - runs every cycle, even during e-stop ──────────────
            trajectory, pp_start_idx, did_replan = self.replanner.replan(
                trajectory,
                original_trajectory,
                pp._current_idx,
                self.robot_pose[0],
                self.robot_pose[1],
                self.robot_pose[2],
                max_vel,
                remaining_time,
                snap=snap,
            )

            if did_replan:
                pp.set_trajectory(trajectory)
                pp._current_idx = pp_start_idx
                self.get_logger().info(
                    f'Pure Pursuit restarted at idx={pp_start_idx}/{len(trajectory)}'
                )
                self.replan_pub.publish(self._make_path_msg(trajectory))

            # ── Velocity — suppressed during e-stop ───────────────────────
            if not self.replanner.emergency_stop_needed(snap):
                self.cmd_vel_pub.publish(pp.compute_cmd(*self.robot_pose))

            # ── Feedback ───────────────────────────────────────────────────
            dist_to_goal = float(np.linalg.norm(
                np.array(final_goal) - np.array(self.robot_pose[:2])
            ))
            feedback_msg.progress_pct = float(
                np.clip((elapsed / total_time) * 100.0, 0.0, 100.0)
            )
            feedback_msg.status = String(data=(
                f'dist={dist_to_goal:.2f}m | '
                f'idx={pp._current_idx}/{len(trajectory)} | '
                f't={elapsed:.1f}s | '
                f'state={self.replanner._state}'
            ))
            goal_handle.publish_feedback(feedback_msg)

            # ── Termination checks ─────────────────────────────────────────
            if dist_to_goal < tolerance:
                self.get_logger().info(f'Goal reached! error={dist_to_goal:.3f}m')
                break

            if pp._current_idx >= len(trajectory) - 3 and dist_to_goal < tolerance * 3:
                self.get_logger().info(
                    f'Close enough at path end: error={dist_to_goal:.3f}m'
                )
                break

            if elapsed > total_time:
                self.get_logger().warn(f'Execution timeout | dist_to_goal={dist_to_goal:.2f}m')
                break

            time.sleep(control_period)

        # ── Shutdown ───────────────────────────────────────────────────────
        self.cmd_vel_pub.publish(Twist())
        elapsed      = time.time() - start_time
        dist_to_goal = float(np.linalg.norm(
            np.array(final_goal) - np.array(self.robot_pose[:2])
        ))
        result.success        = dist_to_goal < tolerance
        result.final_error    = dist_to_goal
        result.execution_time = float(elapsed)
        goal_handle.succeed()
        self.get_logger().info(
            f'Result: success={result.success} | '
            f'error={result.final_error:.3f}m | '
            f'time={elapsed:.1f}s'
        )
        return result


def main(args=None):
    rclpy.init(args=args)
    node     = FollowTrajectoryActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
