import rclpy
import numpy as np
import time
import math
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from turtlebot4_msgs.action import FollowTrajectory
from trajectory_tracking.path_smoother import smooth_path
from trajectory_tracking.trajectory_generator import generate_trajectory
from trajectory_tracking.pure_pursuit import PurePursuit
from trajectory_tracking.obstacle_avoider import ObstacleAvoider
from sensor_msgs.msg import LaserScan


STUCK_THRESHOLD  = 5    # STOP cycles before forcing detour
DETOUR_DISTANCE  = 1.5  # metres sideways to step
DETOUR_FORWARD   = 1.0  # metres forward after stepping


class FollowTrajectoryActionServer(Node):
    def __init__(self):
        super().__init__('follow_trajectory_server')

        self.declare_parameter('total_time', 60.0)   # increased from 30.0
        self.declare_parameter('max_vel', 0.3)
        self.declare_parameter('lookahead', 0.5)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('goal_tolerance', 0.1)

        self.cb_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            FollowTrajectory,
            'follow_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10,
            callback_group=self.cb_group
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10,
            callback_group=self.cb_group
        )

        self.avoider = ObstacleAvoider(
            stop_distance=0.4,
            slow_distance=0.8,
            forward_arc_deg=60.0,
            side_arc_deg=120.0,
            recovery_duration=1.5,
        )

        self.robot_pose = None
        self.get_logger().info('✅ Follow Trajectory Action Server ready')

    # ─── Callbacks ────────────────────────────────────────────────

    def goal_callback(self, goal_request):
        if len(goal_request.waypoints) < 2:
            self.get_logger().warn('❌ Goal rejected: need at least 2 waypoints')
            return GoalResponse.REJECT
        self.get_logger().info(f'✅ Goal accepted: {len(goal_request.waypoints)} waypoints')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel requested')
        return CancelResponse.ACCEPT

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        theta = 2.0 * np.arctan2(ori.z, ori.w)
        self.robot_pose = (pos.x, pos.y, theta)

    def scan_callback(self, msg: LaserScan):
        self.avoider.update_scan(msg)

    # ─── Helpers ──────────────────────────────────────────────────

    def _compute_detour_point(self, escape_left: bool) -> tuple:
        """1.5m sideways + 1m forward from current pose in escape direction."""
        x, y, theta = self.robot_pose
        side_angle = theta + (math.pi / 2 if escape_left else -math.pi / 2)
        side_x = x + DETOUR_DISTANCE * math.cos(side_angle)
        side_y = y + DETOUR_DISTANCE * math.sin(side_angle)
        fwd_x  = side_x + DETOUR_FORWARD * math.cos(theta)
        fwd_y  = side_y + DETOUR_FORWARD * math.sin(theta)
        return (fwd_x, fwd_y)

    def _make_detour_trajectory(self, max_vel: float) -> list:
        escape_left = self.avoider._escape_direction > 0
        detour      = self._compute_detour_point(escape_left)
        self.get_logger().info(
            f'🔀 Detour → ({detour[0]:.2f}, {detour[1]:.2f}) | '
            f'escape={"LEFT" if escape_left else "RIGHT"}'
        )
        return generate_trajectory(
            smooth_path([self.robot_pose[:2], detour], num_points=40),
            total_time=5.0,
            max_vel=max_vel
        )

    def _publish_feedback(self, goal_handle, feedback_msg, progress, status_text):
        feedback_msg.progress_pct = progress
        feedback_msg.status = String(data=status_text)
        goal_handle.publish_feedback(feedback_msg)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info('Robot stopped')

    # ─── Main Execution ───────────────────────────────────────────

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        total_time = self.get_parameter('total_time').value
        max_vel    = self.get_parameter('max_vel').value
        lookahead  = self.get_parameter('lookahead').value
        rate_hz    = self.get_parameter('control_rate').value
        tolerance  = self.get_parameter('goal_tolerance').value

        feedback_msg = FollowTrajectory.Feedback()

        # Wait for valid odom
        self._publish_feedback(goal_handle, feedback_msg, 0.0, 'Waiting for odom...')
        timeout = time.time() + 5.0
        while self.robot_pose is None and time.time() < timeout:
            time.sleep(0.05)

        if self.robot_pose is None:
            self.get_logger().error('❌ No odom received!')
            goal_handle.abort()
            return FollowTrajectory.Result()

        self.get_logger().info(
            f'Robot pose: x={self.robot_pose[0]:.2f}, '
            f'y={self.robot_pose[1]:.2f}, '
            f'θ={np.degrees(self.robot_pose[2]):.1f}°'
        )

        # Build main trajectory
        self._publish_feedback(goal_handle, feedback_msg, 0.0, 'Smoothing path...')
        waypoints  = [(wp.x, wp.y) for wp in goal_handle.request.waypoints]
        smoothed   = smooth_path(waypoints, num_points=200)

        self._publish_feedback(goal_handle, feedback_msg, 0.0, 'Generating trajectory...')
        main_trajectory = generate_trajectory(smoothed, total_time, max_vel)
        final_goal      = main_trajectory[-1][:2]

        pp = PurePursuit(lookahead=lookahead, max_lin_vel=max_vel)
        pp.set_trajectory(main_trajectory)
        current_trajectory = main_trajectory
        on_detour          = False

        self._publish_feedback(goal_handle, feedback_msg, 0.0, 'Tracking started...')

        start_time         = time.time()
        control_period     = 1.0 / rate_hz
        stuck_cycles       = 0
        prev_avoider_state = 'CLEAR'
        result             = FollowTrajectory.Result()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal cancelled')
                self.stop_robot()
                goal_handle.canceled()
                result.success        = False
                result.final_error    = 0.0
                result.execution_time = float(time.time() - start_time)
                return result

            twist      = pp.compute_cmd(*self.robot_pose)
            safe_twist = self.avoider.modify_cmd(twist)
            self.cmd_vel_pub.publish(safe_twist)

            cur_state = self.avoider._state

            # ── Detour logic ───────────────────────────────────────
            if cur_state == 'STOP':
                stuck_cycles += 1
                if stuck_cycles >= STUCK_THRESHOLD:
                    # Force detour immediately
                    detour_traj = self._make_detour_trajectory(max_vel)
                    pp.set_trajectory(detour_traj)
                    pp._current_idx = 0
                    on_detour    = True
                    stuck_cycles = 0

            elif prev_avoider_state == 'RECOVERY' and cur_state == 'CLEAR':
                # Just finished recovery — inject detour so Pure Pursuit
                # doesn't aim back at the same obstacle
                detour_traj = self._make_detour_trajectory(max_vel)
                pp.set_trajectory(detour_traj)
                pp._current_idx = 0
                on_detour    = True
                stuck_cycles = 0

            else:
                stuck_cycles = 0

            # ── Return to main trajectory after detour completes ───
            if on_detour:
                dist_to_detour_end = float(np.linalg.norm(
                    np.array(pp.trajectory[-1][:2]) - np.array(self.robot_pose[:2])
                ))
                if dist_to_detour_end < 0.3 or pp._current_idx >= len(pp.trajectory) - 1:
                    self.get_logger().info('✅ Detour complete — resuming main trajectory')
                    # Find closest point on main trajectory ahead of where we are
                    best_idx = 0
                    best_dist = float('inf')
                    x, y, _ = self.robot_pose
                    for i, (tx, ty, _) in enumerate(main_trajectory):
                        d = math.sqrt((x - tx)**2 + (y - ty)**2)
                        if d < best_dist:
                            best_dist = d
                            best_idx  = i
                    pp.set_trajectory(main_trajectory)
                    pp._current_idx    = min(best_idx + 10, len(main_trajectory) - 1)
                    current_trajectory = main_trajectory
                    on_detour          = False

            prev_avoider_state = cur_state

            # ── Feedback & termination ─────────────────────────────
            elapsed      = time.time() - start_time
            progress     = float(min(100.0, (elapsed / total_time) * 100.0))
            dist_to_goal = float(np.linalg.norm(
                np.array(final_goal) - np.array(self.robot_pose[:2])
            ))

            self._publish_feedback(
                goal_handle, feedback_msg,
                progress,
                f'Tracking... {progress:.1f}% | '
                f'dist_to_goal={dist_to_goal:.2f}m | '
                f'state={cur_state} | '
                f'{"DETOUR" if on_detour else "MAIN"}'
            )

            path_progress = pp._current_idx / len(main_trajectory)
            if dist_to_goal < tolerance and path_progress > 0.8:
                self.get_logger().info('✅ Goal reached!')
                break

            if elapsed > total_time * 1.5:
                self.get_logger().warn('⚠️ Timeout reached')
                break

            time.sleep(control_period)

        self.stop_robot()
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
            f'time={result.execution_time:.1f}s'
        )
        return result


def main(args=None):
    rclpy.init(args=args)
    server = FollowTrajectoryActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(server)
    try:
        executor.spin()
    finally:
        server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
