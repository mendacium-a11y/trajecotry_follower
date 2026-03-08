import numpy as np
import time
from sensor_msgs.msg import LaserScan
import rclpy.logging
from trajectory_tracking.path_smoother import smooth_path
from trajectory_tracking.trajectory_generator import generate_trajectory

logger = rclpy.logging.get_logger('path_replanner')


class PathReplanner:

    def __init__(
        self,
        robot_width: float        = 0.40,
        corridor_width: float     = 0.80,
        lookahead_check: float    = 3.0,
        obstacle_margin: float    = 0.65,
        replan_cooldown: float    = 1.5,
        stop_distance: float      = 0.45,
        stop_arc_deg: float       = 60.0,
        min_move_to_replan: float = 0.05,
    ):
        self.robot_width        = robot_width
        self.corridor_width     = corridor_width
        self.lookahead_check    = lookahead_check
        self.obstacle_margin    = obstacle_margin
        self.replan_cooldown    = replan_cooldown
        self.stop_distance      = stop_distance
        self.stop_arc_rad       = np.radians(stop_arc_deg / 2.0)
        self.min_move_to_replan = min_move_to_replan

        self.latest_scan: LaserScan = None
        self._last_scan_pose        = None
        self._last_replan_time      = -999.0
        self._last_robot_pos        = None
        self._active_bypass_wp2     = None

    # ── Public ─────────────────────────────────────────────────────

    def update_scan(self, scan: LaserScan, robot_pose=None):
        self.latest_scan = scan
        if robot_pose is not None:
            self._last_scan_pose = robot_pose

    def emergency_stop_needed(self) -> bool:
        if self._active_bypass_wp2 is not None:
            return False
        if self.latest_scan is None:
            return False
        scan = self.latest_scan
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) < self.stop_arc_rad and r < self.stop_distance:
                return True
        return False

    def replan(
        self,
        current_trajectory: list,
        original_trajectory: list,
        current_idx: int,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        total_time: float,
        max_vel: float,
        remaining_time: float,
    ):
        now        = time.time()
        time_since = now - self._last_replan_time

        # Guard: don't replan while executing previous bypass
        if self._active_bypass_wp2 is not None:
            bx, by      = self._active_bypass_wp2
            dist_to_wp2 = np.sqrt((robot_x - bx)**2 + (robot_y - by)**2)
            if dist_to_wp2 < 0.25:
                logger.info('✅ Bypass wp2 reached — resuming normal detection')
                self._active_bypass_wp2 = None
            elif time_since < self.replan_cooldown * 3:
                return current_trajectory, current_idx, False
            else:
                logger.warn('⚠️  Bypass timeout — forcing replan')
                self._active_bypass_wp2 = None

        blocked, obs_x, obs_y = self._check_obstacle_in_path(
            current_trajectory, current_idx, robot_x, robot_y, robot_theta
        )
        if not blocked:
            return current_trajectory, current_idx, False

        if time_since < self.replan_cooldown:
            return current_trajectory, current_idx, False

        if self._last_robot_pos is not None:
            dx    = robot_x - self._last_robot_pos[0]
            dy    = robot_y - self._last_robot_pos[1]
            moved = np.sqrt(dx**2 + dy**2)
            if moved < self.min_move_to_replan and time_since < self.replan_cooldown * 2:
                return current_trajectory, current_idx, False

        logger.warn(f'⚠️  Obstacle at ({obs_x:.2f}, {obs_y:.2f}) — replanning')

        closest_orig_idx = int(np.argmin([
            (t[0] - obs_x)**2 + (t[1] - obs_y)**2
            for t in original_trajectory
        ]))
        orig_path_dir = self._path_direction(original_trajectory, closest_orig_idx)

        # Shift scan-detected near face to approximate obstacle centre
        obs_cx = obs_x + orig_path_dir[0] * 0.30
        obs_cy = obs_y + orig_path_dir[1] * 0.30

        side   = self._choose_side(robot_x, robot_y, robot_theta)
        bypass = self._bypass_waypoints(
            original_trajectory, closest_orig_idx,
            obs_cx, obs_cy, side,
            robot_x, robot_y
        )
        if bypass is None:
            logger.warn('⚠️  No clear bypass found — skipping replan')
            return current_trajectory, current_idx, False

        resume_idx = self._find_resume_idx(
            original_trajectory, closest_orig_idx,
            bypass[-1], orig_path_dir,
            obs_cx, obs_cy
        )

        path_after = [(t[0], t[1]) for t in original_trajectory[resume_idx:]]
        if len(path_after) < 2:
            logger.warn('⚠️  Resume path too short — skipping replan')
            return current_trajectory, current_idx, False

        new_path = [(robot_x, robot_y)] + bypass + path_after

        try:
            smoothed = smooth_path(new_path, num_points=200)
            new_traj = generate_trajectory(
                smoothed, max(remaining_time, 5.0), max_vel
            )
        except Exception as e:
            logger.error(f'❌ Replan failed: {e}')
            return current_trajectory, current_idx, False

        self._last_replan_time  = now
        self._last_robot_pos    = (robot_x, robot_y)
        self._active_bypass_wp2 = bypass[1]

        pp_start_idx = min(1, len(new_traj) - 1)

        logger.info(
            f'✅ Replanned | side={side} | '
            f'bypass=[({bypass[0][0]:.2f},{bypass[0][1]:.2f}), '
            f'({bypass[1][0]:.2f},{bypass[1][1]:.2f})] | '
            f'orig_resume={resume_idx}/{len(original_trajectory)}'
        )
        return new_traj, pp_start_idx, True

    # ── Internal ───────────────────────────────────────────────────

    def _scan_to_world(self, robot_x, robot_y, robot_theta) -> list:
        if self.latest_scan is None:
            return []
        scan = self.latest_scan
        pts  = []
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            wx = robot_x + r * np.cos(robot_theta + angle)
            wy = robot_y + r * np.sin(robot_theta + angle)
            pts.append((wx, wy, r, angle))
        return pts

    def _pt_to_seg_dist(self, px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        seg_sq = dx*dx + dy*dy
        if seg_sq < 1e-9:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        t = np.clip(((px-x1)*dx + (py-y1)*dy) / seg_sq, 0.0, 1.0)
        return np.sqrt((px - (x1+t*dx))**2 + (py - (y1+t*dy))**2)

    def _check_obstacle_in_path(self, trajectory, current_idx,
                                  robot_x, robot_y, robot_theta):
        world_pts = self._scan_to_world(robot_x, robot_y, robot_theta)
        if not world_pts:
            return False, None, None

        segments = []
        for i in range(current_idx, len(trajectory) - 1):
            tx, ty, _ = trajectory[i]
            if np.sqrt((tx - robot_x)**2 + (ty - robot_y)**2) > self.lookahead_check:
                break
            segments.append((tx, ty, trajectory[i+1][0], trajectory[i+1][1]))

        if not segments:
            return False, None, None

        blocking = []
        for wx, wy, r, angle in world_pts:
            if abs(angle) > np.pi / 2.0:
                continue
            for x1, y1, x2, y2 in segments:
                if self._pt_to_seg_dist(wx, wy, x1, y1, x2, y2) < self.corridor_width:
                    blocking.append((wx, wy))
                    break

        if not blocking:
            return False, None, None

        return (
            True,
            float(np.mean([p[0] for p in blocking])),
            float(np.mean([p[1] for p in blocking])),
        )

    def _path_direction(self, trajectory, current_idx) -> np.ndarray:
        end = min(current_idx + 8, len(trajectory) - 1)
        tx1, ty1, _ = trajectory[current_idx]
        tx2, ty2, _ = trajectory[end]
        dx, dy = tx2 - tx1, ty2 - ty1
        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1e-6:
            return np.array([1.0, 0.0])
        return np.array([dx / norm, dy / norm])

    def _choose_side(self, robot_x, robot_y, robot_theta) -> str:
        if self.latest_scan is None:
            return 'left'
        scan = self.latest_scan
        left_sum, right_sum, left_n, right_n = 0.0, 0.0, 0, 0
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) > np.pi / 2.0:
                continue
            if angle > 0:
                left_sum  += r; left_n  += 1
            else:
                right_sum += r; right_n += 1
        left_mean  = left_sum  / left_n  if left_n  > 0 else 0.0
        right_mean = right_sum / right_n if right_n > 0 else 0.0
        return 'left' if left_mean > right_mean else 'right'

    def _cast_ray(self, ox, oy, dx, dy, max_dist=2.0) -> float:
        """Distance to nearest obstacle along direction (dx,dy) from (ox,oy)."""
        if self.latest_scan is None or self._last_scan_pose is None:
            return max_dist
        rx, ry, rth = self._last_scan_pose
        scan        = self.latest_scan
        min_dist    = max_dist
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            wx = rx + r * np.cos(rth + angle)
            wy = ry + r * np.sin(rth + angle)
            along = (wx - ox)*dx + (wy - oy)*dy
            if along < 0.05 or along > max_dist:
                continue
            perp = abs((wx - ox)*dy - (wy - oy)*dx)
            if perp < 0.15:
                min_dist = min(min_dist, along)
        return min_dist

    def _point_is_free(self, point, check_radius: float = 0.25) -> bool:
        if self.latest_scan is None or self._last_scan_pose is None:
            return True
        px, py      = point
        rx, ry, rth = self._last_scan_pose
        scan        = self.latest_scan
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            wx = rx + r * np.cos(rth + angle)
            wy = ry + r * np.sin(rth + angle)
            if np.sqrt((wx - px)**2 + (wy - py)**2) < check_radius:
                return False
        return True

    def _path_segment_is_free(self, p1, p2) -> bool:
        """Check the segment p1→p2 is clear of obstacles with robot half-width clearance."""
        if self.latest_scan is None or self._last_scan_pose is None:
            return True
        clearance   = self.robot_width / 2.0 + 0.15
        rx, ry, rth = self._last_scan_pose
        scan        = self.latest_scan
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            wx = rx + r * np.cos(rth + angle)
            wy = ry + r * np.sin(rth + angle)
            if self._pt_to_seg_dist(wx, wy, p1[0], p1[1], p2[0], p2[1]) < clearance:
                return False
        return True

    def _bypass_waypoints(self, trajectory, current_idx,
                           obs_x, obs_y, side,
                           robot_x, robot_y) -> list:
        """
        Returns [wp1, wp2] whose arc wp1→wp2 is clear, or None if both sides blocked.
        obs_x/obs_y should already be the shifted obstacle centre.
        """
        path_dir = self._path_direction(trajectory, current_idx)

        for attempt_side in [side, ('right' if side == 'left' else 'left')]:
            perp = np.array([-path_dir[1], path_dir[0]])
            if attempt_side == 'right':
                perp = -perp

            free_dist = self._cast_ray(obs_x, obs_y, perp[0], perp[1], max_dist=2.0)
            m = float(np.clip(free_dist * 0.55, 0.45, 0.75))

            # wp1: 0.8m before obstacle centre, wp2: 1.2m past it
            wp1 = (obs_x + perp[0]*m - path_dir[0]*m*0.8,
                   obs_y + perp[1]*m - path_dir[1]*m*0.8)
            wp2 = (obs_x + perp[0]*m + path_dir[0]*m*1.2,
                   obs_y + perp[1]*m + path_dir[1]*m*1.2)

            # Only check the bypass arc wp1→wp2 for clearance
            if self._path_segment_is_free(wp1, wp2):
                if attempt_side != side:
                    logger.info(
                        f'↩️  Side flipped to {attempt_side} | '
                        f'margin={m:.2f}m free_dist={free_dist:.2f}m'
                    )
                else:
                    logger.info(
                        f'Bypass margin={m:.2f}m (free_dist={free_dist:.2f}m)'
                    )
                return [wp1, wp2]

        logger.warn('⚠️  Both bypass sides are obstructed')
        return None

    def _find_resume_idx(self, trajectory, current_idx,
                          bypass_wp2, path_dir,
                          obs_x=None, obs_y=None) -> int:
        bx, by   = bypass_wp2
        best_idx = min(current_idx + 1, len(trajectory) - 1)

        for i in range(current_idx, len(trajectory)):
            tx, ty, _ = trajectory[i]
            along = (tx - bx)*path_dir[0] + (ty - by)*path_dir[1]
            if along > 0.3:
                best_idx = i
                break

        if obs_x is not None and obs_y is not None:
            for i in range(best_idx, len(trajectory) - 1):
                tx1, ty1, _ = trajectory[i]
                tx2, ty2, _ = trajectory[i + 1]
                d = self._pt_to_seg_dist(obs_x, obs_y, tx1, ty1, tx2, ty2)
                if d >= self.corridor_width * 2.0:
                    logger.info(f'Resume idx={i} | clear (d={d:.2f}m)')
                    return i
            return len(trajectory) - 1

        return best_idx
