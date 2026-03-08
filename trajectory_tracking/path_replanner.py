"""
path_replanner.py
State machine: NORMAL -> obstacle detected -> BYPASSING -> wp2 reached -> NORMAL
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

from sensor_msgs.msg import LaserScan
import rclpy.logging

from trajectory_tracking.path_smoother import smooth_path
from trajectory_tracking.trajectory_generator import generate_trajectory

logger = rclpy.logging.get_logger('path_replanner')

Pose2D     = Tuple[float, float, float]
Point2D    = Tuple[float, float]
Trajectory = List[Tuple[float, float, float]]


@dataclass
class _ScanSnapshot:
    scan: LaserScan
    pose: Pose2D


class PathReplanner:

    def __init__(
        self,
        robot_radius:        float = 0.22,
        corridor_half_width: float = 0.50,
        lookahead_check:     float = 3.0,
        stop_distance:       float = 0.35,
        stop_arc_deg:        float = 50.0,
        replan_cooldown:     float = 2.0,
        bypass_timeout:      float = 10.0,
        lateral_clearance:   float = 0.18,
    ):
        self.robot_radius        = robot_radius
        self.corridor_half_width = corridor_half_width
        self.lookahead_check     = lookahead_check
        self.stop_distance       = stop_distance
        self.stop_arc_rad        = np.radians(stop_arc_deg / 2.0)
        self.replan_cooldown     = replan_cooldown
        self.bypass_timeout      = bypass_timeout
        self.lateral_clearance   = lateral_clearance
        self._min_lateral        = robot_radius + lateral_clearance

        self._snapshot:          Optional[_ScanSnapshot] = None
        self._state:             str               = 'NORMAL'
        self._bypass_wp2:        Optional[Point2D]  = None
        self._last_replan_time:  float              = -999.0
        self._bypass_start_time: float              = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def update_scan(self, scan: LaserScan, robot_pose: Pose2D) -> None:
        self._snapshot = _ScanSnapshot(scan=scan, pose=robot_pose)

    def emergency_stop_needed(
        self, snap: Optional[_ScanSnapshot] = None
    ) -> bool:
        if snap is None:
            snap = self._snapshot
        if snap is None:
            return False
        threshold = self.stop_distance * (0.6 if self._state == 'BYPASSING' else 1.0)
        scan = snap.scan
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            a = scan.angle_min + i * scan.angle_increment
            if abs(a) <= self.stop_arc_rad and r < threshold:
                return True
        return False

    def replan(
        self,
        current_trajectory:  Trajectory,
        original_trajectory: Trajectory,
        current_idx:         int,
        robot_x:             float,
        robot_y:             float,
        robot_theta:         float,
        max_vel:             float,
        remaining_time:      float,
        snap:                Optional[_ScanSnapshot] = None,
    ) -> Tuple[Trajectory, int, bool]:
        now = time.time()

        # Use provided snapshot or fall back to latest
        if snap is None:
            snap = self._snapshot
        if snap is None:
            return current_trajectory, current_idx, False

        # Project scan to world ONCE — all checks this cycle use this list
        world_pts = self._project_snapshot(snap)

        # ── BYPASSING: only check exit conditions ──────────────────────────
        if self._state == 'BYPASSING':
            if self._bypass_wp2 is not None:
                if np.hypot(robot_x - self._bypass_wp2[0],
                            robot_y - self._bypass_wp2[1]) < 0.30:
                    logger.info('Bypass complete, returning to NORMAL')
                    self._state = 'NORMAL'
                    self._bypass_wp2 = None

            if now - self._bypass_start_time > self.bypass_timeout:
                logger.warn('Bypass timed out, returning to NORMAL')
                self._state = 'NORMAL'
                self._bypass_wp2 = None

            if self._state == 'BYPASSING':
                return current_trajectory, current_idx, False

        # ── NORMAL: corridor detection ─────────────────────────────────────
        blocked, obs_x, obs_y = self._check_corridor(
            current_trajectory, current_idx,
            (robot_x, robot_y, robot_theta), world_pts,
        )

        if not blocked:
            return current_trajectory, current_idx, False

        if now - self._last_replan_time < self.replan_cooldown:
            return current_trajectory, current_idx, False

        logger.warn(f'Obstacle at ({obs_x:.2f}, {obs_y:.2f}), replanning')

        new_traj = self._build_bypass(
            original_trajectory,
            (robot_x, robot_y, robot_theta),
            obs_x, obs_y, max_vel, remaining_time, world_pts,
        )

        if new_traj is None:
            logger.warn('No valid bypass found, holding course')
            return current_trajectory, current_idx, False

        pp_start = self._closest_idx(new_traj, robot_x, robot_y)
        self._last_replan_time  = now
        self._bypass_start_time = now
        self._state             = 'BYPASSING'

        logger.info(
            f'Replanned | pp_start={pp_start}/{len(new_traj)} | '
            f'wp2=({self._bypass_wp2[0]:.2f},{self._bypass_wp2[1]:.2f})'
        )
        return new_traj, pp_start, True

    # ── Internal: scan projection ──────────────────────────────────────────

    def _project_snapshot(self, snap: _ScanSnapshot) -> List[Point2D]:
        """World-frame points from snapshot. Forward 180 deg only to prevent
        rear/side walls from appearing as corridor obstacles."""
        rx, ry, rth = snap.pose
        scan = snap.scan
        pts = []
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            a = scan.angle_min + i * scan.angle_increment
            if abs(a) > np.pi / 2.0:
                continue
            pts.append((rx + r * np.cos(rth + a),
                        ry + r * np.sin(rth + a)))
        return pts

    # ── Internal: detection ────────────────────────────────────────────────

    def _check_corridor(
        self,
        trajectory:  Trajectory,
        current_idx: int,
        robot_pose:  Pose2D,
        world_pts:   List[Point2D],
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        rx, ry, _ = robot_pose
        if not world_pts:
            return False, None, None

        segments = []
        for i in range(current_idx, len(trajectory) - 1):
            tx, ty, _ = trajectory[i]
            if np.hypot(tx - rx, ty - ry) > self.lookahead_check:
                break
            nx, ny, _ = trajectory[i + 1]
            segments.append((tx, ty, nx, ny))

        if not segments:
            return False, None, None

        blocking = []
        for wx, wy in world_pts:
            for x1, y1, x2, y2 in segments:
                if self._pt_seg_dist(wx, wy, x1, y1, x2, y2) < self.corridor_half_width:
                    blocking.append((wx, wy))
                    break

        if not blocking:
            return False, None, None

        # Use closest blocking point, not mean to avoid wall-return bias
        dists   = [np.hypot(p[0] - rx, p[1] - ry) for p in blocking]
        closest = blocking[int(np.argmin(dists))]
        return True, closest[0], closest[1]

    # ── Internal: bypass planning ──────────────────────────────────────────

    def _build_bypass(
        self,
        original_trajectory: Trajectory,
        robot_pose:          Pose2D,
        obs_x: float, obs_y: float,
        max_vel: float, remaining_time: float,
        world_pts: List[Point2D],
    ) -> Optional[Trajectory]:
        robot_x, robot_y, _ = robot_pose

        obs_idx  = int(np.argmin([np.hypot(t[0]-obs_x, t[1]-obs_y)
                                   for t in original_trajectory]))
        path_dir = self._path_direction(original_trajectory, obs_idx)

        # Scan hits near face, shift to approximate obstacle centre
        obs_cx = obs_x + path_dir[0] * 0.25
        obs_cy = obs_y + path_dir[1] * 0.25

        side = self._choose_side()
        for attempt in [side, 'left' if side == 'right' else 'right']:
            traj = self._try_side(
                original_trajectory, obs_idx, obs_cx, obs_cy,
                path_dir, attempt, robot_x, robot_y,
                max_vel, remaining_time, world_pts,
            )
            if traj is not None:
                if attempt != side:
                    logger.info(f'Side flipped to {attempt}')
                return traj
        return None

    def _try_side(
        self,
        original_trajectory: Trajectory,
        obs_idx: int,
        obs_cx: float, obs_cy: float,
        path_dir: np.ndarray,
        side: str,
        robot_x: float, robot_y: float,
        max_vel: float, remaining_time: float,
        world_pts: List[Point2D],
    ) -> Optional[Trajectory]:

        perp = np.array([-path_dir[1], path_dir[0]])
        if side == 'right':
            perp = -perp

        free    = self._cast_ray(obs_cx, obs_cy, perp[0], perp[1], world_pts)
        lateral = float(np.clip(free * 0.55, self._min_lateral, 1.2))

        wp1 = (obs_cx + perp[0]*lateral - path_dir[0]*lateral*0.7,
               obs_cy + perp[1]*lateral - path_dir[1]*lateral*0.7)
        wp2 = (obs_cx + perp[0]*lateral + path_dir[0]*lateral*1.0,
               obs_cy + perp[1]*lateral + path_dir[1]*lateral*1.0)

        cr = self.robot_radius + 0.05
        if not self._point_is_free(wp1, cr, world_pts):   return None
        if not self._point_is_free(wp2, cr, world_pts):   return None
        if not self._segment_is_free(wp1, wp2, self.robot_radius, world_pts): return None

        resume_idx = self._find_resume_idx(
            original_trajectory, obs_idx, wp2, path_dir, obs_cx, obs_cy
        )
        if resume_idx >= len(original_trajectory) - 1:
            return None

        path_after = [(t[0], t[1]) for t in original_trajectory[resume_idx:]]
        if len(path_after) < 2:
            return None

        new_path         = [(robot_x, robot_y), wp1, wp2] + path_after
        self._bypass_wp2 = wp2

        try:
            smoothed = smooth_path(new_path, num_points=200)
            return generate_trajectory(smoothed, max(remaining_time, 5.0), max_vel)
        except Exception as e:
            logger.error(f'Smoothing/generation failed: {e}')
            self._bypass_wp2 = None
            return None

    # ── Internal: geometry ─────────────────────────────────────────────────

    @staticmethod
    def _pt_seg_dist(px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2-x1, y2-y1
        sq = dx*dx + dy*dy
        if sq < 1e-9:
            return np.hypot(px-x1, py-y1)
        t = np.clip(((px-x1)*dx + (py-y1)*dy) / sq, 0.0, 1.0)
        return np.hypot(px-(x1+t*dx), py-(y1+t*dy))

    @staticmethod
    def _point_is_free(pt: Point2D, radius: float, world_pts: List[Point2D]) -> bool:
        px, py = pt
        return all(np.hypot(wx-px, wy-py) >= radius for wx, wy in world_pts)

    @staticmethod
    def _segment_is_free(
        p1: Point2D, p2: Point2D, clearance: float, world_pts: List[Point2D]
    ) -> bool:
        return all(
            PathReplanner._pt_seg_dist(wx, wy, p1[0], p1[1], p2[0], p2[1]) >= clearance
            for wx, wy in world_pts
        )

    @staticmethod
    def _cast_ray(
        ox, oy, dx, dy, world_pts: List[Point2D], max_dist: float = 2.5
    ) -> float:
        min_d = max_dist
        for wx, wy in world_pts:
            along = (wx-ox)*dx + (wy-oy)*dy
            if along < 0.05 or along > max_dist:
                continue
            if abs((wx-ox)*dy - (wy-oy)*dx) < 0.12:
                min_d = min(min_d, along)
        return min_d

    def _choose_side(self) -> str:
        snap = self._snapshot
        if snap is None:
            return 'left'
        scan = snap.scan
        ls, rs, ln, rn = 0.0, 0.0, 0, 0
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            a = scan.angle_min + i * scan.angle_increment
            if abs(a) > np.pi / 2.0:
                continue
            if a > 0: ls += r; ln += 1
            else:     rs += r; rn += 1
        return 'left' if (ls/ln if ln else 0) >= (rs/rn if rn else 0) else 'right'

    @staticmethod
    def _path_direction(trajectory: Trajectory, idx: int) -> np.ndarray:
        end = min(idx + 8, len(trajectory) - 1)
        tx1, ty1, _ = trajectory[idx]
        tx2, ty2, _ = trajectory[end]
        dx, dy = tx2-tx1, ty2-ty1
        norm = np.hypot(dx, dy)
        return np.array([dx/norm, dy/norm]) if norm > 1e-6 else np.array([1.0, 0.0])

    @staticmethod
    def _closest_idx(trajectory: Trajectory, rx: float, ry: float) -> int:
        return int(np.argmin([np.hypot(t[0]-rx, t[1]-ry) for t in trajectory]))

    def _find_resume_idx(
        self,
        trajectory: Trajectory,
        obs_idx: int,
        bypass_wp2: Point2D,
        path_dir: np.ndarray,
        obs_cx: float, obs_cy: float,
    ) -> int:
        bx, by = bypass_wp2
        start  = obs_idx
        for i in range(obs_idx, len(trajectory)):
            tx, ty, _ = trajectory[i]
            if (tx-bx)*path_dir[0] + (ty-by)*path_dir[1] > 0.3:
                start = i
                break

        needed = self.corridor_half_width * 2.0
        for i in range(start, len(trajectory) - 1):
            tx1, ty1, _ = trajectory[i]
            tx2, ty2, _ = trajectory[i+1]
            if self._pt_seg_dist(obs_cx, obs_cy, tx1, ty1, tx2, ty2) >= needed:
                return i
        return len(trajectory) - 1
