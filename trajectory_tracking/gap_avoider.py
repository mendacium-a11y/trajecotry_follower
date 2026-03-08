import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import rclpy.logging

logger = rclpy.logging.get_logger('gap_avoider')


class GapAvoider:
    def __init__(
        self,
        robot_width: float = 0.40,    # TurtleBot4 ~0.35m + 5cm margin
        stop_distance: float = 0.35,  # hard stop threshold
        safe_distance: float = 1.2,   # blending starts here
        scan_arc_deg: float = 180.0,  # forward scan arc (±90°)
        gap_gain: float = 2.0,        # P-gain: gap angle → angular velocity
        log_every_n: int = 20,
    ):
        self.robot_width   = robot_width
        self.stop_distance = stop_distance
        self.safe_distance = safe_distance
        self.scan_arc_rad  = np.radians(scan_arc_deg / 2.0)
        self.gap_gain      = gap_gain
        self._log_n        = log_every_n
        self._count        = 0
        self.latest_scan: LaserScan = None

    def update_scan(self, scan: LaserScan):
        self.latest_scan = scan

    def _get_forward_scan(self):
        """Returns [(angle, distance)] for the forward arc only."""
        scan = self.latest_scan
        points = []
        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) > self.scan_arc_rad:
                continue
            if not np.isfinite(r) or r < scan.range_min:
                r = 0.0            # invalid → treat as wall right here
            elif r > scan.range_max:
                r = scan.range_max
            points.append((angle, r))
        return points

    def _find_gaps(self, scan_points: list) -> list:
        """
        Find contiguous free corridors wider than robot_width.
        Returns list of dicts: {center, width_m, avg_dist}
        """
        free_thresh = self.stop_distance + 0.25  # point is "open" above this

        gaps = []
        in_gap      = False
        gap_start   = None
        gap_start_i = None

        for i, (angle, dist) in enumerate(scan_points):
            free = dist > free_thresh
            if free and not in_gap:
                in_gap      = True
                gap_start   = angle
                gap_start_i = i
            elif not free and in_gap:
                in_gap = False
                self._close_gap(scan_points, gap_start, gap_start_i, i, gaps)

        if in_gap:  # gap open at the edge of scan
            self._close_gap(scan_points, gap_start, gap_start_i,
                            len(scan_points), gaps)
        return gaps

    def _close_gap(self, pts, start_angle, start_i, end_i, gaps):
        end_angle    = pts[end_i - 1][0]
        dists        = [pts[j][1] for j in range(start_i, end_i)]
        avg_dist     = float(np.mean(dists))
        angular_span = abs(end_angle - start_angle)
        width_m      = avg_dist * angular_span        # arc length approximation
        if width_m >= self.robot_width:
            gaps.append({
                'center':   (start_angle + end_angle) / 2.0,
                'width_m':  width_m,
                'avg_dist': avg_dist,
            })

    def modify_cmd(self, twist: Twist) -> Twist:
        if self.latest_scan is None:
            logger.warn('⚠️ No scan data — gap avoidance inactive')
            return twist

        self._count += 1
        pts = self._get_forward_scan()
        if not pts:
            return twist

        dists    = [d for _, d in pts]
        min_dist = float(min(dists))

        # ── Hard stop ─────────────────────────────────────────────
        if min_dist < self.stop_distance:
            left_mean  = float(np.mean([d for a, d in pts if a > 0] or [0.0]))
            right_mean = float(np.mean([d for a, d in pts if a < 0] or [0.0]))
            out = Twist()
            out.linear.x  = 0.0
            out.angular.z = 0.8 if left_mean > right_mean else -0.8
            if self._count % self._log_n == 0:
                logger.warn(
                    f'🛑 STOP min={min_dist:.2f}m | '
                    f'rotating {"LEFT" if out.angular.z > 0 else "RIGHT"}'
                )
            return out

        # ── Blend factor α ────────────────────────────────────────
        # α=1 → full Pure Pursuit (clear), α=0 → full gap steering (close)
        alpha = float(np.clip(
            (min_dist - self.stop_distance) / (self.safe_distance - self.stop_distance),
            0.0, 1.0
        ))

        if alpha >= 1.0:
            if self._count % self._log_n == 0:
                logger.info(f'[GAP] CLEAR | min={min_dist:.2f}m')
            return twist  # completely clear — pass through unchanged

        # ── Find navigable gaps ───────────────────────────────────
        gaps = self._find_gaps(pts)

        if not gaps:
            left_mean  = float(np.mean([d for a, d in pts if a > 0] or [0.0]))
            right_mean = float(np.mean([d for a, d in pts if a < 0] or [0.0]))
            out = Twist()
            out.linear.x  = 0.0
            out.angular.z = 0.8 if left_mean > right_mean else -0.8
            logger.warn(f'⚠️ No gap found — rotating | min={min_dist:.2f}m')
            return out

        # ── Pick gap closest to Pure Pursuit's desired heading ────
        # Pure Pursuit angular.z encodes desired curvature
        # Back-convert to approximate angle: angle ≈ ω / gain
        desired_angle = float(np.clip(
            twist.angular.z / self.gap_gain, -self.scan_arc_rad, self.scan_arc_rad
        ))
        best_gap = min(gaps, key=lambda g: abs(g['center'] - desired_angle))

        # ── Gap P-controller → angular velocity ───────────────────
        gap_angular_z = self.gap_gain * best_gap['center']

        # ── Blend ─────────────────────────────────────────────────
        blended_angular_z = alpha * twist.angular.z + (1.0 - alpha) * gap_angular_z
        blended_linear_x  = twist.linear.x * (alpha + (1.0 - alpha) * 0.4)

        out = Twist()
        out.linear.x  = float(np.clip(blended_linear_x, 0.0, 0.5))
        out.angular.z = float(np.clip(blended_angular_z, -2.0, 2.0))

        if self._count % self._log_n == 0:
            logger.info(
                f'[GAP] min={min_dist:.2f}m | α={alpha:.2f} | '
                f'gaps={len(gaps)} | best_gap={np.degrees(best_gap["center"]):.1f}° '
                f'w={best_gap["width_m"]:.2f}m | '
                f'ω: {twist.angular.z:.2f}→{blended_angular_z:.2f}'
            )

        return out
