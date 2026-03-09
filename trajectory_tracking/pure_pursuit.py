"""
pure_pursuit.py
Implements the Pure Pursuit controller for a differential-drive robot.
Computes linear and angular velocity commands by finding a lookahead point
on the trajectory ahead of the robot and steering toward it.

Linear velocity is now derived from the time-parameterized trajectory so that
the trapezoidal velocity profile (encoded in the timestamps) is respected.
Angular velocity is still computed geometrically via lookahead.
"""

import numpy as np
from geometry_msgs.msg import Twist
from typing import List, Tuple


class PurePursuit:
    def __init__(self, lookahead=0.5, max_lin_vel=0.3, max_ang_vel=1.0):
        self.lookahead = lookahead
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.trajectory = []
        self._current_idx = 0

    def set_trajectory(self, traj: List[Tuple[float, float, float]]):
        self.trajectory = traj
        self._current_idx = 0

    def _desired_velocity(self) -> float:
        """
        Compute desired linear speed from the timestamps of the current
        trajectory segment (current_idx → current_idx+1).
        Falls back to max_lin_vel at the last point or if dt ≤ 0.
        """
        idx = self._current_idx
        if idx + 1 >= len(self.trajectory):
            return self.max_lin_vel

        x0, y0, t0 = self.trajectory[idx]
        x1, y1, t1 = self.trajectory[idx + 1]

        dt = t1 - t0
        if dt <= 0.0:
            return self.max_lin_vel

        seg_dist = np.hypot(x1 - x0, y1 - y0)
        return float(np.clip(seg_dist / dt, 0.0, self.max_lin_vel))

    def compute_cmd(self, robot_x: float, robot_y: float, robot_theta: float) -> Twist:
        if not self.trajectory:
            return Twist()

        # Advance current index to closest point
        min_dist = float('inf')
        for i in range(self._current_idx, len(self.trajectory)):
            tx, ty, _ = self.trajectory[i]
            dist = np.hypot(robot_x - tx, robot_y - ty)
            if dist < min_dist:
                min_dist = dist
                self._current_idx = i

        # Find lookahead point forward from current index
        lookahead_idx = None
        for i in range(self._current_idx, len(self.trajectory)):
            tx, ty, _ = self.trajectory[i]
            dist = np.hypot(robot_x - tx, robot_y - ty)
            if dist >= self.lookahead:
                lookahead_idx = i
                break

        # If no lookahead point found, drive directly to final point
        if lookahead_idx is None:
            lx, ly, _ = self.trajectory[-1]
            dist_to_final = np.hypot(robot_x - lx, robot_y - ly)
            if dist_to_final < 0.05:
                return Twist()

            alpha = np.arctan2(ly - robot_y, lx - robot_x) - robot_theta
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
            twist = Twist()
            # CHANGED: use profile speed, still capped by distance for slowdown
            twist.linear.x = min(self._desired_velocity(), dist_to_final)
            twist.angular.z = np.clip(
                (2 * twist.linear.x / self.lookahead) * np.sin(alpha),
                -self.max_ang_vel, self.max_ang_vel,
            )
            return twist

        lx, ly, _ = self.trajectory[lookahead_idx]
        alpha = np.arctan2(ly - robot_y, lx - robot_x) - robot_theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        # CHANGED: profile-derived speed instead of hardcoded max
        lin_vel = self._desired_velocity()
        ang_vel = np.clip(
            (2 * lin_vel / self.lookahead) * np.sin(alpha),
            -self.max_ang_vel, self.max_ang_vel,
        )

        twist = Twist()
        twist.linear.x = lin_vel
        twist.angular.z = ang_vel
        return twist
