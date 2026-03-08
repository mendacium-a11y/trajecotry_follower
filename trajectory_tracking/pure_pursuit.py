"""
pure_pursuit.py
Implements the Pure Pursuit controller for a differential-drive robot.
Computes linear and angular velocity commands by finding a lookahead point
on the trajectory ahead of the robot and steering toward it.
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
        self._current_idx = 0  # Trajectory progress index

    def set_trajectory(self, traj: List[Tuple[float, float, float]]):
        self.trajectory = traj
        self._current_idx = 0

    def compute_cmd(self, robot_x: float, robot_y: float, robot_theta: float) -> Twist:
        if not self.trajectory:
            return Twist()

        # Advance current index to closest point
        min_dist = float('inf')
        for i in range(self._current_idx, len(self.trajectory)):
            tx, ty, _ = self.trajectory[i]
            dist = np.sqrt((robot_x - tx)**2 + (robot_y - ty)**2)
            if dist < min_dist:
                min_dist = dist
                self._current_idx = i

        # Find lookahead point forward from current index
        lookahead_idx = None
        for i in range(self._current_idx, len(self.trajectory)):
            tx, ty, _ = self.trajectory[i]
            dist = np.sqrt((robot_x - tx)**2 + (robot_y - ty)**2)
            if dist >= self.lookahead:
                lookahead_idx = i
                break

        # If no lookahead point found, drive directly to final point
        if lookahead_idx is None:
            lx, ly, _ = self.trajectory[-1]
            dist_to_final = np.sqrt((robot_x - lx)**2 + (robot_y - ly)**2)
            if dist_to_final < 0.05:   # Very close, perform full stop
                return Twist()
            # Drive straight toward final point at reduced speed
            alpha = np.arctan2(ly - robot_y, lx - robot_x) - robot_theta
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
            twist = Twist()
            twist.linear.x = min(self.max_lin_vel, dist_to_final)  # Slow down near goal
            twist.angular.z = np.clip(
                (2 * twist.linear.x / self.lookahead) * np.sin(alpha),
                -self.max_ang_vel, self.max_ang_vel
            )
            return twist

        lx, ly, _ = self.trajectory[lookahead_idx]
        alpha = np.arctan2(ly - robot_y, lx - robot_x) - robot_theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        lin_vel = self.max_lin_vel
        ang_vel = (2 * lin_vel / self.lookahead) * np.sin(alpha)
        ang_vel = np.clip(ang_vel, -self.max_ang_vel, self.max_ang_vel)

        twist = Twist()
        twist.linear.x = lin_vel
        twist.angular.z = ang_vel
        return twist
