import numpy as np
from typing import List, Tuple

def generate_trajectory(
    smoothed_path: List[Tuple[float, float]],
    total_time: float = 30.0,
    max_vel: float = 0.3,
    accel: float = 0.5
) -> List[Tuple[float, float, float]]:

    pts = np.array(smoothed_path)
    diffs = np.diff(pts, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(distances)])
    total_dist = cum_dist[-1]

    # Trapezoidal profile
    t_accel = max_vel / accel
    dist_accel = 0.5 * accel * t_accel**2

    if 2 * dist_accel > total_dist:
        # Short path: pure accel/decel only
        t_accel = np.sqrt(total_dist / accel)
        max_vel = accel * t_accel
        dist_accel = 0.5 * accel * t_accel**2

    dist_cruise = total_dist - 2 * dist_accel
    t_cruise = dist_cruise / max_vel

    # FIX: actually map distances to times using trapezoid
    def dist_to_time(d):
        if d <= dist_accel:
            return np.sqrt(2 * d / accel)
        elif d <= dist_accel + dist_cruise:
            return t_accel + (d - dist_accel) / max_vel
        else:
            d_decel = d - dist_accel - dist_cruise
            return t_accel + t_cruise + (max_vel - np.sqrt(max(0, max_vel**2 - 2 * accel * d_decel))) / accel

    times = np.array([dist_to_time(d) for d in cum_dist])
    # Normalize to total_time
    if times[-1] > 0:
        times = times / times[-1] * total_time

    return [(float(x), float(y), float(t)) for (x, y), t in zip(smoothed_path, times)]
