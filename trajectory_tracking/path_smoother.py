"""
path_smoother.py
Provides functionality to smooth a coarse set of 2D waypoints into a 
continuous path using B-Spline interpolation. Falls back to linear 
interpolation if too few waypoints are provided.
"""

import numpy as np
from scipy.interpolate import splprep, splev


def smooth_path(waypoints, num_points: int = 200):
    pts = np.array(waypoints, dtype=float)

    # Deduplicate points closer than 0.08m — prevents spline knot clustering
    # when bypass waypoints (robot_pos, wp1, wp2) are close together
    filtered = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - filtered[-1]) >= 0.08:
            filtered.append(p)

    n = len(filtered)

    # Fewer than 4 points → linear interpolation
    if n < 4:
        arr    = np.array(filtered)
        cumlen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))])
        if cumlen[-1] < 1e-6:
            return [tuple(arr[0])] * num_points
        t_norm = cumlen / cumlen[-1]
        t_new  = np.linspace(0, 1, num_points)
        return list(zip(np.interp(t_new, t_norm, arr[:, 0]),
                        np.interp(t_new, t_norm, arr[:, 1])))

    arr = np.array(filtered).T
    # s > 0 prevents oscillation on close waypoints (s=0.0 causes Runge overshoot)
    try:
        tck, _ = splprep(arr, s=n * 0.01, k=min(3, n - 1))
    except Exception:
        tck, _ = splprep(arr, s=n * 0.1, k=1)

    x, y = splev(np.linspace(0, 1, num_points), tck)
    return list(zip(x, y))
