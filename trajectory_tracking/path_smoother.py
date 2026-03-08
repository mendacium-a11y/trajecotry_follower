from scipy.interpolate import splprep, splev
import numpy as np

def smooth_path(waypoints, num_points=200):
    """Input: [(x0,y0), (x1,y1), ...] → Output: smooth [(x,y), ...]"""
    pts = np.array(waypoints).T
    tck, u = splprep(pts, s=0.0)  # s=0 for interpolation
    u_new = np.linspace(0, 1, num_points)
    x, y = splev(u_new, tck)
    return list(zip(x, y))
