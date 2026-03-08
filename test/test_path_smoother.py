import pytest
import numpy as np
from trajectory_tracking.path_smoother import smooth_path


# ── Output shape ───────────────────────────────────────────────────────────

def test_output_length_default():
    wps = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0)]
    result = smooth_path(wps)
    assert len(result) == 200

def test_output_length_custom():
    wps = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
    result = smooth_path(wps, num_points=50)
    assert len(result) == 50

def test_output_is_list_of_tuples():
    wps = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0)]
    result = smooth_path(wps)
    assert isinstance(result, list)
    assert all(len(pt) == 2 for pt in result)

# ── Endpoints ──────────────────────────────────────────────────────────────

def test_starts_near_first_waypoint():
    wps = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0)]
    result = smooth_path(wps)
    assert np.hypot(result[0][0] - 0.0, result[0][1] - 0.0) < 0.15

def test_ends_near_last_waypoint():
    wps = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0)]
    result = smooth_path(wps)
    assert np.hypot(result[-1][0] - 3.0, result[-1][1] - 0.0) < 0.15

# ── Edge cases ─────────────────────────────────────────────────────────────

def test_minimum_two_waypoints():
    """Assignment minimum — must not crash with exactly 2 waypoints."""
    wps = [(0.0, 0.0), (3.0, 0.0)]
    result = smooth_path(wps)
    assert len(result) == 200

def test_collinear_waypoints():
    """All points on a straight line — no oscillation expected."""
    wps = [(float(i), 0.0) for i in range(6)]
    result = smooth_path(wps)
    y_vals = [pt[1] for pt in result]
    assert max(abs(y) for y in y_vals) < 0.1  # stays on the line

def test_duplicate_points_handled():
    """Deduplicated points should not cause splprep to crash."""
    wps = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0)]
    result = smooth_path(wps)
    assert len(result) == 200

def test_close_bypass_points():
    """Simulates replanner input: robot_pos, wp1, wp2 are <0.5m apart."""
    wps = [
        (0.0,  0.0),
        (0.3,  0.2),   # wp1 — close
        (0.6,  0.0),   # wp2 — close
        (2.0,  0.0),
        (4.0,  0.0),
    ]
    result = smooth_path(wps)
    # Must not produce wild oscillations — max lateral deviation < 1m
    y_vals = [pt[1] for pt in result]
    assert max(abs(y) for y in y_vals) < 1.0
