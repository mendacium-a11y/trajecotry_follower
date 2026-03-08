import pytest
import numpy as np
from trajectory_tracking.path_smoother import smooth_path
from trajectory_tracking.trajectory_generator import generate_trajectory


def _straight_path(n=20):
    return [(float(i) / (n - 1) * 5.0, 0.0) for i in range(n)]


# ── Output format ──────────────────────────────────────────────────────────

def test_output_length_matches_input():
    path = _straight_path(50)
    traj = generate_trajectory(path, total_time=30.0, max_vel=0.3)
    assert len(traj) == 50

def test_output_is_xyz_tuples():
    path = _straight_path()
    traj = generate_trajectory(path)
    assert all(len(pt) == 3 for pt in traj)

def test_xy_matches_input_path():
    path = _straight_path()
    traj = generate_trajectory(path)
    for (px, py), (tx, ty, _) in zip(path, traj):
        assert abs(px - tx) < 1e-9
        assert abs(py - ty) < 1e-9

# ── Timing properties ──────────────────────────────────────────────────────

def test_time_starts_at_zero():
    traj = generate_trajectory(_straight_path())
    assert traj[0][2] == pytest.approx(0.0, abs=1e-6)

def test_time_ends_at_total_time():
    traj = generate_trajectory(_straight_path(), total_time=45.0)
    assert traj[-1][2] == pytest.approx(45.0, abs=1e-4)

def test_time_is_monotonically_increasing():
    traj = generate_trajectory(_straight_path(), total_time=30.0)
    times = [pt[2] for pt in traj]
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

def test_no_negative_timestamps():
    traj = generate_trajectory(_straight_path())
    assert all(pt[2] >= 0.0 for pt in traj)

# ── Edge cases ─────────────────────────────────────────────────────────────

def test_very_short_path():
    """Path shorter than two accel distances — pure accel/decel profile."""
    path = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)]
    traj = generate_trajectory(path, total_time=10.0, max_vel=0.3, accel=0.5)
    times = [pt[2] for pt in traj]
    assert times[-1] == pytest.approx(10.0, abs=1e-4)
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

def test_two_point_path():
    path = [(0.0, 0.0), (5.0, 0.0)]
    traj = generate_trajectory(path, total_time=30.0, max_vel=0.3)
    assert len(traj) == 2
    assert traj[0][2] == pytest.approx(0.0, abs=1e-6)
    assert traj[-1][2] == pytest.approx(30.0, abs=1e-4)

def test_smoothed_path_roundtrip():
    """Full pipeline: waypoints → smooth → trajectory."""
    wps = [(0.0, 0.0), (2.0, 1.0), (4.0, 0.0), (6.0, 1.0)]
    smoothed = smooth_path(wps, num_points=100)
    traj = generate_trajectory(smoothed, total_time=60.0, max_vel=0.3)
    times = [pt[2] for pt in traj]
    assert traj[-1][2] == pytest.approx(60.0, abs=1e-4)
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))
