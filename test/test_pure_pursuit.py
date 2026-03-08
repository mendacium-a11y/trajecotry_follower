import pytest
import numpy as np
from trajectory_tracking.pure_pursuit import PurePursuit


def _straight_traj(length=5.0, n=50):
    """Straight trajectory along x-axis."""
    return [(length * i / (n - 1), 0.0, float(i)) for i in range(n)]


# ── Empty / unset trajectory ───────────────────────────────────────────────

def test_empty_trajectory_returns_zero_twist():
    pp = PurePursuit()
    cmd = pp.compute_cmd(0.0, 0.0, 0.0)
    assert cmd.linear.x  == 0.0
    assert cmd.angular.z == 0.0

def test_set_trajectory_resets_index():
    pp = PurePursuit()
    traj = _straight_traj()
    pp.set_trajectory(traj)
    pp._current_idx = 20
    pp.set_trajectory(traj)   # reset
    assert pp._current_idx == 0

# ── At goal ────────────────────────────────────────────────────────────────

def test_at_final_point_returns_zero_twist():
    """Robot already at the last point — should stop."""
    pp = PurePursuit(lookahead=0.5)
    traj = _straight_traj(length=5.0)
    pp.set_trajectory(traj)
    pp._current_idx = len(traj) - 1
    cmd = pp.compute_cmd(5.0, 0.0, 0.0)
    assert cmd.linear.x  == pytest.approx(0.0, abs=0.05)
    assert cmd.angular.z == pytest.approx(0.0, abs=0.05)

# ── Velocity limits ────────────────────────────────────────────────────────

def test_linear_velocity_does_not_exceed_max():
    pp = PurePursuit(lookahead=0.5, max_lin_vel=0.3)
    traj = _straight_traj()
    pp.set_trajectory(traj)
    cmd = pp.compute_cmd(0.0, 0.0, 0.0)
    assert cmd.linear.x <= 0.3 + 1e-6

def test_angular_velocity_does_not_exceed_max():
    pp = PurePursuit(lookahead=0.5, max_lin_vel=0.3, max_ang_vel=1.0)
    traj = _straight_traj()
    pp.set_trajectory(traj)
    # Robot facing away from path — max angular correction expected
    cmd = pp.compute_cmd(0.0, 0.0, np.pi)
    assert abs(cmd.angular.z) <= 1.0 + 1e-6

# ── Direction ──────────────────────────────────────────────────────────────

def test_robot_behind_path_drives_forward():
    pp = PurePursuit(lookahead=0.5, max_lin_vel=0.3)
    traj = _straight_traj()
    pp.set_trajectory(traj)
    cmd = pp.compute_cmd(0.0, 0.0, 0.0)
    assert cmd.linear.x > 0.0

def test_target_left_gives_positive_angular():
    """Target to the left → positive angular.z (CCW)."""
    pp = PurePursuit(lookahead=0.5, max_lin_vel=0.3)
    # Trajectory goes up along y-axis; robot faces along x → target is to the left
    traj = [(0.0, float(i) * 0.2, float(i)) for i in range(20)]
    pp.set_trajectory(traj)
    cmd = pp.compute_cmd(0.0, 0.0, 0.0)
    assert cmd.angular.z > 0.0

def test_target_right_gives_negative_angular():
    """Target to the right → negative angular.z (CW)."""
    pp = PurePursuit(lookahead=0.5, max_lin_vel=0.3)
    traj = [(0.0, -float(i) * 0.2, float(i)) for i in range(20)]
    pp.set_trajectory(traj)
    cmd = pp.compute_cmd(0.0, 0.0, 0.0)
    assert cmd.angular.z < 0.0

# ── Index advancement ──────────────────────────────────────────────────────

def test_current_index_advances_as_robot_moves():
    pp = PurePursuit(lookahead=0.5, max_lin_vel=0.3)
    traj = _straight_traj(length=5.0, n=50)
    pp.set_trajectory(traj)
    pp.compute_cmd(0.0, 0.0, 0.0)
    idx_start = pp._current_idx
    pp.compute_cmd(2.5, 0.0, 0.0)
    assert pp._current_idx > idx_start
