import pytest
import numpy as np
from trajectory_tracking.path_smoother import smooth_path
from trajectory_tracking.trajectory_generator import generate_trajectory
from trajectory_tracking.pure_pursuit import PurePursuit


def _run_pipeline(waypoints, robot_x=0.0, robot_y=0.0, robot_theta=0.0):
    smoothed = smooth_path(waypoints, num_points=100)
    traj = generate_trajectory(smoothed, total_time=30.0, max_vel=0.3)
    pp = PurePursuit(lookahead=0.5, max_lin_vel=0.3)
    pp.set_trajectory(traj)
    cmd = pp.compute_cmd(robot_x, robot_y, robot_theta)
    return traj, pp, cmd


def test_straight_path_produces_forward_motion():
    wps = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    _, _, cmd = _run_pipeline(wps)
    assert cmd.linear.x > 0.0
    assert abs(cmd.angular.z) < 0.3  # nearly straight, low steering


def test_curved_path_produces_nonzero_steering():
    wps = [(0.0, 0.0), (2.0, 1.0), (4.0, 0.0), (6.0, 1.0)]
    _, _, cmd = _run_pipeline(wps)
    assert cmd.linear.x > 0.0
    assert abs(cmd.angular.z) > 0.0  # curved path requires steering


def test_robot_offset_from_path_corrects_back():
    """Robot starts 0.3m to the right of a straight path — should steer left."""
    wps = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    _, _, cmd = _run_pipeline(wps, robot_x=0.0, robot_y=-0.3, robot_theta=0.0)
    assert cmd.angular.z > 0.0  # steer left (positive) to get back on path


def test_velocity_limits_respected_full_pipeline():
    wps = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
    _, _, cmd = _run_pipeline(wps)
    assert cmd.linear.x <= 0.3 + 1e-6
    assert abs(cmd.angular.z) <= 1.0 + 1e-6


def test_trajectory_output_feeds_pure_pursuit_without_error():
    """Full pipeline must complete without exceptions for a complex path."""
    wps = [(float(i), float(i % 2)) for i in range(6)]
    traj, pp, cmd = _run_pipeline(wps)
    assert len(traj) == 100
    assert traj[0][2] == pytest.approx(0.0, abs=1e-6)
    assert traj[-1][2] == pytest.approx(30.0, abs=1e-4)
    assert cmd.linear.x >= 0.0
