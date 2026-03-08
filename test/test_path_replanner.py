import pytest
import numpy as np
from unittest.mock import MagicMock
from trajectory_tracking.path_replanner import PathReplanner, _ScanSnapshot


def _make_scan(ranges, angle_min=-np.pi/2, angle_max=np.pi/2):
    """Build a minimal mock LaserScan."""
    scan = MagicMock()
    scan.ranges       = ranges
    scan.angle_min    = angle_min
    scan.angle_max    = angle_max
    scan.angle_increment = (angle_max - angle_min) / max(len(ranges) - 1, 1)
    scan.range_min    = 0.1
    scan.range_max    = 10.0
    return scan


def _straight_traj(length=8.0, n=80):
    return [(length * i / (n - 1), 0.0, float(i)) for i in range(n)]


def _clear_scan(n=180):
    """Scan with nothing closer than 5m — open field."""
    return _make_scan([5.0] * n)


def _obstacle_scan(obs_dist=1.5, n=180):
    """Obstacle directly ahead at obs_dist metres."""
    ranges = [5.0] * n
    mid = n // 2
    for k in range(-5, 6):
        ranges[mid + k] = obs_dist
    return _make_scan(ranges)


# ── Initial state ──────────────────────────────────────────────────────────

def test_initial_state_is_normal():
    rp = PathReplanner()
    assert rp._state == 'NORMAL'

def test_no_snapshot_returns_false_estop():
    rp = PathReplanner()
    assert rp.emergency_stop_needed() is False

def test_no_snapshot_replan_returns_unchanged():
    rp = PathReplanner()
    traj = _straight_traj()
    result, idx, did = rp.replan(traj, traj, 0, 0.0, 0.0, 0.0, 0.3, 30.0)
    assert did is False
    assert result is traj

# ── Emergency stop ─────────────────────────────────────────────────────────

def test_estop_triggered_by_close_obstacle():
    rp   = PathReplanner(stop_distance=0.5)
    scan = _obstacle_scan(obs_dist=0.3)
    snap = _ScanSnapshot(scan=scan, pose=(0.0, 0.0, 0.0))
    assert rp.emergency_stop_needed(snap) is True

def test_estop_not_triggered_by_far_obstacle():
    rp   = PathReplanner(stop_distance=0.5)
    scan = _obstacle_scan(obs_dist=2.0)
    snap = _ScanSnapshot(scan=scan, pose=(0.0, 0.0, 0.0))
    assert rp.emergency_stop_needed(snap) is False

def test_estop_relaxed_during_bypassing():
    """Threshold is 60% during BYPASSING — same distance should not trigger."""
    rp        = PathReplanner(stop_distance=0.5)
    rp._state = 'BYPASSING'
    scan      = _obstacle_scan(obs_dist=0.35)   # inside 0.5 but outside 0.5*0.6=0.3
    snap      = _ScanSnapshot(scan=scan, pose=(0.0, 0.0, 0.0))
    assert rp.emergency_stop_needed(snap) is False

# ── Corridor detection ─────────────────────────────────────────────────────

def test_clear_path_does_not_replan():
    rp   = PathReplanner(replan_cooldown=0.0)
    traj = _straight_traj()
    scan = _clear_scan()
    rp.update_scan(scan, (0.0, 0.0, 0.0))
    _, _, did = rp.replan(traj, traj, 0, 0.0, 0.0, 0.0, 0.3, 30.0)
    assert did is False

def test_obstacle_in_corridor_triggers_replan():
    rp   = PathReplanner(replan_cooldown=0.0, corridor_half_width=0.5)
    traj = _straight_traj(length=8.0)
    # Obstacle 1.5m ahead, directly on the path
    scan = _obstacle_scan(obs_dist=1.5)
    rp.update_scan(scan, (0.0, 0.0, 0.0))
    _, _, did = rp.replan(traj, traj, 0, 0.0, 0.0, 0.0, 0.3, 30.0)
    assert did is True

def test_replan_sets_state_to_bypassing():
    rp   = PathReplanner(replan_cooldown=0.0)
    traj = _straight_traj(length=8.0)
    scan = _obstacle_scan(obs_dist=1.5)
    rp.update_scan(scan, (0.0, 0.0, 0.0))
    rp.replan(traj, traj, 0, 0.0, 0.0, 0.0, 0.3, 30.0)
    assert rp._state == 'BYPASSING'

# ── Cooldown ───────────────────────────────────────────────────────────────

def test_cooldown_prevents_immediate_second_replan():
    rp   = PathReplanner(replan_cooldown=999.0)   # huge cooldown
    traj = _straight_traj(length=8.0)
    scan = _obstacle_scan(obs_dist=1.5)
    rp.update_scan(scan, (0.0, 0.0, 0.0))
    # Force first replan by bypassing cooldown
    rp._last_replan_time = -999.0
    rp._state = 'NORMAL'
    _, _, did1 = rp.replan(traj, traj, 0, 0.0, 0.0, 0.0, 0.3, 30.0)
    rp._state = 'NORMAL'   # reset to allow detection
    _, _, did2 = rp.replan(traj, traj, 0, 0.0, 0.0, 0.0, 0.3, 30.0)
    assert did1 is True
    assert did2 is False   # cooldown blocks second replan

# ── BYPASSING state ────────────────────────────────────────────────────────

def test_bypassing_state_blocks_new_replan():
    rp        = PathReplanner(replan_cooldown=0.0)
    rp._state = 'BYPASSING'
    rp._bypass_start_time = float('inf')   # prevent timeout
    traj      = _straight_traj()
    scan      = _obstacle_scan(obs_dist=1.5)
    rp.update_scan(scan, (0.0, 0.0, 0.0))
    _, _, did = rp.replan(traj, traj, 0, 0.0, 0.0, 0.0, 0.3, 30.0)
    assert did is False

def test_bypass_complete_when_wp2_reached():
    rp             = PathReplanner(replan_cooldown=0.0)
    rp._state      = 'BYPASSING'
    rp._bypass_wp2 = (1.0, 0.5)
    rp._bypass_start_time = float('inf')
    traj           = _straight_traj()
    scan           = _clear_scan()
    rp.update_scan(scan, (1.0, 0.5, 0.0))
    # Robot is at wp2 — should exit BYPASSING
    rp.replan(traj, traj, 0, 1.0, 0.5, 0.0, 0.3, 30.0)
    assert rp._state == 'NORMAL'

def test_bypass_timeout_resets_to_normal():
    import time
    rp                   = PathReplanner(bypass_timeout=0.001)
    rp._state            = 'BYPASSING'
    rp._bypass_wp2       = (100.0, 100.0)   # far away — won't be reached
    rp._bypass_start_time = time.time() - 1.0   # already expired
    traj                 = _straight_traj()
    scan                 = _clear_scan()
    rp.update_scan(scan, (0.0, 0.0, 0.0))
    rp.replan(traj, traj, 0, 0.0, 0.0, 0.0, 0.3, 30.0)
    assert rp._state == 'NORMAL'
