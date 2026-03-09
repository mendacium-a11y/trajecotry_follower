# Trajectory Tracking & Path Smoothing for Differential Drive Robots

This repository contains a ROS 2 package implementing a Path Smoothing and Trajectory Tracking action server for a differential drive robot (TurtleBot 4). It accepts discrete 2D waypoints, generates a smooth time-parameterized trajectory avoiding sudden sharp turns, and tracks it using a Pure Pursuit controller. 

## Table of Contents
1. [Setup and Execution](#setup-and-execution)
2. [Design Choices and Architecture](#design-choices-and-architecture)
3. [Extending to a Real Robot](#extending-to-a-real-robot)
4. [Additional Implementation: Obstacle Avoidance](#additional-implementation-obstacle-avoidance)
5. [AI Tools Used](#ai-tools-used)

---

## Setup and Execution

### Prerequisites
- ROS 2 Humble
- `turtlebot4_simulator` package
- Python 3 with `numpy` and `scipy`

### Installation
1. Clone this package into the `src` folder of your ROS 2 workspace:
   ```bash
   cd ~/tbot4_ws/src
   git clone <repository_url> trajectory_tracking
   ```
2. Install Python dependencies:
   ```bash
   pip install scipy numpy
   ```
3. Build the workspace:
   ```bash
   cd ~/tbot4_ws
   colcon build --packages-select trajectory_tracking
   source install/setup.bash
   ```

### Execution

1. **Launch the Simulation:**
   ```bash
   ros2 launch turtlebot4_ignition_bringup ignition.launch.py
   ```

2. **Run the Action Server:**
   ```bash
   ros2 launch trajectory_tracking tracking.launch.py
   ```
   Or directly:
   ```bash
   ros2 run trajectory_tracking follow_trajectory_server
   ```

3. **Send a Goal:**
   ```bash
   ros2 action send_goal /follow_trajectory turtlebot4_msgs/action/FollowTrajectory \
     "{waypoints: [
       {x: 0.0, y: 0.0, theta: 0.0},
       {x: 2.0, y: 1.0, theta: 0.0},
       {x: 4.0, y: 0.0, theta: 0.0},
       {x: 6.0, y: 1.0, theta: 0.0},
       {x: 8.0, y: 0.0, theta: 0.0}
     ]}"
   ```
   > Note: Only the `x` and `y` fields are used. `theta` is accepted by the message type but ignored — heading is derived from the path direction.

4. **Visualization in RViz:**
   Add the following topics to see planning in real-time:
   - `/input_waypoints` — `visualization_msgs/MarkerArray` (green=start, red=goal, cyan=intermediate)
   - `/trajectory_path` — `nav_msgs/Path` (original planned path)
   - `/replanned_trajectory` — `nav_msgs/Path` (appears when bypassing an obstacle)

### Running Tests
```bash
colcon test --packages-select trajectory_tracking
colcon test-result --verbose
```
47 tests covering path smoothing output shape and edge cases, trajectory generation timing properties, Pure Pursuit velocity limits and steering direction, replanner 
state machine transitions and cooldown logic, and full pipeline integration.

**Simulation Results (offline kinematic simulation):**
The controller was validated offline over a sinusoidal 8-metre waypoint path at 0.3 m/s:
- Mean cross-track error: **0.025m** (2.5cm)
- Max cross-track error: **0.184m** at sharpest curve apex
- 87.5% of timesteps within 5cm of planned path
- Final goal error: **0.099m** (within 0.25m tolerance)

To reproduce: `python3 scripts/plot.py` (from workspace root with `source install/setup.bash`)

![plot](tracking_performance.png)
---

## Design Choices and Architecture

The system is modular, split into specialized Python modules orchestrated by the main Action Server.

### 1. Path Smoothing (`path_smoother.py`)
- **Algorithm:** B-Spline interpolation via `scipy.interpolate.splprep`
- **Why:** Discrete waypoints contain sharp corners that force a differential drive robot to decelerate or stop to turn in place. B-Splines guarantee C² continuity (smooth curvature), allowing the robot to maintain translational velocity through curves.
- **Key detail:** Uses `s = n * 0.01` (adaptive smoothing factor) rather than `s=0.0` (exact interpolation). Exact interpolation causes Runge-style oscillations when waypoints are closely spaced — which happens every time the replanner injects bypass waypoints that can be under 0.5m apart. Duplicate and densely clustered points are filtered out before fitting.
- **Fallback:** Linear interpolation when fewer than 4 unique points remain after deduplication.

### 2. Trajectory Generation (`trajectory_generator.py`)
- **Algorithm:** Trapezoidal velocity profile mapping arc-length distances to timestamps
- **Why:** Provides a realistic kinematic profile — bounded acceleration and deceleration — so wheel motors are not commanded to instantaneously jump to full speed. Given `total_time` and `max_vel`, it computes optimal acceleration, cruise, and deceleration phases. The output `[(x, y, t), ...]` gives the Pure Pursuit controller a time-reference to measure progress against.
- **Short path handling:** If the path is too short to reach `max_vel`, the profile degrades gracefully to a pure triangular (accel/decel only) profile.

### 3. Trajectory Tracking Controller (`pure_pursuit.py`)
- **Algorithm:** Pure Pursuit
- **Why:** Computes steering commands by chasing a lookahead point ahead of the robot on the path rather than enforcing strict timestamp following. This makes it naturally robust to odometry drift and control lag — if the robot falls behind, it simply looks further ahead on the path and corrects. A PID controller on cross-track error would accumulate integral windup under the same conditions.
- **Lookahead tuning:** Shorter lookahead = tighter path tracking but higher oscillation risk. Longer = smoother driving but cuts corners on tight curves. Default `0.5m` is a good balance for TurtleBot 4 at `0.3 m/s`.
- **Near-goal behavior:** When no lookahead point is found ahead (robot is near the end), the controller steers directly toward the final point at reduced speed proportional to remaining distance for a smooth stop.

- **Note:**Pure Pursuit is a geometric controller by design — it tracks a spatial path via lookahead, not a time-indexed reference. The trapezoidal timestamps are retained in the tuple for external monitoring and cross-track error computation. A time-indexed controller like PID requires tight timing synchronization and accumulates integral windup under execution delays; Pure Pursuit's spatial lookahead is inherently robust to those same delay

### 4. Action Server Architecture (`follow_trajectory_action_server.py`)
- **Action Server paradigm:** Allows any client to track progress (`progress_pct`, distance to goal) and cancel mid-flight without killing the node.
- **Concurrency:** Uses `MultiThreadedExecutor` with `ReentrantCallbackGroup` so `/odom` and `/scan` callbacks are processed concurrently without blocking the 20Hz control loop.
- **Snapshot pattern:** The scan and robot pose are captured atomically once per control cycle and passed to both `emergency_stop_needed()` and `replan()`, guaranteeing both checks operate on identical sensor data within a single cycle.

---

## Extending to a Real Robot

Transitioning from TurtleBot 4 simulation to a physical differential-drive robot requires addressing several real-world imperfections:

1. **Odometry Drift:**
   Simulation `/odom` is near-perfect. On a real robot, wheel slip and encoder noise accumulate drift rapidly.
   **Solution:** Run a localization stack — AMCL with a pre-built map, or an EKF fusing IMU and wheel encoders (e.g., `robot_localization` package). The Pure Pursuit controller works over any reliable pose source as long as TF frames are consistent.

2. **Actuator Limits and Control Lag:**
   Real motors have inertia and cannot track instantaneous velocity step changes.
   **Solution:** Pass `/cmd_vel` through a velocity smoother node that enforces physical jerk limits before commands reach the hardware driver.

3. **LiDAR Noise and False Positives:**
   Real lidar has range noise, scan shadows, and reflective surface artifacts that can trigger false replans.
   **Solution:** Apply a median filter over scan ranges before passing to the replanner, and increase `corridor_half_width` slightly to avoid triggering on single noisy returns.

4. **Communication and Hardware Watchdog:**
   If `/odom` drops (cable fault, node crash), the robot will continue on stale pose data.
   **Solution:** Add a watchdog timer on the odom callback — if no odometry is received for more than 200ms, publish a zero `Twist` and abort the goal.

5. **Global vs. Local Frame:**
   The current planner assumes all waypoints are in the `odom` frame. On a real robot with a map, waypoints should be given in the `map` frame and transformed to `odom` via TF before planning.

---

## Additional Implementation: Obstacle Avoidance

The system dynamically bypasses unexpected obstacles at runtime using a geometric local planner in `path_replanner.py`.

### How It Works

**1. State Machine**

The replanner operates as a two-state machine:
- `NORMAL` — monitoring the path corridor for obstacles
- `BYPASSING` — executing a computed bypass; new replans are blocked until the bypass completes

This prevents the oscillation that occurs when the replanner repeatedly re-triggers mid-bypass because the obstacle is still visible in the scan.

**2. Corridor Detection**

Each control cycle, scan points are projected into world frame using the pose captured atomically with the scan. Points within `corridor_half_width` of any upcoming path segment (within `lookahead_check` metres) are flagged as blocking. The closest blocking point is used as the obstacle estimate — not the mean — to avoid wall returns pulling the estimate away from the actual obstacle.

**3. Geometric Bypass Generation**

- A ray is cast perpendicular to the path direction from the obstacle centre to measure free space on each side
- Two bypass waypoints (WP1, WP2) are placed laterally offset from the obstacle — WP1 before it, WP2 after
- All three are validated: WP1 free, WP2 free, and the segment WP1→WP2 clear of obstacles
- If the preferred side is blocked, the algorithm automatically flips to the other side

**4. Trajectory Hot-Swap**

The new waypoint list `[robot_position, WP1, WP2, ...original_path_from_resume_idx]` is fed back through the B-Spline smoother and trapezoidal generator. The Pure Pursuit controller's trajectory is replaced in-flight without stopping, producing a fluid dodge manoeuvre.

**5. Emergency Stop**

If an obstacle enters the forward arc within `stop_distance` (0.35m), the robot halts immediately. Critically, the replanner still runs during the stop so a bypass is computed while the robot is stationary — the robot resumes motion as soon as a clear path is ready.

### Testing Obstacle Avoidance

Spawn a static box obstacle in Gazebo:
```bash
ros2 run ros_gz_sim create \
  -world plain \
  -name test_box \
  -x 3.75 -y 0.0 -z 0.25 \
  -string "
<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='test_box'>
    <link name='link'>
      <collision name='collision'>
        <geometry><box><size>0.5 0.5 0.5</size></box></geometry>
      </collision>
      <visual name='visual'>
        <geometry><box><size>0.5 0.5 0.5</size></box></geometry>
        <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"
```

Send a goal that routes directly through the obstacle:
```bash
ros2 action send_goal /follow_trajectory turtlebot4_msgs/action/FollowTrajectory \
  "{waypoints: [
    {x: 0.0,  y:  0.0, theta: 0.0},
    {x: 3.75, y:  0.0, theta: 0.0},
    {x: 7.5,  y:  0.0, theta: 0.0}
  ]}"
```

Watch `/replanned_trajectory` appear in RViz as the robot detects the box and computes a bypass. The feedback status field will transition from `state=NORMAL` to `state=BYPASSING` and back to `state=NORMAL` after the obstacle is cleared.

---

## AI Tools Used

This project used AI tooling throughout development:

- **Perplexity (Claude Sonnet 4.6):** Used for implementation guidance throughout the project — designing the state machine architecture for the replanner after diagnosing that the original cooldown-timer approach was causing mid-bypass re-triggering and oscillation, debugging the spline oscillation issue caused by `s=0.0` in `splprep` when bypass waypoints are densely clustered, and identifying the adaptive `s = n * 0.01` fix. Also used for ROS 2 architecture decisions, bug diagnosis across the full pipeline, and test case design.
- **Google Antigravity (AI-assisted coding):** Used for inline code generation, ROS 2 boilerplate scaffolding (publisher/subscriber setup, action server structure), and Python type hints.

All algorithmic decisions — choice of Pure Pursuit over PID, trapezoidal profiling, B-Spline smoothing, geometric bypass approach — were made and understood by me. AI was used as a debugging and implementation accelerator, not a decision-maker.
