import numpy as np
import matplotlib.pyplot as plt
from trajectory_tracking.path_smoother import smooth_path
from trajectory_tracking.trajectory_generator import generate_trajectory
from trajectory_tracking.pure_pursuit import PurePursuit

waypoints = [(0.0,0.0),(2.0,1.0),(4.0,0.0),(6.0,1.0),(8.0,0.0)]
smoothed = smooth_path(waypoints, num_points=200)
traj = generate_trajectory(smoothed, total_time=60.0, max_vel=0.3)

pp = PurePursuit(lookahead=0.5, max_lin_vel=0.3)
pp.set_trajectory(traj)

# Simulate robot
x, y, theta = 0.0, 0.0, 0.0
dt = 0.05
xte_log, t_log, robot_path = [], [], []

for step in range(1200):
    cmd = pp.compute_cmd(x, y, theta)
    x += cmd.linear.x * np.cos(theta) * dt
    y += cmd.linear.x * np.sin(theta) * dt
    theta += cmd.angular.z * dt
    robot_path.append((x, y))
    
    # XTE calculation
    idx = pp._current_idx
    min_d = float('inf')
    for i in range(max(0, idx-1), min(idx+5, len(traj)-1)):
        x1,y1,_ = traj[i]; x2,y2,_ = traj[i+1]
        dx,dy = x2-x1, y2-y1
        sq = dx*dx+dy*dy
        if sq < 1e-9: continue
        t_val = max(0,min(1,((x-x1)*dx+(y-y1)*dy)/sq))
        min_d = min(min_d, np.hypot(x-(x1+t_val*dx), y-(y1+t_val*dy)))
    xte_log.append(min_d)
    t_log.append(step * dt)
    if np.hypot(x - traj[-1][0], y - traj[-1][1]) < 0.1:
        break

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Path tracking
tx = [p[0] for p in traj]; ty = [p[1] for p in traj]
rx = [p[0] for p in robot_path]; ry = [p[1] for p in robot_path]
ax1.plot(tx, ty, 'b--', label='Planned trajectory', linewidth=2)
ax1.plot(rx, ry, 'r-', label='Robot path', linewidth=1.5)
ax1.scatter(*zip(*waypoints), c='green', zorder=5, label='Waypoints', s=80)
ax1.set_title('Path Tracking'); ax1.legend(); ax1.set_aspect('equal'); ax1.grid(True)

# XTE
ax2.plot(t_log, xte_log, 'purple', linewidth=1.5)
ax2.axhline(y=np.mean(xte_log), color='red', linestyle='--', label=f'Mean XTE: {np.mean(xte_log):.3f}m')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Cross-Track Error (m)')
ax2.set_title('Cross-Track Error over Time'); ax2.legend(); ax2.grid(True)
print("=== Tracking Performance Summary ===")
print(f"Total steps simulated:     {len(xte_log)}")
print(f"Total time simulated:      {t_log[-1]:.2f}s")
print(f"Final position:            ({x:.3f}, {y:.3f})")
print(f"Final goal:                ({traj[-1][0]:.3f}, {traj[-1][1]:.3f})")
print(f"Final distance to goal:    {np.hypot(x - traj[-1][0], y - traj[-1][1]):.4f}m")
print(f"")
print(f"Cross-Track Error Stats:")
print(f"  Mean XTE:                {np.mean(xte_log):.4f}m")
print(f"  Max XTE:                 {np.max(xte_log):.4f}m")
print(f"  Std Dev XTE:             {np.std(xte_log):.4f}m")
print(f"  % time under 0.05m:      {np.mean(np.array(xte_log) < 0.05)*100:.1f}%")
print(f"  % time under 0.10m:      {np.mean(np.array(xte_log) < 0.10)*100:.1f}%")
print(f"  % time under 0.20m:      {np.mean(np.array(xte_log) < 0.20)*100:.1f}%")
print(f"")
print(f"Velocity Stats:")
print(f"  Lookahead distance:      0.5m")
print(f"  Max linear vel:          0.3 m/s")
print(f"  Control frequency:       20 Hz (dt=0.05s)")
print(f"=====================================")

plt.tight_layout()
plt.savefig('tracking_performance.png', dpi=150)
