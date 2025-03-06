import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import matplotlib as mpl

# Set up the figure with a 3D subplot
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.style.use('dark_background')

# Create a figure with 2x2 grid for different simulations
fig = plt.figure(figsize=(14, 4), dpi=300)
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

# Add title to the figure
# fig.suptitle('https://github.com/matthewjdoyle/physics-visualisations', fontsize=22, fontweight='bold', y=0.98)

# Set up the 4 subplots for different physics simulations
ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # Double pendulum
ax2 = fig.add_subplot(gs[0, 1], projection='3d')  # N-body simulation
ax3 = fig.add_subplot(gs[0, 2], projection='3d')  # Wave equation
ax4 = fig.add_subplot(gs[0, 3], projection='3d')  # Lorenz attractor

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Set background color for all subplots
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set pane edge colors to be transparent
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)
    
    # Remove axis lines and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Make the axis lines invisible
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # Hide the spines
    ax.xaxis._axinfo['axisline']['linewidth'] = 0
    ax.yaxis._axinfo['axisline']['linewidth'] = 0
    ax.zaxis._axinfo['axisline']['linewidth'] = 0
    ax.xaxis._axinfo['grid']['linewidth'] = 0
    ax.yaxis._axinfo['grid']['linewidth'] = 0
    ax.zaxis._axinfo['grid']['linewidth'] = 0

# Add titles to each subplot
# ax1.set_title('Double Pendulum', fontsize=14, pad=10)
# ax2.set_title('N-Body Simulation', fontsize=14, pad=10)
# ax3.set_title('Wave Propagation', fontsize=14, pad=10)
# ax4.set_title('Lorenz Attractor', fontsize=14, pad=10)

# 1. Double Pendulum Simulation
def double_pendulum_deriv(t, y, L1, L2, m1, m2, g):
    """Return the derivatives of the double pendulum system."""
    theta1, omega1, theta2, omega2 = y
    
    c = np.cos(theta1 - theta2)
    s = np.sin(theta1 - theta2)
    
    theta1_dot = omega1
    theta2_dot = omega2
    
    omega1_dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * omega1**2 * c + L2 * omega2**2) -
                 (m1 + m2) * g * np.sin(theta1)) / (L1 * (m1 + m2 * s**2))
    
    omega2_dot = ((m1 + m2) * (L1 * omega1**2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
                 m2 * L2 * omega2**2 * s * c) / (L2 * (m1 + m2 * s**2))
    
    return [theta1_dot, omega1_dot, theta2_dot, omega2_dot]

# Parameters for double pendulum
L1, L2 = 1.0, 1.0  # lengths of rods
m1, m2 = 1.0, 1.0  # masses
g = 9.81  # gravitational acceleration

# Initial conditions: [theta1, omega1, theta2, omega2]
y0_dp = [np.pi/2, 0, np.pi/2, 0]

# Time points
t_span = (0, 10)
t_eval = np.linspace(0, 10, 300)

# Solve the ODE
sol_dp = solve_ivp(double_pendulum_deriv, t_span, y0_dp, args=(L1, L2, m1, m2, g),
                  t_eval=t_eval, method='RK45')

# Extract solution
theta1 = sol_dp.y[0]
theta2 = sol_dp.y[2]

# Convert to Cartesian coordinates
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
z1 = np.zeros_like(x1)

x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)
z2 = np.zeros_like(x2)

# Initialize double pendulum plot
dp_line, = ax1.plot([], [], [], 'o-', lw=2, markersize=8, color='cyan')
dp_trace, = ax1.plot([], [], [], '-', lw=1, alpha=0.3, color='cyan')
dp_trace_x, dp_trace_y, dp_trace_z = [], [], []

# Set axis limits for double pendulum
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_zlim(-0.5, 0.5)
ax1.set_box_aspect([1, 1, 0.2])

# 2. N-Body Simulation
class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.trajectory = [position]

# Initialize bodies for N-body simulation
bodies = [
    Body(1.0, [0, 0, 0], [0, 0, 0], 'yellow'),  # Sun
    Body(0.1, [1, 0, 0], [0, 1.0, 0.2], 'lightblue'),  # Planet 1
    Body(0.05, [0, 1.5, 0], [-0.8, 0, 0.1], 'white'),  # Planet 2
    Body(0.02, [0, -1.2, 0.5], [0.9, 0, 0], 'red'),  # Planet 3
    Body(0.03, [-1.8, 0, -0.2], [0, -0.7, 0], 'lightgreen')  # Planet 4
]

G = 1.0  # Gravitational constant for simulation

def compute_acceleration(bodies, body_idx):
    """Compute acceleration for a body due to gravitational forces from other bodies."""
    body = bodies[body_idx]
    acceleration = np.zeros(3)
    
    for i, other in enumerate(bodies):
        if i != body_idx:
            r = other.position - body.position
            distance = np.linalg.norm(r)
            if distance > 0.1:  # Avoid extremely close encounters
                acceleration += G * other.mass * r / (distance ** 3)
    
    return acceleration

def update_nbody(bodies, dt=0.01):
    """Update positions and velocities of all bodies."""
    accelerations = [compute_acceleration(bodies, i) for i in range(len(bodies))]
    
    for i, body in enumerate(bodies):
        body.velocity += accelerations[i] * dt
        body.position += body.velocity * dt
        body.trajectory.append(body.position.copy())
        
        # Keep only the last 50 positions for the trajectory
        if len(body.trajectory) > 50:
            body.trajectory.pop(0)

# Initialize N-body plot
nbody_points = []
nbody_trails = []

for body in bodies:
    point, = ax2.plot([], [], [], 'o', markersize=30*body.mass, color=body.color)
    trail, = ax2.plot([], [], [], '-', linewidth=3, alpha=0.6, color=body.color)
    nbody_points.append(point)
    nbody_trails.append(trail)

# Set axis limits for N-body simulation
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_zlim(-1.5, 1.5)
ax2.set_box_aspect([1, 1, 1])

# 3. Wave Equation Simulation
# Set up the mesh
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Initialize wave parameters
wave_speed = 2.0
time = 0
dt_wave = 0.05

# Initialize wave plot
wave_surf = ax3.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, 
                            linewidth=0, antialiased=True)

# Set axis limits for wave simulation
ax3.set_xlim(-3, 3)
ax3.set_ylim(-3, 3)
ax3.set_zlim(-1, 1)
ax3.set_box_aspect([1, 1, 0.3])

# 4. Lorenz Attractor
def lorenz_system(t, state, sigma, rho, beta):
    """Compute derivatives for the Lorenz system."""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Parameters for Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial condition
y0_lorenz = [0.1, 0.0, 0.0]

# Time points
t_span_lorenz = (0, 30)
t_eval_lorenz = np.linspace(0, 30, 1000)

# Solve the ODE
sol_lorenz = solve_ivp(lorenz_system, t_span_lorenz, y0_lorenz, 
                      args=(sigma, rho, beta), t_eval=t_eval_lorenz, method='RK45')

# Extract solution
x_lorenz = sol_lorenz.y[0]
y_lorenz = sol_lorenz.y[1]
z_lorenz = sol_lorenz.y[2]

# Initialize Lorenz attractor plot
lorenz_line, = ax4.plot([], [], [], '-', lw=1, color='magenta')
lorenz_point, = ax4.plot([], [], [], 'o', markersize=6, color='white')

# Set axis limits for Lorenz attractor
ax4.set_xlim(-25, 25)
ax4.set_ylim(-35, 35)
ax4.set_zlim(0, 50)
ax4.set_box_aspect([1, 1.4, 1])

# Animation function
def animate(frame):
    # 1. Update Double Pendulum
    dp_idx = frame % len(x1)
    dp_line.set_data_3d([0, x1[dp_idx], x2[dp_idx]], [0, y1[dp_idx], y2[dp_idx]], [0, z1[dp_idx], z2[dp_idx]])
    
    # Add to trace
    dp_trace_x.append(x2[dp_idx])
    dp_trace_y.append(y2[dp_idx])
    dp_trace_z.append(z2[dp_idx])
    
    # Keep only the last 50 points for the trace
    if len(dp_trace_x) > 50:
        dp_trace_x.pop(0)
        dp_trace_y.pop(0)
        dp_trace_z.pop(0)
    
    dp_trace.set_data_3d(dp_trace_x, dp_trace_y, dp_trace_z)
    
    # 2. Update N-Body Simulation
    # Update positions for 5 steps per frame for smoother motion
    for _ in range(5):
        update_nbody(bodies)
    
    # Update plots
    for i, (point, trail, body) in enumerate(zip(nbody_points, nbody_trails, bodies)):
        point.set_data_3d([body.position[0]], [body.position[1]], [body.position[2]])
        
        # Update trail
        if body.trajectory:
            trail_x = [pos[0] for pos in body.trajectory]
            trail_y = [pos[1] for pos in body.trajectory]
            trail_z = [pos[2] for pos in body.trajectory]
            trail.set_data_3d(trail_x, trail_y, trail_z)
    
    # 3. Update Wave Simulation
    global time
    time += dt_wave
    
    # Create a wave pattern
    Z = 0.5 * np.sin(np.sqrt(X**2 + Y**2) - wave_speed * time)
    Z += 0.3 * np.sin(np.sqrt((X-1)**2 + (Y+1)**2) - wave_speed * time * 1.5)
    
    # Remove old surface and create a new one
    for coll in ax3.collections:
        coll.remove()
    wave_surf = ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                linewidth=0, antialiased=True)
    
    # 4. Update Lorenz Attractor
    lorenz_idx = (frame * 3) % len(x_lorenz)
    end_idx = min(lorenz_idx + 100, len(x_lorenz))
    lorenz_line.set_data_3d(x_lorenz[lorenz_idx:end_idx], 
                           y_lorenz[lorenz_idx:end_idx], 
                           z_lorenz[lorenz_idx:end_idx])
    lorenz_point.set_data_3d([x_lorenz[lorenz_idx]], 
                            [y_lorenz[lorenz_idx]], 
                            [z_lorenz[lorenz_idx]])
    
    # Rotate the view for each subplot
    ax1.view_init(elev=-70, azim=frame % 360)
    ax2.view_init(elev=20, azim=(frame % 360) / 2)
    ax3.view_init(elev=50, azim=frame % 360)
    ax4.view_init(elev=30, azim=(frame % 360) / 3)
    
    return (dp_line, dp_trace, *nbody_points, *nbody_trails, 
            wave_surf, lorenz_line, lorenz_point)

# Create animation
frames = 300  # 10 seconds at 30 fps
ani = FuncAnimation(fig, animate, frames=frames, interval=33.33, blit=False)

# Set up the writer
writer = FFMpegWriter(fps=30, metadata=dict(artist='Physics Visualizations'), 
                     bitrate=5000)

# Save the animation
print("Rendering animation... This may take a while.")
ani.save('physics_banner.mp4', writer=writer)
print("Animation saved as 'physics_banner.mp4'")

# Uncomment to display the animation in a notebook
# from IPython.display import HTML
# HTML(ani.to_html5_video())
