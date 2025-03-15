import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.integrate import solve_ivp

# -----------------------------
# Simulation Parameters
# -----------------------------
# Lorenz system parameters
sigma = 10.0  # Prandtl number
rho = 28.0    # Rayleigh number
beta = 8.0/3  # Geometric factor

# Initial conditions
x0, y0, z0 = 0.1, 0.0, 0.0

# Time parameters
t_span = (0, 50)
t_eval = np.linspace(0, 50, 5000)

# Visualization parameters
trail_length = 1000  # Number of points to show in the trail
color_by_speed = True  # Whether to color the trail by speed

# -----------------------------
# Physics Simulation Functions
# -----------------------------
def lorenz_system(t, state, sigma, rho, beta):
    """Compute derivatives for the Lorenz system."""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def solve_lorenz(sigma, rho, beta, x0, y0, z0, t_span, t_eval):
    """Solve the Lorenz system with the given parameters."""
    # Initial state
    y0 = [x0, y0, z0]
    
    # Solve the ODE
    sol = solve_ivp(
        lorenz_system, 
        t_span, 
        y0, 
        args=(sigma, rho, beta),
        t_eval=t_eval, 
        method='RK45'
    )
    
    # Extract solution
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]
    t = sol.t
    
    return t, x, y, z

def calculate_fixed_points(sigma, rho, beta):
    """Calculate the fixed points of the Lorenz system."""
    # For the Lorenz system, the fixed points are:
    # 1. Origin (0, 0, 0) if rho <= 1
    # 2. Two symmetric points if rho > 1
    fixed_points = []
    
    # Origin is always a fixed point
    fixed_points.append((0, 0, 0))
    
    # If rho > 1, there are two additional fixed points
    if rho > 1:
        c = np.sqrt(beta * (rho - 1))
        fixed_points.append((c, c, rho - 1))
        fixed_points.append((-c, -c, rho - 1))
    
    return fixed_points

# Initial solution
t, x, y, z = solve_lorenz(sigma, rho, beta, x0, y0, z0, t_span, t_eval)

# -----------------------------
# Set Up the Figure and Plot
# -----------------------------
plt.style.use('dark_background')  # Use dark theme for better contrast
fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor('#1C1C1C')  # Dark background

# Create a grid for the plots - main 3D plot and info panel
gs = plt.GridSpec(1, 2, width_ratios=[3, 1])

# Main 3D plot for the Lorenz attractor
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.set_facecolor('#1C1C1C')
ax1.set_title('Lorenz Attractor Simulation', color='white', fontsize=14)
ax1.set_xlabel('X', color='white')
ax1.set_ylabel('Y', color='white')
ax1.set_zlabel('Z', color='white')

# Information panel
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#1C1C1C')
ax2.axis('off')

# Adjust layout
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, wspace=0.3)

# -----------------------------
# Initialize Plot Elements
# -----------------------------
# Set axis limits based on the solution
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), max(y)
z_min, z_max = min(z), max(z)

margin = 0.1  # Add a margin
x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min

ax1.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
ax1.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
ax1.set_zlim(z_min - margin * z_range, z_max + margin * z_range)

# 3D trajectory
lorenz_line, = ax1.plot([], [], [], '-', lw=1, alpha=0.8, color='cyan')
lorenz_point, = ax1.plot([], [], [], 'o', markersize=6, color='white')

# Fixed points
fixed_points = calculate_fixed_points(sigma, rho, beta)
fixed_point_markers = []
for fp in fixed_points:
    fp_marker, = ax1.plot([fp[0]], [fp[1]], [fp[2]], 'o', markersize=8, color='red')
    fixed_point_markers.append(fp_marker)

# Information text
info_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes,
                    verticalalignment='top', fontsize=10, color='white',
                    linespacing=1.5)

# -----------------------------
# Interactive Widgets
# -----------------------------
slider_color = '#4A90E2'  # Nice blue color for sliders
text_color = 'white'
slider_kwargs = dict(
    color=slider_color,
    track_color='#2C2C2C',
    handle_style={'facecolor': slider_color, 'edgecolor': 'white', 'size': 10}
)

# Slider for sigma (Prandtl number)
ax_sigma = plt.axes([0.1, 0.25, 0.8, 0.03])
sigma_slider = Slider(ax_sigma, "σ (Prandtl)", 0.1, 20.0, valinit=sigma, valstep=0.1, **slider_kwargs)
sigma_slider.label.set_color(text_color)

# Slider for rho (Rayleigh number)
ax_rho = plt.axes([0.1, 0.20, 0.8, 0.03])
rho_slider = Slider(ax_rho, "ρ (Rayleigh)", 0.1, 50.0, valinit=rho, valstep=0.1, **slider_kwargs)
rho_slider.label.set_color(text_color)

# Slider for beta (geometric factor)
ax_beta = plt.axes([0.1, 0.15, 0.8, 0.03])
beta_slider = Slider(ax_beta, "β (Geometric)", 0.1, 10.0, valinit=beta, valstep=0.1, **slider_kwargs)
beta_slider.label.set_color(text_color)

# Slider for initial conditions
ax_init = plt.axes([0.1, 0.10, 0.8, 0.03])
init_slider = Slider(ax_init, "Initial Condition", -10.0, 10.0, valinit=x0, valstep=0.1, **slider_kwargs)
init_slider.label.set_color(text_color)

# Reset button
ax_reset = plt.axes([0.45, 0.03, 0.1, 0.04])
reset_button = Button(ax_reset, 'Reset', color='#2C2C2C', hovercolor='#4A90E2')
reset_button.label.set_color(text_color)

# -----------------------------
# Animation Functions
# -----------------------------
def init():
    """Initialize the animation."""
    lorenz_line.set_data([], [])
    lorenz_line.set_3d_properties([])
    lorenz_point.set_data([], [])
    lorenz_point.set_3d_properties([])
    
    return lorenz_line, lorenz_point

def animate(frame):
    """Update the animation for each frame."""
    global t, x, y, z, trail_length, color_by_speed
    
    # Get current frame index (loop if we reach the end)
    idx = frame % len(t)
    
    # Calculate the start index for the trail
    start_idx = max(0, idx - trail_length)
    
    # Update 3D trajectory
    if color_by_speed and idx > 0 and start_idx < idx:
        try:
            # Calculate speed for coloring
            dx = np.diff(x[start_idx:idx+1])
            dy = np.diff(y[start_idx:idx+1])
            dz = np.diff(z[start_idx:idx+1])
            speeds = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Normalize speeds for coloring
            if len(speeds) > 0:
                # Use a simple color gradient
                speed_avg = np.mean(speeds)
                normalized_speed = min(1.0, max(0.0, speed_avg / 5.0))  # Scale to 0-1 range
                color = plt.cm.plasma(normalized_speed)
                lorenz_line.set_color(color)
        except (ValueError, RuntimeWarning):
            # If there's any issue with the calculation, use default color
            lorenz_line.set_color('cyan')
    else:
        lorenz_line.set_color('cyan')
    
    lorenz_line.set_data(x[start_idx:idx+1], y[start_idx:idx+1])
    lorenz_line.set_3d_properties(z[start_idx:idx+1])
    
    lorenz_point.set_data([x[idx]], [y[idx]])
    lorenz_point.set_3d_properties([z[idx]])
    
    # Update information text
    info_text.set_text(
        f"Lorenz System Parameters:\n"
        f"σ (Prandtl) = {sigma:.1f}\n"
        f"ρ (Rayleigh) = {rho:.1f}\n"
        f"β (Geometric) = {beta:.2f}\n\n"
        f"Current State:\n"
        f"X = {x[idx]:.2f}\n"
        f"Y = {y[idx]:.2f}\n"
        f"Z = {z[idx]:.2f}\n"
        f"Time = {t[idx]:.2f}\n\n"
        f"Fixed Points: {len(fixed_points)}\n"
        f"• Origin (0, 0, 0)\n"
        + (f"• C1 ({fixed_points[1][0]:.1f}, {fixed_points[1][1]:.1f}, {fixed_points[1][2]:.1f})\n"
           f"• C2 ({fixed_points[2][0]:.1f}, {fixed_points[2][1]:.1f}, {fixed_points[2][2]:.1f})" 
           if len(fixed_points) > 1 else "")
    )
    
    # Rotate the view for a dynamic perspective
    ax1.view_init(elev=30, azim=frame % 360)
    
    return lorenz_line, lorenz_point, info_text

# -----------------------------
# Update Functions for Widgets
# -----------------------------
def update_parameters(val=None):
    """Update the simulation with new parameters."""
    global sigma, rho, beta, x0, y0, z0, t, x, y, z, fixed_points, fixed_point_markers
    
    # Get current values from sliders
    sigma = sigma_slider.val
    rho = rho_slider.val
    beta = beta_slider.val
    x0 = init_slider.val
    y0 = init_slider.val  # Use same value for simplicity
    z0 = init_slider.val  # Use same value for simplicity
    
    # Solve the system with new parameters
    t, x, y, z = solve_lorenz(sigma, rho, beta, x0, y0, z0, t_span, t_eval)
    
    # Update axis limits
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    z_min, z_max = min(z), max(z)
    
    margin = 0.1  # Add a margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax1.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax1.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax1.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
    
    # Update fixed points
    # Remove old fixed point markers
    for marker in fixed_point_markers:
        marker.remove()
    
    fixed_point_markers = []
    fixed_points = calculate_fixed_points(sigma, rho, beta)
    for fp in fixed_points:
        fp_marker, = ax1.plot([fp[0]], [fp[1]], [fp[2]], 'o', markersize=8, color='red')
        fixed_point_markers.append(fp_marker)
    
    # Redraw the figure
    fig.canvas.draw_idle()

def reset(event):
    """Reset all parameters to their initial values."""
    sigma_slider.reset()
    rho_slider.reset()
    beta_slider.reset()
    init_slider.reset()
    update_parameters()

# Connect callbacks
sigma_slider.on_changed(update_parameters)
rho_slider.on_changed(update_parameters)
beta_slider.on_changed(update_parameters)
init_slider.on_changed(update_parameters)
reset_button.on_clicked(reset)

# Create animation
ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True, init_func=init)

plt.show() 