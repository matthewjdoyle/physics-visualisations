import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib as mpl

# -----------------------------
# Simulation Parameters
# -----------------------------
L1, L2 = 1.0, 1.0  # lengths of rods (m)
m1, m2 = 1.0, 1.0  # masses (kg)
g = 9.81  # gravitational acceleration (m/s²)
damping = 0.0  # damping coefficient

# Initial conditions: [theta1, omega1, theta2, omega2]
y0 = [np.pi/2, 0, np.pi/2, 0]

# Time points
t_span = (0, 30)
t_eval = np.linspace(0, 30, 1000)

# Trace parameters
max_trace_points = 500  # Maximum number of points to keep in the trace
trace_fade = True  # Whether to fade the trace over time

# -----------------------------
# Physics Simulation Functions
# -----------------------------
def double_pendulum_deriv(t, y, L1, L2, m1, m2, g, damping):
    """Return the derivatives of the double pendulum system with optional damping."""
    theta1, omega1, theta2, omega2 = y
    
    c = np.cos(theta1 - theta2)
    s = np.sin(theta1 - theta2)
    
    theta1_dot = omega1
    theta2_dot = omega2
    
    # Add damping terms (-damping * omega)
    omega1_dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * omega1**2 * c + L2 * omega2**2) -
                 (m1 + m2) * g * np.sin(theta1) - damping * omega1) / (L1 * (m1 + m2 * s**2))
    
    omega2_dot = ((m1 + m2) * (L1 * omega1**2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
                 m2 * L2 * omega2**2 * s * c - damping * omega2) / (L2 * (m1 + m2 * s**2))
    
    return [theta1_dot, omega1_dot, theta2_dot, omega2_dot]

def calculate_energy(theta1, omega1, theta2, omega2, L1, L2, m1, m2, g):
    """Calculate the total energy (kinetic + potential) of the double pendulum system."""
    # Positions
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    # Velocities
    v1x = L1 * omega1 * np.cos(theta1)
    v1y = L1 * omega1 * np.sin(theta1)
    v2x = v1x + L2 * omega2 * np.cos(theta2)
    v2y = v1y + L2 * omega2 * np.sin(theta2)
    
    # Kinetic energy
    K1 = 0.5 * m1 * (v1x**2 + v1y**2)
    K2 = 0.5 * m2 * (v2x**2 + v2y**2)
    
    # Potential energy (zero at the pivot)
    U1 = m1 * g * (y1 + L1)  # Add L1 to make y=0 at the pivot
    U2 = m2 * g * (y2 + L1 + L2)  # Add L1+L2 to make y=0 at the pivot
    
    # Total energy
    return K1 + K2 + U1 + U2

def solve_pendulum(L1, L2, m1, m2, g, damping, y0, t_span, t_eval):
    """Solve the double pendulum ODE with the given parameters."""
    sol = solve_ivp(
        double_pendulum_deriv, 
        t_span, 
        y0, 
        args=(L1, L2, m1, m2, g, damping),
        t_eval=t_eval, 
        method='RK45'
    )
    
    # Extract solution
    theta1 = sol.y[0]
    omega1 = sol.y[1]
    theta2 = sol.y[2]
    omega2 = sol.y[3]
    
    # Convert to Cartesian coordinates
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    # Calculate energy at each time point
    energy = np.array([calculate_energy(theta1[i], omega1[i], theta2[i], omega2[i], L1, L2, m1, m2, g) 
                      for i in range(len(theta1))])
    
    return sol.t, theta1, omega1, theta2, omega2, x1, y1, x2, y2, energy

# Initial solution
t, theta1, omega1, theta2, omega2, x1, y1, x2, y2, energy = solve_pendulum(
    L1, L2, m1, m2, g, damping, y0, t_span, t_eval
)

# -----------------------------
# Set Up the Figure and Plot
# -----------------------------
plt.style.use('dark_background')  # Use dark theme for better contrast
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#1C1C1C')  # Dark background

# Create a grid for the plots
gs = plt.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])

# Main pendulum animation plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#1C1C1C')
ax1.set_aspect('equal')
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_title('Double Pendulum Simulation', color='white', fontsize=14)
ax1.grid(alpha=0.2)

# Energy plot
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#1C1C1C')
ax2.set_xlim(0, t_span[1])
ax2.set_ylim(min(energy) * 0.9, max(energy) * 1.1)
ax2.set_title('Energy over Time', color='white')
ax2.set_xlabel('Time (s)', color='white')
ax2.set_ylabel('Energy (J)', color='white')
ax2.grid(alpha=0.2)

# Phase space plot
ax3 = fig.add_subplot(gs[0, 1])
ax3.set_facecolor('#1C1C1C')
ax3.set_title('Phase Space (θ₁ vs θ₂)', color='white')
ax3.set_xlabel('θ₁ (rad)', color='white')
ax3.set_ylabel('θ₂ (rad)', color='white')
ax3.set_xlim(-np.pi, np.pi)
ax3.set_ylim(-np.pi, np.pi)
ax3.grid(alpha=0.2)

# Information panel
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#1C1C1C')
ax4.axis('off')

# Adjust layout
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, wspace=0.3, hspace=0.3)

# -----------------------------
# Initialize Plot Elements
# -----------------------------
# Pendulum rods and masses
pendulum_line, = ax1.plot([], [], 'o-', lw=2, markersize=10, color='cyan')
trace, = ax1.plot([], [], '-', lw=1, alpha=0.5, color='cyan')
trace_x, trace_y = [], []

# Energy plot
energy_line, = ax2.plot([], [], lw=2, color='#4A90E2')
energy_point, = ax2.plot([], [], 'o', markersize=6, color='white')

# Phase space plot
phase_line, = ax3.plot([], [], '-', lw=1, alpha=0.5, color='magenta')
phase_point, = ax3.plot([], [], 'o', markersize=6, color='white')
phase_x, phase_y = [], []

# Information text
info_text = ax4.text(0.05, 0.95, '', transform=ax4.transAxes,
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

# Sliders for pendulum parameters
ax_L1 = plt.axes([0.1, 0.25, 0.35, 0.03])
L1_slider = Slider(ax_L1, "Length 1 (m)", 0.5, 2.0, valinit=L1, valstep=0.1, **slider_kwargs)
L1_slider.label.set_color(text_color)

ax_L2 = plt.axes([0.1, 0.20, 0.35, 0.03])
L2_slider = Slider(ax_L2, "Length 2 (m)", 0.5, 2.0, valinit=L2, valstep=0.1, **slider_kwargs)
L2_slider.label.set_color(text_color)

ax_m1 = plt.axes([0.1, 0.15, 0.35, 0.03])
m1_slider = Slider(ax_m1, "Mass 1 (kg)", 0.5, 5.0, valinit=m1, valstep=0.1, **slider_kwargs)
m1_slider.label.set_color(text_color)

ax_m2 = plt.axes([0.1, 0.10, 0.35, 0.03])
m2_slider = Slider(ax_m2, "Mass 2 (kg)", 0.5, 5.0, valinit=m2, valstep=0.1, **slider_kwargs)
m2_slider.label.set_color(text_color)

ax_g = plt.axes([0.1, 0.05, 0.35, 0.03])
g_slider = Slider(ax_g, "Gravity (m/s²)", 1.0, 20.0, valinit=g, valstep=0.1, **slider_kwargs)
g_slider.label.set_color(text_color)

ax_damping = plt.axes([0.6, 0.25, 0.35, 0.03])
damping_slider = Slider(ax_damping, "Damping", 0.0, 1.0, valinit=damping, valstep=0.01, **slider_kwargs)
damping_slider.label.set_color(text_color)

# Initial conditions sliders
ax_theta1 = plt.axes([0.6, 0.20, 0.35, 0.03])
theta1_slider = Slider(ax_theta1, "Initial θ₁ (rad)", -np.pi, np.pi, valinit=y0[0], valstep=0.1, **slider_kwargs)
theta1_slider.label.set_color(text_color)

ax_theta2 = plt.axes([0.6, 0.15, 0.35, 0.03])
theta2_slider = Slider(ax_theta2, "Initial θ₂ (rad)", -np.pi, np.pi, valinit=y0[2], valstep=0.1, **slider_kwargs)
theta2_slider.label.set_color(text_color)

ax_omega1 = plt.axes([0.6, 0.10, 0.35, 0.03])
omega1_slider = Slider(ax_omega1, "Initial ω₁ (rad/s)", -5.0, 5.0, valinit=y0[1], valstep=0.1, **slider_kwargs)
omega1_slider.label.set_color(text_color)

ax_omega2 = plt.axes([0.6, 0.05, 0.35, 0.03])
omega2_slider = Slider(ax_omega2, "Initial ω₂ (rad/s)", -5.0, 5.0, valinit=y0[3], valstep=0.1, **slider_kwargs)
omega2_slider.label.set_color(text_color)

# Reset button
ax_reset = plt.axes([0.1, 0.30, 0.1, 0.04])
reset_button = Button(ax_reset, 'Reset', color='#2C2C2C', hovercolor='#4A90E2')
reset_button.label.set_color(text_color)

# Trace options
ax_trace = plt.axes([0.25, 0.30, 0.2, 0.04])
trace_radio = RadioButtons(ax_trace, ('Trace On', 'Trace Off'), active=0)
for circle in trace_radio.circles:
    circle.set_facecolor(slider_color)
for label in trace_radio.labels:
    label.set_color(text_color)

# -----------------------------
# Animation Functions
# -----------------------------
def init():
    """Initialize the animation."""
    pendulum_line.set_data([], [])
    trace.set_data([], [])
    energy_line.set_data([], [])
    energy_point.set_data([], [])
    phase_line.set_data([], [])
    phase_point.set_data([], [])
    return pendulum_line, trace, energy_line, energy_point, phase_line, phase_point

def animate(frame):
    """Update the animation for each frame."""
    global trace_x, trace_y, phase_x, phase_y
    
    # Get current frame index (loop if we reach the end)
    idx = frame % len(t)
    
    # Update pendulum position
    pendulum_line.set_data([0, x1[idx], x2[idx]], [0, y1[idx], y2[idx]])
    
    # Update trace if enabled
    if trace_radio.value_selected == 'Trace On':
        trace_x.append(x2[idx])
        trace_y.append(y2[idx])
        
        # Keep only the last max_trace_points points
        if len(trace_x) > max_trace_points:
            trace_x.pop(0)
            trace_y.pop(0)
        
        # Apply fading effect if enabled
        if trace_fade and len(trace_x) > 1:
            # Create a line collection with varying colors
            trace.set_data(trace_x, trace_y)
            
            # Set alpha based on age of points
            alpha_values = np.linspace(0.1, 1.0, len(trace_x))
            trace.set_alpha(alpha_values[-1])  # Use the last alpha value
    else:
        trace.set_data([], [])
        trace_x, trace_y = [], []
    
    # Update energy plot
    energy_line.set_data(t[:idx+1], energy[:idx+1])
    energy_point.set_data([t[idx]], [energy[idx]])
    
    # Update phase space plot
    # Normalize angles to [-π, π]
    theta1_norm = ((theta1[idx] + np.pi) % (2 * np.pi)) - np.pi
    theta2_norm = ((theta2[idx] + np.pi) % (2 * np.pi)) - np.pi
    
    phase_x.append(theta1_norm)
    phase_y.append(theta2_norm)
    
    # Keep only the last max_trace_points points
    if len(phase_x) > max_trace_points:
        phase_x.pop(0)
        phase_y.pop(0)
    
    phase_line.set_data(phase_x, phase_y)
    phase_point.set_data([theta1_norm], [theta2_norm])
    
    # Update information text
    info_text.set_text(
        f"System Parameters:\n"
        f"L₁ = {L1:.1f} m, L₂ = {L2:.1f} m\n"
        f"m₁ = {m1:.1f} kg, m₂ = {m2:.1f} kg\n"
        f"g = {g:.1f} m/s², damping = {damping:.2f}\n\n"
        f"Current State:\n"
        f"θ₁ = {theta1[idx]:.2f} rad\n"
        f"θ₂ = {theta2[idx]:.2f} rad\n"
        f"ω₁ = {omega1[idx]:.2f} rad/s\n"
        f"ω₂ = {omega2[idx]:.2f} rad/s\n\n"
        f"Energy = {energy[idx]:.2f} J\n"
        f"Time = {t[idx]:.2f} s"
    )
    
    return pendulum_line, trace, energy_line, energy_point, phase_line, phase_point, info_text

# -----------------------------
# Update Functions for Widgets
# -----------------------------
def update_simulation(val=None):
    """Update the simulation with new parameters."""
    global L1, L2, m1, m2, g, damping, y0, t, theta1, omega1, theta2, omega2, x1, y1, x2, y2, energy
    global trace_x, trace_y, phase_x, phase_y
    
    # Get current values from sliders
    L1 = L1_slider.val
    L2 = L2_slider.val
    m1 = m1_slider.val
    m2 = m2_slider.val
    g = g_slider.val
    damping = damping_slider.val
    
    # Get initial conditions
    y0 = [
        theta1_slider.val,
        omega1_slider.val,
        theta2_slider.val,
        omega2_slider.val
    ]
    
    # Solve the system with new parameters
    t, theta1, omega1, theta2, omega2, x1, y1, x2, y2, energy = solve_pendulum(
        L1, L2, m1, m2, g, damping, y0, t_span, t_eval
    )
    
    # Clear traces
    trace_x, trace_y = [], []
    phase_x, phase_y = [], []
    
    # Update energy plot limits
    ax2.set_ylim(min(energy) * 0.9, max(energy) * 1.1)
    
    # Redraw the figure
    fig.canvas.draw_idle()

def reset(event):
    """Reset all parameters to their initial values."""
    L1_slider.reset()
    L2_slider.reset()
    m1_slider.reset()
    m2_slider.reset()
    g_slider.reset()
    damping_slider.reset()
    theta1_slider.reset()
    theta2_slider.reset()
    omega1_slider.reset()
    omega2_slider.reset()
    update_simulation()

# Connect callbacks
L1_slider.on_changed(update_simulation)
L2_slider.on_changed(update_simulation)
m1_slider.on_changed(update_simulation)
m2_slider.on_changed(update_simulation)
g_slider.on_changed(update_simulation)
damping_slider.on_changed(update_simulation)
theta1_slider.on_changed(update_simulation)
theta2_slider.on_changed(update_simulation)
omega1_slider.on_changed(update_simulation)
omega2_slider.on_changed(update_simulation)
reset_button.on_clicked(reset)
trace_radio.on_clicked(lambda label: fig.canvas.draw_idle())

# Create animation
ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True, init_func=init)

plt.show() 