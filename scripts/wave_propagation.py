import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# Simulation Parameters
# -----------------------------
# Grid parameters
grid_size = 50  # Number of points in each dimension
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
X, Y = np.meshgrid(x, y)

# Wave parameters
wave_speed = 1.0  # Wave propagation speed
damping = 0.02  # Damping coefficient
dt = 0.05  # Time step
time = 0  # Current simulation time

# Source parameters
source_type = 'point'  # 'point', 'line', or 'plane'
source_frequency = 1.0  # Frequency of the source
source_amplitude = 1.0  # Amplitude of the source
source_position = (0, 0)  # Position of the source

# Boundary conditions
boundary_type = 'absorbing'  # 'reflecting', 'absorbing', or 'periodic'

# Medium parameters
medium_type = 'uniform'  # 'uniform', 'gradient', or 'random'
refraction_index = 1.0  # Refraction index for non-uniform medium

# Initialize wave field and its time derivative
Z = np.zeros((grid_size, grid_size))  # Current wave field
Z_prev = np.zeros((grid_size, grid_size))  # Previous wave field
Z_next = np.zeros((grid_size, grid_size))  # Next wave field

# -----------------------------
# Physics Simulation Functions
# -----------------------------
def initialize_medium(medium_type, refraction_index):
    """Initialize the medium properties (wave speed at each point)."""
    if medium_type == 'uniform':
        # Uniform medium
        medium = np.ones((grid_size, grid_size)) * wave_speed
    elif medium_type == 'gradient':
        # Gradient medium (wave speed increases along x-axis)
        medium = np.ones((grid_size, grid_size)) * wave_speed
        for i in range(grid_size):
            medium[:, i] *= (0.5 + i / grid_size) * refraction_index
    elif medium_type == 'random':
        # Random medium with some structure
        medium = np.ones((grid_size, grid_size)) * wave_speed
        # Add random variations
        np.random.seed(42)  # For reproducibility
        random_var = np.random.rand(grid_size, grid_size) * 0.5 + 0.75
        # Smooth the random variations
        from scipy.ndimage import gaussian_filter
        random_var = gaussian_filter(random_var, sigma=3)
        medium *= random_var * refraction_index
    elif medium_type == 'lens':
        # Create a converging lens in the center
        medium = np.ones((grid_size, grid_size)) * wave_speed
        center_x, center_y = grid_size // 2, grid_size // 2
        radius = grid_size // 4
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < radius:
                    # Inside the lens, higher refractive index
                    medium[i, j] *= 1.0 / refraction_index
    elif medium_type == 'barrier':
        # Create a barrier that blocks waves
        medium = np.ones((grid_size, grid_size)) * wave_speed
        # Vertical barrier in the middle with a small gap
        barrier_x = grid_size // 2
        gap_start = grid_size // 3
        gap_end = 2 * grid_size // 3
        for i in range(grid_size):
            if i < gap_start or i > gap_end:
                medium[i, barrier_x] = 0.01  # Very slow wave speed (effectively blocking)
    else:
        # Default to uniform
        medium = np.ones((grid_size, grid_size)) * wave_speed
    
    return medium

def add_source(Z, source_type, source_position, source_frequency, source_amplitude, time):
    """Add a wave source to the field."""
    i_center, j_center = int((source_position[0] + 5) * grid_size / 10), int((source_position[1] + 5) * grid_size / 10)
    
    # Ensure the source position is within the grid
    i_center = max(0, min(i_center, grid_size - 1))
    j_center = max(0, min(j_center, grid_size - 1))
    
    if source_type == 'point':
        # Point source (oscillating point)
        Z[i_center, j_center] = source_amplitude * np.sin(2 * np.pi * source_frequency * time)
    elif source_type == 'line':
        # Line source (horizontal line)
        Z[i_center, :] = source_amplitude * np.sin(2 * np.pi * source_frequency * time)
    elif source_type == 'plane':
        # Plane wave from left edge
        Z[:, 0] = source_amplitude * np.sin(2 * np.pi * source_frequency * time)
    elif source_type == 'circular':
        # Circular wave (expanding circle)
        radius = (time * wave_speed) % 5  # Modulo to repeat the wave
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt(((i - i_center) / grid_size * 10)**2 + ((j - j_center) / grid_size * 10)**2)
                if abs(dist - radius) < 0.5:
                    Z[i, j] += source_amplitude * 0.5
    elif source_type == 'gaussian_pulse':
        # Gaussian pulse (single pulse that expands)
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt(((i - i_center) / grid_size * 10)**2 + ((j - j_center) / grid_size * 10)**2)
                Z[i, j] += source_amplitude * np.exp(-(dist - time * wave_speed)**2)
    
    return Z

def apply_boundary_conditions(Z, boundary_type):
    """Apply boundary conditions to the wave field."""
    if boundary_type == 'reflecting':
        # Reflecting boundaries (Dirichlet boundary conditions)
        Z[0, :] = 0
        Z[-1, :] = 0
        Z[:, 0] = 0
        Z[:, -1] = 0
    elif boundary_type == 'absorbing':
        # Absorbing boundaries (simple implementation)
        # Apply damping near the boundaries
        width = 5  # Width of the absorbing layer
        damping_factor = 0.9  # Damping factor
        
        # Apply damping to the boundaries
        for i in range(width):
            factor = damping_factor ** (width - i)
            Z[i, :] *= factor
            Z[-i-1, :] *= factor
            Z[:, i] *= factor
            Z[:, -i-1] *= factor
    elif boundary_type == 'periodic':
        # Periodic boundaries
        Z[0, :] = Z[-2, :]
        Z[-1, :] = Z[1, :]
        Z[:, 0] = Z[:, -2]
        Z[:, -1] = Z[:, 1]
    
    return Z

def update_wave_field(Z, Z_prev, medium, dt, damping):
    """Update the wave field using the wave equation."""
    # Create a copy of the current field for the next step
    Z_next = np.zeros_like(Z)
    
    # Apply the wave equation: ∂²Z/∂t² = c²∇²Z - damping*∂Z/∂t
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            if medium[i, j] > 0:  # Skip points where the medium blocks waves
                # Laplacian (∇²Z)
                laplacian = (Z[i+1, j] + Z[i-1, j] + Z[i, j+1] + Z[i, j-1] - 4 * Z[i, j]) / (dx**2)
                
                # Wave equation discretized
                # Z_next = 2*Z - Z_prev + c²*dt²*∇²Z - damping*(Z - Z_prev)
                c_squared = medium[i, j]**2
                Z_next[i, j] = 2 * Z[i, j] - Z_prev[i, j] + c_squared * dt**2 * laplacian - damping * (Z[i, j] - Z_prev[i, j])
    
    return Z_next

# -----------------------------
# Set Up the Figure and Plot
# -----------------------------
plt.style.use('dark_background')  # Use dark theme for better contrast
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#1C1C1C')  # Dark background

# Create a grid for the plots
gs = plt.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])

# Main 3D plot for the wave simulation
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.set_facecolor('#1C1C1C')
ax1.set_title('Wave Propagation Simulation', color='white', fontsize=14)
ax1.set_xlabel('X', color='white')
ax1.set_ylabel('Y', color='white')
ax1.set_zlabel('Amplitude', color='white')
ax1.set_zlim(-1.5, 1.5)

# 2D view of the wave field (top view)
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#1C1C1C')
ax2.set_title('Wave Field (Top View)', color='white')
ax2.set_xlabel('X', color='white')
ax2.set_ylabel('Y', color='white')

# Medium properties visualization
ax3 = fig.add_subplot(gs[0, 1])
ax3.set_facecolor('#1C1C1C')
ax3.set_title('Medium Properties', color='white')
ax3.set_xlabel('X', color='white')
ax3.set_ylabel('Y', color='white')

# Information panel
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#1C1C1C')
ax4.axis('off')

# Adjust layout
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, wspace=0.3, hspace=0.3)

# -----------------------------
# Initialize Simulation
# -----------------------------
# Calculate grid spacing
dx = (x[-1] - x[0]) / (grid_size - 1)

# Initialize the medium
medium = initialize_medium(medium_type, refraction_index)

# Initialize plot elements
# 3D surface plot
wave_surf = ax1.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)

# 2D color plot (top view)
wave_img = ax2.imshow(Z, cmap='plasma', origin='lower', extent=[-5, 5, -5, 5], vmin=-1, vmax=1)
plt.colorbar(wave_img, ax=ax2, label='Amplitude')

# Medium visualization
medium_img = ax3.imshow(medium, cmap='viridis', origin='lower', extent=[-5, 5, -5, 5])
plt.colorbar(medium_img, ax=ax3, label='Wave Speed')

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

# Slider for wave speed
ax_speed = plt.axes([0.1, 0.25, 0.35, 0.03])
speed_slider = Slider(ax_speed, "Wave Speed", 0.1, 3.0, valinit=wave_speed, valstep=0.1, **slider_kwargs)
speed_slider.label.set_color(text_color)

# Slider for damping
ax_damping = plt.axes([0.1, 0.20, 0.35, 0.03])
damping_slider = Slider(ax_damping, "Damping", 0.0, 0.1, valinit=damping, valstep=0.005, **slider_kwargs)
damping_slider.label.set_color(text_color)

# Slider for source frequency
ax_freq = plt.axes([0.1, 0.15, 0.35, 0.03])
freq_slider = Slider(ax_freq, "Source Frequency", 0.1, 3.0, valinit=source_frequency, valstep=0.1, **slider_kwargs)
freq_slider.label.set_color(text_color)

# Slider for source amplitude
ax_amp = plt.axes([0.1, 0.10, 0.35, 0.03])
amp_slider = Slider(ax_amp, "Source Amplitude", 0.1, 2.0, valinit=source_amplitude, valstep=0.1, **slider_kwargs)
amp_slider.label.set_color(text_color)

# Slider for refraction index
ax_refract = plt.axes([0.1, 0.05, 0.35, 0.03])
refract_slider = Slider(ax_refract, "Refraction Index", 0.5, 2.0, valinit=refraction_index, valstep=0.1, **slider_kwargs)
refract_slider.label.set_color(text_color)

# Radio buttons for source type
ax_source = plt.axes([0.6, 0.05, 0.15, 0.15])
source_radio = RadioButtons(
    ax_source, 
    ('point', 'line', 'plane', 'circular', 'gaussian_pulse'),
    active=0
)
for circle in source_radio.circles:
    circle.set_facecolor(slider_color)
for label in source_radio.labels:
    label.set_color(text_color)

# Radio buttons for boundary type
ax_boundary = plt.axes([0.8, 0.05, 0.15, 0.1])
boundary_radio = RadioButtons(
    ax_boundary, 
    ('reflecting', 'absorbing', 'periodic'),
    active=1
)
for circle in boundary_radio.circles:
    circle.set_facecolor(slider_color)
for label in boundary_radio.labels:
    label.set_color(text_color)

# Radio buttons for medium type
ax_medium = plt.axes([0.8, 0.2, 0.15, 0.15])
medium_radio = RadioButtons(
    ax_medium, 
    ('uniform', 'gradient', 'random', 'lens', 'barrier'),
    active=0
)
for circle in medium_radio.circles:
    circle.set_facecolor(slider_color)
for label in medium_radio.labels:
    label.set_color(text_color)

# Reset button
ax_reset = plt.axes([0.6, 0.25, 0.1, 0.04])
reset_button = Button(ax_reset, 'Reset', color='#2C2C2C', hovercolor='#4A90E2')
reset_button.label.set_color(text_color)

# -----------------------------
# Animation Functions
# -----------------------------
def init():
    """Initialize the animation."""
    # Clear the surface plot
    ax1.clear()
    ax1.set_facecolor('#1C1C1C')
    ax1.set_title('Wave Propagation Simulation', color='white', fontsize=14)
    ax1.set_xlabel('X', color='white')
    ax1.set_ylabel('Y', color='white')
    ax1.set_zlabel('Amplitude', color='white')
    ax1.set_zlim(-1.5, 1.5)
    
    # Create a new surface plot
    wave_surf = ax1.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)
    
    # Update the 2D image
    wave_img.set_array(Z)
    
    return [wave_surf, wave_img]

def animate(frame):
    """Update the animation for each frame."""
    global Z, Z_prev, Z_next, time, medium
    global wave_speed, damping, source_type, source_frequency, source_amplitude, source_position, boundary_type, medium_type, refraction_index
    
    # Update time
    time += dt
    
    # Add source
    Z = add_source(Z, source_type, source_position, source_frequency, source_amplitude, time)
    
    # Apply boundary conditions
    Z = apply_boundary_conditions(Z, boundary_type)
    
    # Update wave field
    Z_next = update_wave_field(Z, Z_prev, medium, dt, damping)
    
    # Shift time steps
    Z_prev = Z.copy()
    Z = Z_next.copy()
    
    # Clear the previous surface plot
    ax1.clear()
    ax1.set_facecolor('#1C1C1C')
    ax1.set_title('Wave Propagation Simulation', color='white', fontsize=14)
    ax1.set_xlabel('X', color='white')
    ax1.set_ylabel('Y', color='white')
    ax1.set_zlabel('Amplitude', color='white')
    ax1.set_zlim(-1.5, 1.5)
    
    # Create a new surface plot
    wave_surf = ax1.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)
    
    # Update the 2D image
    wave_img.set_array(Z)
    
    # Update information text
    info_text.set_text(
        f"Simulation Parameters:\n"
        f"Time: {time:.2f} s\n"
        f"Wave Speed: {wave_speed:.2f} m/s\n"
        f"Damping: {damping:.3f}\n"
        f"Source Type: {source_type}\n"
        f"Source Frequency: {source_frequency:.1f} Hz\n"
        f"Source Amplitude: {source_amplitude:.1f}\n"
        f"Boundary Type: {boundary_type}\n"
        f"Medium Type: {medium_type}\n"
        f"Refraction Index: {refraction_index:.1f}"
    )
    
    # Rotate the view for a dynamic perspective
    ax1.view_init(elev=30, azim=frame % 360)
    
    return [wave_surf, wave_img, info_text]

# -----------------------------
# Update Functions for Widgets
# -----------------------------
def update_wave_speed(val):
    """Update the wave speed."""
    global wave_speed, medium, medium_type, refraction_index
    wave_speed = val
    medium = initialize_medium(medium_type, refraction_index)
    medium_img.set_array(medium)
    fig.canvas.draw_idle()

def update_damping(val):
    """Update the damping coefficient."""
    global damping
    damping = val

def update_source_frequency(val):
    """Update the source frequency."""
    global source_frequency
    source_frequency = val

def update_source_amplitude(val):
    """Update the source amplitude."""
    global source_amplitude
    source_amplitude = val

def update_refraction_index(val):
    """Update the refraction index."""
    global refraction_index, medium, medium_type
    refraction_index = val
    medium = initialize_medium(medium_type, refraction_index)
    medium_img.set_array(medium)
    fig.canvas.draw_idle()

def update_source_type(label):
    """Update the source type."""
    global source_type
    source_type = label

def update_boundary_type(label):
    """Update the boundary type."""
    global boundary_type
    boundary_type = label

def update_medium_type(label):
    """Update the medium type."""
    global medium_type, medium, refraction_index
    medium_type = label
    medium = initialize_medium(medium_type, refraction_index)
    medium_img.set_array(medium)
    fig.canvas.draw_idle()

def reset_simulation(event):
    """Reset the simulation to initial conditions."""
    global Z, Z_prev, Z_next, time
    Z = np.zeros((grid_size, grid_size))
    Z_prev = np.zeros((grid_size, grid_size))
    Z_next = np.zeros((grid_size, grid_size))
    time = 0
    fig.canvas.draw_idle()

# Connect callbacks
speed_slider.on_changed(update_wave_speed)
damping_slider.on_changed(update_damping)
freq_slider.on_changed(update_source_frequency)
amp_slider.on_changed(update_source_amplitude)
refract_slider.on_changed(update_refraction_index)
source_radio.on_clicked(update_source_type)
boundary_radio.on_clicked(update_boundary_type)
medium_radio.on_clicked(update_medium_type)
reset_button.on_clicked(reset_simulation)

# Create animation
ani = FuncAnimation(fig, animate, frames=1000, interval=50, blit=True, init_func=init)

plt.show() 