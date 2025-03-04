import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# -----------------------------
# Simulation parameters (initial)
# -----------------------------
L = 10.0               # Side length of the cubic container (meters)
dt = 0.05              # Time step (seconds)
mass = 1.0             # Mass (kg)
r = 0.5                # Effective collision radius of particles (meters)

# Global collision counters
wall_collision_count = 0       # Collisions with the container walls
particle_collision_count = 0   # Collisions between particles

# Initial simulation settings
N_init = 50            # Initial number of particles
temperature_init = 200 # Initial temperature (affects particle speed)

# Add simulation time and dynamic parameters display
sim_time = 0  # Initialize simulation time

# -----------------------------
# Particle Initialization Function
# -----------------------------
def init_particles(N, L, temperature):
    """
    Initialize N particles with random positions in a cube of side L,
    and assign velocities with magnitude proportional to sqrt(temperature)
    in a random isotropic direction.
    """
    # Random positions in [0, L] for x, y, z
    positions = np.random.rand(N, 3) * L
    
    # Generate random directions: sample from a normal distribution and normalize
    v = np.random.normal(size=(N, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v / norms
    
    # Set speed proportional to sqrt(temperature)
    speed = np.sqrt(temperature)
    velocities = speed * v_unit
    
    return positions, velocities

# Initialize particles
positions, velocities = init_particles(N_init, L, temperature_init)

# -----------------------------
# Set Up the Figure and 3D Plot
# -----------------------------
plt.style.use('dark_background')  # Use dark theme for better contrast
fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor('#1C1C1C')  # Dark background

# Create a 3D subplot with custom styling
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_facecolor('#1C1C1C')  # Match background color

fig.subplots_adjust(bottom=0.30, left=0.08, right=0.75)

# Create a scatter plot for particle positions with glowing effect
scatter_colors = plt.cm.viridis(np.linspace(0, 1, N_init))  # Use viridis colormap
scat = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                 s=100, c=scatter_colors,
                 alpha=1, edgecolor='white', linewidth=0.5)

# Customize axis appearance
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor('white')
    axis.pane.set_alpha(0.3)
    axis.line.set_color('white')
    axis.label.set_color('white')
    axis.pane.fill = True  # Enable filling for the x-axis pane
    axis.pane.fill = True  
    axis.pane.fill = True  
    axis.pane.set_alpha(0.1)  # Set low opacity for the x-axis pane
    axis.pane.set_alpha(0.1)  
    axis.pane.set_alpha(0.1)  
    
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(0, L)
ax.set_title('3D Ideal Gas Simulation', color='white', pad=20, fontsize=14)

# Create an info text overlay with better styling
info_text = ax.text2D(0.0, 0.55, '', transform=ax.transAxes,
                      horizontalalignment='right', verticalalignment='top',
                      fontsize=10, color='white',
                      bbox=dict(facecolor='#2C2C2C', edgecolor='white',
                              alpha=0.8, boxstyle='round,pad=0.5'))

# -----------------------------
# Interactive Widgets Styling
# -----------------------------
slider_color = '#4A90E2'  # Nice blue color for sliders
text_color = 'white'
slider_kwargs = dict(
    color=slider_color,
    track_color='#2C2C2C',
    handle_style={'facecolor': slider_color, 'edgecolor': 'white', 'size': 10}
)

# Slider for Box Size (L)
ax_vol = plt.axes([0.15, 0.20, 0.65, 0.03])
vol_slider = Slider(
    ax=ax_vol,
    label='Box Size (L)',
    valmin=5,
    valmax=20,
    valinit=L,
    valstep=0.5,
    **slider_kwargs
)
vol_slider.label.set_color(text_color)

# Slider for Temperature
ax_temp = plt.axes([0.15, 0.15, 0.65, 0.03])
temp_slider = Slider(
    ax=ax_temp,
    label='Temperature',
    valmin=10,
    valmax=1000,
    valinit=temperature_init,
    valstep=10,
    **slider_kwargs
)
temp_slider.label.set_color(text_color)

# Slider for Number of Particles
ax_num = plt.axes([0.15, 0.10, 0.65, 0.03])
num_slider = Slider(
    ax=ax_num,
    label='Particles',
    valmin=10,
    valmax=200,
    valinit=N_init,
    valstep=1,
    **slider_kwargs
)
num_slider.label.set_color(text_color)

# Style the Reset button
ax_button = plt.axes([0.15, 0.025, 0.1, 0.04])
reset_button = Button(ax_button, 'Reset',
                     color='#2C2C2C',
                     hovercolor='#4A90E2')
reset_button.label.set_color(text_color)

# -----------------------------
# Text Annotation Styling
# -----------------------------
ax_annotation = fig.add_axes([0.65, 0.15, 0.20, 0.75])
ax_annotation.axis('off')
ax_annotation.patch.set_facecolor('#1C1C1C')
annotation_text = (
    "Physics Explanation (3D Ideal Gas Model):\n\n"
    "• Each particle represents a gas molecule, with color indicating its speed:\n"
    "  - Brighter/warmer colors = faster particles\n"
    "  - Darker/cooler colors = slower particles\n\n"
    "• Particle speed (v) is proportional to √T, following the kinetic theory of gases:\n"
    "  ½mv² = (3/2)kT\n\n"
    "• Average speed is related to temperature by:\n"
    "  v_avg = √(3kT/m)\n\n"
    "• The Ideal Gas Law states that PV = nRT, where:\n"
    "  P = Pressure\n"
    "  V = Volume (L³)\n"
    "  n = Number of moles\n"
    "  R = Gas constant\n"
    "  T = Temperature\n\n"
    "• All collisions are perfectly elastic, conserving both momentum and kinetic energy.\n\n"
    "Interactive Controls:\n"
    "• Adjust temperature to see changes in particle speeds\n"
    "• Modify volume to observe pressure changes\n"
    "• Add/remove particles to study density effects"
)
ax_annotation.text(0, 1, annotation_text,
                  verticalalignment='top',
                  fontsize=10,
                  color='white',
                  linespacing=1.5)

# -----------------------------
# Animation Update Function
# -----------------------------
def update(frame):
    global positions, velocities, wall_collision_count, particle_collision_count, L, sim_time

    # Update simulation time
    sim_time += dt

    # Update positions using current velocities
    positions += velocities * dt

    # Calculate average speed
    speeds = np.linalg.norm(velocities, axis=1)
    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)

    # -------------------------
    # Wall Collisions (in 3D)
    # -------------------------
    N = len(positions)
    for i in range(N):
        # X-direction
        if positions[i, 0] <= r and velocities[i, 0] < 0:  # Only count when moving towards wall
            positions[i, 0] = r
            velocities[i, 0] *= -1
            wall_collision_count += 1
        elif positions[i, 0] >= L - r and velocities[i, 0] > 0:  # Only count when moving towards wall
            positions[i, 0] = L - r
            velocities[i, 0] *= -1
            wall_collision_count += 1
        
        # Y-direction
        if positions[i, 1] <= r and velocities[i, 1] < 0:
            positions[i, 1] = r
            velocities[i, 1] *= -1
            wall_collision_count += 1
        elif positions[i, 1] >= L - r and velocities[i, 1] > 0:
            positions[i, 1] = L - r
            velocities[i, 1] *= -1
            wall_collision_count += 1
        
        # Z-direction
        if positions[i, 2] <= r and velocities[i, 2] < 0:
            positions[i, 2] = r
            velocities[i, 2] *= -1
            wall_collision_count += 1
        elif positions[i, 2] >= L - r and velocities[i, 2] > 0:
            positions[i, 2] = L - r
            velocities[i, 2] *= -1
            wall_collision_count += 1

    # -------------------------
    # Particle Collisions (3D)
    # -------------------------
    for i in range(N):
        for j in range(i+1, N):
            diff = positions[i] - positions[j]
            distance = np.linalg.norm(diff)
            if distance < 2 * r and distance > 0:
                # Unit vector along the line joining centers
                n = diff / distance
                # Check if particles are moving towards each other
                if np.dot(velocities[i] - velocities[j], n) < 0:
                    # Relative velocity along n
                    v_rel = np.dot(velocities[i] - velocities[j], n)
                    # Elastic collision update for identical masses
                    velocities[i] = velocities[i] - v_rel * n
                    velocities[j] = velocities[j] + v_rel * n
                    particle_collision_count += 1

    # Update particle colors based on their speeds relative to max speed
    colors = plt.cm.plasma(speeds / max_speed)
    scat.set_color(colors)
    
    # Update particle positions with glowing effect
    scat._offsets3d = (positions[:,0], positions[:,1], positions[:,2])
    
    # Compute expected pressure (Ideal Gas Law): P_expected = nRT/V
    R = 8.314  # Universal gas constant J/(mol·K)
    n = N / 6.022e23  # Number of moles (assuming each particle represents one molecule)
    T = temperature_init  # current temperature from slider
    V = L**3  # Volume of the cube
    P_expected = (n * R * T) / V
    
    # Format the info text with better layout and speed information
    info_text.set_text(
        f'Simulation Time: {sim_time:.2f} s\n'
        f'─────────────────────\n'
        f'Parameters:\n'
        f'Particles (N): {N_init}\n'
        f'Temperature: {temperature_init:.1f} K\n'
        f'Box Size: {L:.1f} m\n'
        f'─────────────────────\n'
        f'Statistics:\n'
        f'Average Speed: {avg_speed:.2f} m/s\n'
        f'Max Speed: {max_speed:.2f} m/s\n'
        f'Expected Pressure: {P_expected:.2e} Pa\n'
        f'Wall Collisions: {wall_collision_count}\n'
        f'Particle Collisions: {particle_collision_count}'
    )
    
    return scat, info_text

# Create the animation object. (blit is set to False for 3D animations)
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)

# -----------------------------
# Slider and Button Update Functions
# -----------------------------
def update_volume(val):
    global L
    L = vol_slider.val
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    fig.canvas.draw_idle()

def update_temperature(val):
    global velocities, temperature_init
    temperature_init = temp_slider.val
    # Preserve direction but update speed according to new temperature
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocity_directions = velocities / speeds
    new_speeds = np.sqrt(temperature_init) * np.ones_like(speeds)
    velocities = velocity_directions * new_speeds
    fig.canvas.draw_idle()

def update_particle_count(val):
    global positions, velocities, N_init, scat
    new_N = int(num_slider.val)
    
    if new_N > N_init:
        # Add particles
        n_new = new_N - N_init
        new_positions = np.random.rand(n_new, 3) * L
        new_velocities = np.random.normal(size=(n_new, 3))
        new_velocities /= np.linalg.norm(new_velocities, axis=1, keepdims=True)
        new_velocities *= np.sqrt(temperature_init)
        
        positions = np.vstack((positions, new_positions))
        velocities = np.vstack((velocities, new_velocities))
    elif new_N < N_init:
        # Remove random particles
        keep_indices = np.random.choice(N_init, new_N, replace=False)
        positions = positions[keep_indices]
        velocities = velocities[keep_indices]
    
    N_init = new_N
    
    # Update scatter plot - create new scatter plot instead of updating existing one
    scat.remove()  # Remove old scatter plot
    speeds = np.linalg.norm(velocities, axis=1)
    colors = plt.cm.viridis((speeds - speeds.min()) / (speeds.max() - speeds.min()))
    scat = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                     s=100, c=colors,
                     alpha=1, edgecolor='white', linewidth=0.5)
    
    fig.canvas.draw_idle()

def reset(event):
    global positions, velocities, wall_collision_count, particle_collision_count, L, temperature_init, N_init, sim_time
    # Reset all parameters to initial values
    L = 10.0
    temperature_init = 300
    N_init = 50
    wall_collision_count = 0
    particle_collision_count = 0
    sim_time = 0
    
    # Reset sliders to initial values
    vol_slider.reset()
    temp_slider.reset()
    num_slider.reset()
    
    # Reinitialize particles
    positions, velocities = init_particles(N_init, L, temperature_init)
    
    # Reset axis limits
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    
    # Update scatter plot
    scatter_colors = plt.cm.viridis(np.linspace(0, 1, N_init))
    scat.set_offsets(positions)
    scat.set_color(scatter_colors)
    
    fig.canvas.draw_idle()

# Connect the update functions to the sliders and button
vol_slider.on_changed(update_volume)
temp_slider.on_changed(update_temperature)
num_slider.on_changed(update_particle_count)
reset_button.on_clicked(reset)

# -----------------------------
# Run the Visualization
# -----------------------------
plt.show()
