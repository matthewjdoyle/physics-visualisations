import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# Simulation Parameters
# -----------------------------
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
scale_factor = 1e9  # Scale factor for visualization (makes G larger for more visible effects)
G_scaled = G * scale_factor

# Initial number of bodies
N_bodies = 5

# Simulation parameters
dt = 0.01  # Time step
softening = 0.1  # Softening parameter to avoid singularities
show_trails = True  # Whether to show trails
max_trail_length = 100  # Maximum number of points in each trail
collision_detection = True  # Whether to detect and handle collisions
collision_threshold = 0.2  # Distance threshold for collision detection
energy_conservation = False  # Whether to enforce energy conservation

# -----------------------------
# Body Class Definition
# -----------------------------
class Body:
    def __init__(self, mass, position, velocity, color=None, radius=None):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.trajectory = [position.copy()]
        
        # Set color based on mass if not provided
        if color is None:
            # Create a color based on mass (larger mass = warmer color)
            normalized_mass = np.clip(mass / 10.0, 0, 1)
            self.color = plt.cm.plasma(normalized_mass)
        else:
            self.color = color
            
        # Set radius based on mass if not provided
        if radius is None:
            self.radius = np.cbrt(mass) * 0.1  # Radius proportional to cube root of mass
        else:
            self.radius = radius

    def update_position(self, dt):
        """Update position based on velocity."""
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())
        
        # Keep trajectory at a reasonable length
        if len(self.trajectory) > max_trail_length:
            self.trajectory.pop(0)

    def update_velocity(self, dt):
        """Update velocity based on acceleration."""
        self.velocity += self.acceleration * dt

    def reset_acceleration(self):
        """Reset acceleration to zero."""
        self.acceleration = np.zeros(3)

# -----------------------------
# Physics Simulation Functions
# -----------------------------
def initialize_random_bodies(n, box_size=10.0, max_mass=10.0, max_velocity=1.0):
    """Initialize n random bodies within a box of given size."""
    bodies = []
    
    # Create a central massive body (like a star)
    central_mass = max_mass * 5
    central_body = Body(
        mass=central_mass,
        position=np.array([0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        color='yellow'
    )
    bodies.append(central_body)
    
    # Create n-1 smaller bodies
    for i in range(n-1):
        # Random position within the box
        position = (np.random.rand(3) - 0.5) * box_size
        
        # Ensure the body is not too close to the central body
        while np.linalg.norm(position) < 1.0:
            position = (np.random.rand(3) - 0.5) * box_size
        
        # Random mass
        mass = np.random.uniform(0.1, max_mass)
        
        # Calculate orbital velocity for a circular orbit (simplified)
        distance = np.linalg.norm(position)
        orbit_speed = np.sqrt(G_scaled * central_mass / distance)
        
        # Create velocity perpendicular to position vector for orbital motion
        position_unit = position / distance
        # Create a perpendicular vector in the x-y plane
        perpendicular = np.array([-position_unit[1], position_unit[0], 0])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        # Set velocity with some randomness
        velocity = perpendicular * orbit_speed * np.random.uniform(0.8, 1.2)
        
        # Add some random z-component to make it 3D
        velocity[2] = np.random.uniform(-0.2, 0.2) * orbit_speed
        
        # Create the body
        body = Body(mass=mass, position=position, velocity=velocity)
        bodies.append(body)
    
    return bodies

def compute_acceleration(bodies, body_idx, G, softening):
    """Compute acceleration for a body due to gravitational forces from other bodies."""
    body = bodies[body_idx]
    acceleration = np.zeros(3)
    
    for i, other in enumerate(bodies):
        if i != body_idx:
            r = other.position - body.position
            distance = np.linalg.norm(r)
            
            # Use softening to avoid singularities
            if distance > softening:
                # F = G * m1 * m2 / r^2
                # a = F / m1 = G * m2 / r^2
                acceleration += G * other.mass * r / (distance ** 3)
    
    return acceleration

def detect_collisions(bodies, threshold):
    """Detect and handle collisions between bodies."""
    collisions = []
    
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            r = bodies[i].position - bodies[j].position
            distance = np.linalg.norm(r)
            
            # Check if bodies are close enough to collide
            if distance < threshold * (bodies[i].radius + bodies[j].radius):
                collisions.append((i, j))
    
    return collisions

def handle_collision(bodies, i, j):
    """Handle collision between bodies i and j (perfectly inelastic collision)."""
    body1 = bodies[i]
    body2 = bodies[j]
    
    # Calculate total momentum
    total_momentum = body1.mass * body1.velocity + body2.mass * body2.velocity
    
    # Calculate total mass
    total_mass = body1.mass + body2.mass
    
    # Calculate center of mass position
    com_position = (body1.mass * body1.position + body2.mass * body2.position) / total_mass
    
    # Calculate new velocity (conserve momentum)
    new_velocity = total_momentum / total_mass
    
    # Create new body at center of mass
    new_body = Body(
        mass=total_mass,
        position=com_position,
        velocity=new_velocity
    )
    
    # Replace the more massive body with the new body
    if body1.mass >= body2.mass:
        bodies[i] = new_body
        return j  # Return index of the less massive body to be removed
    else:
        bodies[j] = new_body
        return i  # Return index of the less massive body to be removed

def update_system(bodies, dt, G, softening, collision_detection, collision_threshold):
    """Update the entire N-body system by one time step."""
    # Compute accelerations
    for i, body in enumerate(bodies):
        body.reset_acceleration()
        body.acceleration = compute_acceleration(bodies, i, G, softening)
    
    # Update velocities
    for body in bodies:
        body.update_velocity(dt)
    
    # Update positions
    for body in bodies:
        body.update_position(dt)
    
    # Handle collisions if enabled
    if collision_detection:
        collisions = detect_collisions(bodies, collision_threshold)
        
        # Process collisions (remove indices in reverse order to avoid index issues)
        indices_to_remove = []
        for i, j in collisions:
            idx_to_remove = handle_collision(bodies, i, j)
            indices_to_remove.append(idx_to_remove)
        
        # Remove bodies that were absorbed in collisions
        for idx in sorted(indices_to_remove, reverse=True):
            if idx < len(bodies):  # Safety check
                bodies.pop(idx)
    
    return bodies

def calculate_system_energy(bodies, G):
    """Calculate the total energy (kinetic + potential) of the system."""
    kinetic_energy = 0
    potential_energy = 0
    
    # Calculate kinetic energy
    for body in bodies:
        kinetic_energy += 0.5 * body.mass * np.sum(body.velocity**2)
    
    # Calculate potential energy
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i < j:  # Avoid double counting
                r = np.linalg.norm(body1.position - body2.position)
                if r > 0:  # Avoid division by zero
                    potential_energy -= G * body1.mass * body2.mass / r
    
    return kinetic_energy + potential_energy

# -----------------------------
# Set Up the Figure and Plot
# -----------------------------
plt.style.use('dark_background')  # Use dark theme for better contrast
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#1C1C1C')  # Dark background

# Create a grid for the plots
gs = plt.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])

# Main 3D plot for the N-body simulation
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.set_facecolor('#1C1C1C')
ax1.set_title('N-Body Gravitational Simulation', color='white', fontsize=14)
ax1.set_xlabel('X', color='white')
ax1.set_ylabel('Y', color='white')
ax1.set_zlabel('Z', color='white')

# Energy plot
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#1C1C1C')
ax2.set_title('System Energy over Time', color='white')
ax2.set_xlabel('Time Step', color='white')
ax2.set_ylabel('Energy', color='white')
ax2.grid(alpha=0.2)

# Orbital parameters plot
ax3 = fig.add_subplot(gs[0, 1])
ax3.set_facecolor('#1C1C1C')
ax3.set_title('Orbital Parameters', color='white')
ax3.set_xlabel('Body Index', color='white')
ax3.set_ylabel('Value', color='white')
ax3.grid(alpha=0.2)

# Information panel
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#1C1C1C')
ax4.axis('off')

# Adjust layout
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, wspace=0.3, hspace=0.3)

# -----------------------------
# Initialize Simulation
# -----------------------------
# Initialize bodies
bodies = initialize_random_bodies(N_bodies)

# Set axis limits based on initial positions
max_pos = max([np.max(np.abs(body.position)) for body in bodies]) * 1.5
ax1.set_xlim(-max_pos, max_pos)
ax1.set_ylim(-max_pos, max_pos)
ax1.set_zlim(-max_pos, max_pos)

# Initialize plot elements
body_points = []
body_trails = []
for body in bodies:
    point, = ax1.plot([], [], [], 'o', markersize=20*body.radius, color=body.color)
    trail, = ax1.plot([], [], [], '-', linewidth=1, alpha=0.5, color=body.color)
    body_points.append(point)
    body_trails.append(trail)

# Energy plot
energy_line, = ax2.plot([], [], lw=2, color='#4A90E2')
energy_values = []
time_steps = []

# Orbital parameters plot (semi-major axis, eccentricity)
bar_positions = np.arange(len(bodies))
semi_major_bars = ax3.bar(bar_positions - 0.2, np.zeros(len(bodies)), width=0.4, color='cyan', alpha=0.7, label='Semi-major Axis')
eccentricity_bars = ax3.bar(bar_positions + 0.2, np.zeros(len(bodies)), width=0.4, color='magenta', alpha=0.7, label='Eccentricity')
ax3.legend(loc='upper right', facecolor='#2C2C2C', edgecolor='white', labelcolor='white')

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

# Slider for gravitational constant
ax_G = plt.axes([0.1, 0.25, 0.35, 0.03])
G_slider = Slider(ax_G, "G (scaled)", 0.1, 10.0, valinit=1.0, valstep=0.1, **slider_kwargs)
G_slider.label.set_color(text_color)

# Slider for time step
ax_dt = plt.axes([0.1, 0.20, 0.35, 0.03])
dt_slider = Slider(ax_dt, "Time Step", 0.001, 0.05, valinit=dt, valstep=0.001, **slider_kwargs)
dt_slider.label.set_color(text_color)

# Slider for softening parameter
ax_soft = plt.axes([0.1, 0.15, 0.35, 0.03])
soft_slider = Slider(ax_soft, "Softening", 0.01, 1.0, valinit=softening, valstep=0.01, **slider_kwargs)
soft_slider.label.set_color(text_color)

# Slider for number of bodies
ax_nbodies = plt.axes([0.1, 0.10, 0.35, 0.03])
nbodies_slider = Slider(ax_nbodies, "Number of Bodies", 2, 20, valinit=N_bodies, valstep=1, **slider_kwargs)
nbodies_slider.label.set_color(text_color)

# Slider for trail length
ax_trail = plt.axes([0.1, 0.05, 0.35, 0.03])
trail_slider = Slider(ax_trail, "Trail Length", 10, 500, valinit=max_trail_length, valstep=10, **slider_kwargs)
trail_slider.label.set_color(text_color)

# Checkboxes for options
ax_check = plt.axes([0.6, 0.05, 0.2, 0.15])
check = CheckButtons(
    ax_check, 
    ['Show Trails', 'Collision Detection', 'Energy Conservation'],
    [show_trails, collision_detection, energy_conservation]
)
for i, label in enumerate(check.labels):
    label.set_color(text_color)
for i, rect in enumerate(check.rectangles):
    rect.set_facecolor('#2C2C2C')
    rect.set_edgecolor('white')

# Reset button
ax_reset = plt.axes([0.6, 0.25, 0.1, 0.04])
reset_button = Button(ax_reset, 'Reset', color='#2C2C2C', hovercolor='#4A90E2')
reset_button.label.set_color(text_color)

# Add body button
ax_add = plt.axes([0.75, 0.25, 0.1, 0.04])
add_button = Button(ax_add, 'Add Body', color='#2C2C2C', hovercolor='#4A90E2')
add_button.label.set_color(text_color)

# -----------------------------
# Animation Functions
# -----------------------------
def calculate_orbital_parameters(bodies):
    """Calculate orbital parameters for each body relative to the most massive body."""
    # Find the most massive body (assumed to be the central body)
    central_idx = np.argmax([body.mass for body in bodies])
    central_body = bodies[central_idx]
    
    semi_major_axes = []
    eccentricities = []
    
    for i, body in enumerate(bodies):
        if i == central_idx:
            # Central body has no orbit around itself
            semi_major_axes.append(0)
            eccentricities.append(0)
            continue
        
        # Calculate relative position and velocity
        r = body.position - central_body.position
        v = body.velocity - central_body.velocity
        
        # Calculate distance
        distance = np.linalg.norm(r)
        
        # Calculate specific energy
        specific_energy = 0.5 * np.sum(v**2) - G_scaled * central_body.mass / distance
        
        # Calculate semi-major axis
        if specific_energy >= 0:
            # Parabolic or hyperbolic orbit
            semi_major_axis = float('inf')
        else:
            semi_major_axis = -G_scaled * central_body.mass / (2 * specific_energy)
        
        # Calculate angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Calculate eccentricity
        if h_mag > 0:
            eccentricity = np.sqrt(1 + 2 * specific_energy * h_mag**2 / (G_scaled * central_body.mass)**2)
        else:
            eccentricity = 0
        
        semi_major_axes.append(min(semi_major_axis, 10))  # Cap for visualization
        eccentricities.append(min(eccentricity, 1))  # Cap for visualization
    
    return semi_major_axes, eccentricities

def init():
    """Initialize the animation."""
    for point, trail in zip(body_points, body_trails):
        point.set_data([], [])
        point.set_3d_properties([])
        trail.set_data([], [])
        trail.set_3d_properties([])
    
    energy_line.set_data([], [])
    
    return body_points + body_trails + [energy_line]

def animate(frame):
    """Update the animation for each frame."""
    global bodies, energy_values, time_steps, G_scaled, dt, softening, max_trail_length
    global show_trails, collision_detection, energy_conservation
    
    # Update system for 5 steps per frame for smoother motion
    for _ in range(5):
        bodies = update_system(bodies, dt, G_scaled, softening, collision_detection, collision_threshold)
        
        # Calculate and store energy
        energy = calculate_system_energy(bodies, G_scaled)
        energy_values.append(energy)
        time_steps.append(len(time_steps))
        
        # Keep energy and time_steps at a reasonable length
        if len(energy_values) > 1000:
            energy_values.pop(0)
            time_steps.pop(0)
    
    # Update body positions and trails
    for i, (body, point, trail) in enumerate(zip(bodies, body_points, body_trails)):
        point.set_data([body.position[0]], [body.position[1]])
        point.set_3d_properties([body.position[2]])
        
        if show_trails and len(body.trajectory) > 1:
            trail_x = [pos[0] for pos in body.trajectory]
            trail_y = [pos[1] for pos in body.trajectory]
            trail_z = [pos[2] for pos in body.trajectory]
            trail.set_data(trail_x, trail_y)
            trail.set_3d_properties(trail_z)
        else:
            trail.set_data([], [])
            trail.set_3d_properties([])
    
    # Update energy plot
    if energy_values:
        energy_line.set_data(time_steps, energy_values)
        ax2.set_xlim(0, max(time_steps))
        ax2.set_ylim(min(energy_values) * 0.9, max(energy_values) * 1.1)
    
    # Calculate and update orbital parameters
    if len(bodies) > 1:
        semi_major_axes, eccentricities = calculate_orbital_parameters(bodies)
        
        # Update bar chart
        for i, (semi, ecc) in enumerate(zip(semi_major_axes, eccentricities)):
            if i < len(semi_major_bars):
                semi_major_bars[i].set_height(semi)
                eccentricity_bars[i].set_height(ecc)
        
        # Update x-axis limits if needed
        ax3.set_xlim(-0.5, len(bodies) - 0.5)
        ax3.set_ylim(0, max(max(semi_major_axes), 1) * 1.1)
    
    # Update information text
    info_text.set_text(
        f"System Information:\n"
        f"Number of Bodies: {len(bodies)}\n"
        f"G (scaled): {G_scaled:.2f}\n"
        f"Time Step: {dt:.4f}\n"
        f"Softening: {softening:.2f}\n\n"
        f"Total Mass: {sum(body.mass for body in bodies):.2f}\n"
        f"Energy: {energy_values[-1]:.2e}\n"
        f"Energy Change: {(energy_values[-1] - energy_values[0]) / energy_values[0] * 100:.2f}%\n\n"
        f"Options:\n"
        f"Trails: {'On' if show_trails else 'Off'}\n"
        f"Collisions: {'On' if collision_detection else 'Off'}\n"
        f"Energy Conservation: {'On' if energy_conservation else 'Off'}"
    )
    
    # Rotate the view for a dynamic perspective
    ax1.view_init(elev=30, azim=frame % 360)
    
    return body_points + body_trails + [energy_line, info_text]

# -----------------------------
# Update Functions for Widgets
# -----------------------------
def update_G(val):
    """Update the gravitational constant."""
    global G_scaled
    G_scaled = G * scale_factor * val
    
def update_dt(val):
    """Update the time step."""
    global dt
    dt = val
    
def update_softening(val):
    """Update the softening parameter."""
    global softening
    softening = val
    
def update_trail_length(val):
    """Update the maximum trail length."""
    global max_trail_length
    max_trail_length = int(val)
    
    # Update existing trajectories
    for body in bodies:
        if len(body.trajectory) > max_trail_length:
            body.trajectory = body.trajectory[-max_trail_length:]

def update_options(label):
    """Update simulation options based on checkbox selection."""
    global show_trails, collision_detection, energy_conservation
    
    if label == 'Show Trails':
        show_trails = not show_trails
    elif label == 'Collision Detection':
        collision_detection = not collision_detection
    elif label == 'Energy Conservation':
        energy_conservation = not energy_conservation

def reset_simulation(event):
    """Reset the simulation to initial conditions."""
    global bodies, energy_values, time_steps, body_points, body_trails
    
    # Get current number of bodies
    n_bodies = int(nbodies_slider.val)
    
    # Re-initialize bodies
    bodies = initialize_random_bodies(n_bodies)
    
    # Reset energy values and time steps
    energy_values = []
    time_steps = []
    
    # Clear existing plot elements
    for point in body_points:
        point.remove()
    for trail in body_trails:
        trail.remove()
    
    # Reinitialize plot elements
    body_points = []
    body_trails = []
    for body in bodies:
        point, = ax1.plot([], [], [], 'o', markersize=20*body.radius, color=body.color)
        trail, = ax1.plot([], [], [], '-', linewidth=1, alpha=0.5, color=body.color)
        body_points.append(point)
        body_trails.append(trail)
    
    # Reset axis limits
    max_pos = max([np.max(np.abs(body.position)) for body in bodies]) * 1.5
    ax1.set_xlim(-max_pos, max_pos)
    ax1.set_ylim(-max_pos, max_pos)
    ax1.set_zlim(-max_pos, max_pos)
    
    # Reset orbital parameters plot
    ax3.clear()
    ax3.set_facecolor('#1C1C1C')
    ax3.set_title('Orbital Parameters', color='white')
    ax3.set_xlabel('Body Index', color='white')
    ax3.set_ylabel('Value', color='white')
    ax3.grid(alpha=0.2)
    
    bar_positions = np.arange(len(bodies))
    semi_major_bars = ax3.bar(bar_positions - 0.2, np.zeros(len(bodies)), width=0.4, color='cyan', alpha=0.7, label='Semi-major Axis')
    eccentricity_bars = ax3.bar(bar_positions + 0.2, np.zeros(len(bodies)), width=0.4, color='magenta', alpha=0.7, label='Eccentricity')
    ax3.legend(loc='upper right', facecolor='#2C2C2C', edgecolor='white', labelcolor='white')
    
    fig.canvas.draw_idle()

def add_new_body(event):
    """Add a new body to the simulation."""
    global bodies, body_points, body_trails
    
    if len(bodies) >= 20:  # Limit the number of bodies
        return
    
    # Find the most massive body (assumed to be the central body)
    central_idx = np.argmax([body.mass for body in bodies])
    central_body = bodies[central_idx]
    
    # Create a new body in a random position
    angle = np.random.uniform(0, 2*np.pi)
    distance = np.random.uniform(3, 8)
    
    # Position in the orbital plane with small z-component
    position = np.array([
        distance * np.cos(angle),
        distance * np.sin(angle),
        np.random.uniform(-0.5, 0.5)
    ])
    
    # Calculate orbital velocity for a circular orbit
    orbit_speed = np.sqrt(G_scaled * central_body.mass / distance)
    
    # Velocity perpendicular to position vector
    velocity = np.array([
        -position[1],
        position[0],
        0
    ])
    velocity = velocity / np.linalg.norm(velocity) * orbit_speed
    
    # Add some random z-component
    velocity[2] = np.random.uniform(-0.1, 0.1) * orbit_speed
    
    # Random mass
    mass = np.random.uniform(0.1, 2.0)
    
    # Create the new body
    new_body = Body(mass=mass, position=position, velocity=velocity)
    bodies.append(new_body)
    
    # Add plot elements for the new body
    point, = ax1.plot([], [], [], 'o', markersize=20*new_body.radius, color=new_body.color)
    trail, = ax1.plot([], [], [], '-', linewidth=1, alpha=0.5, color=new_body.color)
    body_points.append(point)
    body_trails.append(trail)
    
    # Update orbital parameters plot
    ax3.clear()
    ax3.set_facecolor('#1C1C1C')
    ax3.set_title('Orbital Parameters', color='white')
    ax3.set_xlabel('Body Index', color='white')
    ax3.set_ylabel('Value', color='white')
    ax3.grid(alpha=0.2)
    
    bar_positions = np.arange(len(bodies))
    semi_major_bars = ax3.bar(bar_positions - 0.2, np.zeros(len(bodies)), width=0.4, color='cyan', alpha=0.7, label='Semi-major Axis')
    eccentricity_bars = ax3.bar(bar_positions + 0.2, np.zeros(len(bodies)), width=0.4, color='magenta', alpha=0.7, label='Eccentricity')
    ax3.legend(loc='upper right', facecolor='#2C2C2C', edgecolor='white', labelcolor='white')
    
    fig.canvas.draw_idle()

def update_nbodies(val):
    """Update the number of bodies."""
    reset_simulation(None)  # Reset with the new number of bodies

# Connect callbacks
G_slider.on_changed(update_G)
dt_slider.on_changed(update_dt)
soft_slider.on_changed(update_softening)
trail_slider.on_changed(update_trail_length)
nbodies_slider.on_changed(update_nbodies)
check.on_clicked(update_options)
reset_button.on_clicked(reset_simulation)
add_button.on_clicked(add_new_body)

# Create animation
ani = FuncAnimation(fig, animate, frames=1000, interval=50, blit=True, init_func=init)

plt.show() 