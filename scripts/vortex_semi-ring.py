import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

"""
Vortex Filament Dynamics Visualization
--------------------------------------
This script visualizes the dynamics of a vortex filament using the local induction approximation (LIA).
The LIA describes the self-induced motion of a vortex filament in an inviscid fluid.

The dynamics are affected by:
- Flow velocity: External flow field that advects the vortex filament
- Mutual friction: Interaction between the vortex and normal fluid components
- Spatial resolution: Controlled by delta parameter for point spacing

Interactive controls allow adjustment of these parameters in real-time.
"""

# -----------------------------
# Simulation parameters (initial)
# -----------------------------
dt = 0.01                # Time step
N_points = 200           # Initial number of points to discretize the filament
circulation_init = 9.97e-4  # Initial circulation (quantum of circulation)
flow_velocity_init = 0.0 # Initial external flow velocity
alpha_lia = 0.1          # Local induction approximation parameter
L = 10                 # Domain size
# Initial flow direction vector (will be controlled by user)
flow_direction = np.array([1.0, 0.0, 0.0])  # Default: flow in x-direction

# Mutual friction coefficients (will be controlled by sliders)
alpha_mf = 0.1           # Mutual friction coefficient alpha
alpha_prime = 0.05       # Mutual friction coefficient alpha'

# Spatial resolution parameters
delta = 1e-2              # Minimum allowed distance between points
max_delta = 2 * delta    # Maximum allowed distance between points
core_radius = 8.244e-9       # Vortex core radius (corea in Fortran code)

# Time series data storage
max_time_points = 500    # Maximum number of time points to store
time_data = []           # Time points
metrics_data = {
    'point_count': [],       # Number of points in the filament
    'avg_velocity': [],      # Average velocity magnitude
    'max_velocity': [],      # Maximum velocity magnitude
    'filament_length': [],   # Total length of the filament
    'avg_curvature': [],     # Average curvature of the filament
    'kinetic_energy': []     # Approximate kinetic energy
}
current_metric = 'avg_velocity'  # Default metric to display

# Add simulation time
sim_time = 0  # Initialize simulation time

# -----------------------------
# Timestep Check Function
# -----------------------------
def timestep_check(dt, delta, quant_circ, core_radius):
    """
    Check if the timestep is compatible with the spatial resolution.
    
    Parameters:
    - dt: Current timestep
    - delta: Spatial resolution parameter
    - quant_circ: Quantum of circulation
    - core_radius: Vortex core radius
    
    Returns:
    - is_valid: Boolean indicating if timestep is valid
    - dt_max: Maximum allowed timestep
    """
    delta_min = delta / 2.0
    dt_max = abs((delta_min**2) / (quant_circ * np.log(delta_min / (np.pi * core_radius))))
    
    if dt < dt_max:
        print(f" dt is below maximum possible dt: {dt_max:.4e}")
        return True, dt_max
    else:
        print(f" WARNING: set dt below {dt_max:.4e}")
        print(f" Current dt = {dt:.4e} is too large!")
        return False, dt_max

# Check timestep validity at initialization
is_valid, dt_max = timestep_check(dt, delta, circulation_init, core_radius)
if not is_valid:
    # Adjust dt to be safe (e.g., 90% of maximum)
    dt = 0.9 * dt_max
    print(f" Adjusting dt to {dt:.4e}")

# -----------------------------
# Vortex Filament Initialization
# -----------------------------
def init_vortex_semi_ring(N_points, radius=3.0, perturbation=0.1):
    """
    Initialize a vortex semi-ring with a small perturbation that stands on a surface at z=0.
    """
    # Parametric representation of a semi-circle
    theta = np.linspace(0, np.pi, N_points, endpoint=True)
    
    # Base semi-ring (standing on the z=0 surface)
    x = np.zeros_like(theta)
    y = radius * np.cos(theta) # Start at z=0 (on the surface)
    z = radius * np.sin(theta)
    
    # Add a small perturbation in z-direction to make dynamics more interesting
    z += perturbation * np.sin(3*theta) * np.sin(theta)  # Ensure ends stay at z=0
    
    # Stack coordinates
    filament = np.column_stack((x, y, z))
    
    return filament

# Initialize vortex filament as a semi-ring
filament = init_vortex_semi_ring(N_points)

# -----------------------------
# Local Induction Approximation (LIA) with Surface Interaction
# -----------------------------
def compute_lia_velocity(filament, circulation, alpha_lia):
    """
    Compute the self-induced velocity of the vortex filament using LIA,
    with modifications to handle the surface interaction.
    """
    N = len(filament)
    velocity = np.zeros_like(filament)
    
    for i in range(N):
        # For endpoints, special handling to keep them on the surface
        if i == 0 or i == N-1:
            # Only allow movement along the surface (x-y plane at z=0)
            if i == 0:  # First endpoint
                next_point = filament[i+1]
                tangent = next_point - filament[i]
                tangent[2] = 0  # Project onto x-y plane (z=0 is the surface now)
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 0:
                    tangent = tangent / tangent_norm
                    # Move along the surface in the direction of the tangent
                    velocity[i] = circulation * tangent * 0.1  # Reduced speed for stability
            elif i == N-1:  # Last endpoint
                prev_point = filament[i-1]
                tangent = filament[i] - prev_point
                tangent[2] = 0  # Project onto x-y plane
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 0:
                    tangent = tangent / tangent_norm
                    # Move along the surface in the direction of the tangent
                    velocity[i] = circulation * tangent * 0.1  # Reduced speed for stability
            
            # Ensure z-component is zero to keep endpoints on the surface
            velocity[i, 2] = 0
            continue
        
        # For non-endpoint points, use regular LIA with neighboring points
        prev_point = filament[i-1]
        next_point = filament[i+1]
        
        # Compute tangent vector
        tangent = next_point - prev_point
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            tangent = tangent / tangent_norm
        
        # Compute local curvature vector
        segment1 = prev_point - filament[i]
        segment2 = next_point - filament[i]
        
        # Normalize segments
        norm1 = np.linalg.norm(segment1)
        norm2 = np.linalg.norm(segment2)
        
        if norm1 > 0 and norm2 > 0:
            segment1 = segment1 / norm1
            segment2 = segment2 / norm2
            
            # Compute binormal vector (cross product of segments)
            binormal = np.cross(segment1, segment2)
            binormal_norm = np.linalg.norm(binormal)
            
            if binormal_norm > 0:
                binormal = binormal / binormal_norm
                
                # Local curvature (approximation)
                curvature = binormal_norm / (0.5 * (norm1 + norm2))
                
                # LIA velocity = Γ/(4π) * ln(1/α) * κ * b
                # where Γ is circulation, α is a small parameter, κ is curvature, b is binormal
                velocity[i] = circulation * np.log(1/alpha_lia) * curvature * binormal / (4 * np.pi)
    
    # Apply surface interaction: reflect velocity if too close to surface
    for i in range(1, N-1):  # Skip endpoints which are handled separately
        if filament[i, 2] < 0.1 and velocity[i, 2] < 0:
            # Reflect the z-component of velocity if moving toward the surface
            velocity[i, 2] *= -0.8  # Damping factor for energy dissipation
    
    return velocity

# -----------------------------
# Apply Mutual Friction
# -----------------------------
def apply_mutual_friction(velocity, tangent, normal, alpha_mf, alpha_prime):
    """
    Apply mutual friction to the vortex velocity.
    
    The mutual friction force is given by:
    F_mf = -alpha_mf * (v × t) × t - alpha_prime * v × t
    
    where:
    - v is the velocity
    - t is the unit tangent vector
    - × denotes cross product
    """
    # Calculate v × t
    v_cross_t = np.cross(velocity, tangent)
    
    # Calculate (v × t) × t
    v_cross_t_cross_t = np.cross(v_cross_t, tangent)
    
    # Apply mutual friction
    friction_force = -alpha_mf * v_cross_t_cross_t - alpha_prime * v_cross_t
    
    # Return velocity with mutual friction applied
    return velocity + friction_force

# -----------------------------
# Filament Resolution Management
# -----------------------------
def adjust_filament_resolution(filament, delta, max_delta):
    """
    Adjust the spatial resolution of the vortex filament by adding or removing points.
    
    Parameters:
    - filament: The current vortex filament points
    - delta: Minimum allowed distance between points
    - max_delta: Maximum allowed distance between points
    
    Returns:
    - new_filament: The adjusted vortex filament
    """
    # Start with the first point
    new_filament = [filament[0]]
    
    # Process each segment
    i = 0
    while i < len(filament) - 1:
        current_point = filament[i]
        next_point = filament[i + 1]
        
        # Calculate distance between points
        distance = np.linalg.norm(next_point - current_point)
        
        if distance < delta and i > 0 and i < len(filament) - 2:
            # Points are too close - skip the next point (remove it)
            i += 2
            # Make sure we don't lose the last point
            if i >= len(filament) - 1 and not np.array_equal(next_point, filament[-1]):
                new_filament.append(filament[-1])
        elif distance > max_delta:
            # Points are too far apart - add a new point
            # Use linear interpolation for simplicity
            mid_point = 0.5 * (current_point + next_point)
            
            # If we have enough points around, use cubic interpolation for better shape preservation
            if i > 0 and i < len(filament) - 2:
                # Get additional points for cubic interpolation
                prev_point = filament[i - 1]
                next_next_point = filament[i + 2]
                
                # Calculate parameter t along the curve (0 to 1)
                t = 0.5  # Midpoint
                
                # Catmull-Rom spline interpolation
                t2 = t * t
                t3 = t2 * t
                
                # Catmull-Rom coefficients
                c0 = -0.5 * t3 + t2 - 0.5 * t
                c1 = 1.5 * t3 - 2.5 * t2 + 1.0
                c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
                c3 = 0.5 * t3 - 0.5 * t2
                
                # Interpolate
                mid_point = c0 * prev_point + c1 * current_point + c2 * next_point + c3 * next_next_point
            
            new_filament.append(mid_point)
            new_filament.append(next_point)
            i += 1
        else:
            # Distance is acceptable, keep the next point
            new_filament.append(next_point)
            i += 1
    
    # Ensure we have the last point
    if len(new_filament) > 0 and not np.array_equal(new_filament[-1], filament[-1]):
        new_filament.append(filament[-1])
    
    # Ensure we have at least 3 points for a valid filament
    if len(new_filament) < 3:
        return filament  # Return original if we lost too many points
    
    # Convert to numpy array
    return np.array(new_filament)

# -----------------------------
# Calculate Diagnostic Metrics
# -----------------------------
def calculate_metrics(filament, velocities):
    """
    Calculate various diagnostic metrics for the vortex filament.
    
    Parameters:
    - filament: The vortex filament points
    - velocities: Velocity at each point
    
    Returns:
    - metrics: Dictionary of calculated metrics
    """
    metrics = {}
    
    # Number of points
    metrics['point_count'] = len(filament)
    
    # Velocity statistics
    speeds = np.linalg.norm(velocities, axis=1)
    metrics['avg_velocity'] = np.mean(speeds) if len(speeds) > 0 else 0
    metrics['max_velocity'] = np.max(speeds) if len(speeds) > 0 else 0
    
    # Calculate filament length
    length = 0
    for i in range(len(filament) - 1):
        length += np.linalg.norm(filament[i+1] - filament[i])
    metrics['filament_length'] = length
    
    # Calculate average curvature
    curvatures = []
    for i in range(1, len(filament) - 1):
        prev_point = filament[i-1]
        current_point = filament[i]
        next_point = filament[i+1]
        
        segment1 = prev_point - current_point
        segment2 = next_point - current_point
        
        norm1 = np.linalg.norm(segment1)
        norm2 = np.linalg.norm(segment2)
        
        if norm1 > 0 and norm2 > 0:
            segment1 = segment1 / norm1
            segment2 = segment2 / norm2
            
            # Compute binormal vector (cross product of segments)
            binormal = np.cross(segment1, segment2)
            binormal_norm = np.linalg.norm(binormal)
            
            if binormal_norm > 0:
                # Local curvature (approximation)
                curvature = binormal_norm / (0.5 * (norm1 + norm2))
                curvatures.append(curvature)
    
    metrics['avg_curvature'] = np.mean(curvatures) if len(curvatures) > 0 else 0
    
    # Approximate kinetic energy (proportional to length * avg_velocity^2)
    metrics['kinetic_energy'] = length * metrics['avg_velocity']**2
    
    return metrics

# -----------------------------
# Set Up the Figure and Plot
# -----------------------------
plt.style.use('dark_background')  # Use dark theme for better contrast
fig = plt.figure(figsize=(12, 8))  # Adjusted figure width
fig.patch.set_facecolor('#1C1C1C')  # Dark background

# Create grid for subplots with smaller time series plot
gs = plt.GridSpec(1, 2, width_ratios=[4, 1], figure=fig)  # Changed ratio from [2, 1] to [4, 1]

# Create 3D axis for vortex visualization
ax = fig.add_subplot(gs[0], projection='3d')
ax.set_facecolor('#1C1C1C')  # Match background color
ax.set_xlim(-L/2, L/2)
ax.set_ylim(-L/2, L/2)
ax.set_zlim(0, L/2)  # Start z at 0 for the surface
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Vortex Semi-Ring Dynamics', color='white', fontsize=14)

# Create 2D axis for time series plot (smaller now)
ax_time = fig.add_subplot(gs[1], aspect='equal')  # Set aspect to 'equal' for square axis
ax_time.set_facecolor('#1C1C1C')
ax_time.set_title('Time Series: ' + current_metric, color='white', fontsize=12)  # Smaller font
ax_time.set_xlabel('Time (s)', color='white', fontsize=10)  # Smaller font
ax_time.set_ylabel('Value', color='white', fontsize=10)  # Smaller font
ax_time.tick_params(axis='x', colors='white', labelsize=8)  # Smaller ticks
ax_time.tick_params(axis='y', colors='white', labelsize=8)  # Smaller ticks
ax_time.grid(True, linestyle='--', alpha=0.3)

# Initialize time series line
time_line, = ax_time.plot([], [], lw=1.5, color='#4A90E2')  # Thinner line

# Make gridlines and axis lines invisible
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
ax.xaxis._axinfo['tick']['inward_factor'] = 0
ax.xaxis._axinfo['tick']['outward_factor'] = 0
ax.yaxis._axinfo['tick']['inward_factor'] = 0
ax.yaxis._axinfo['tick']['outward_factor'] = 0
ax.zaxis._axinfo['tick']['inward_factor'] = 0
ax.zaxis._axinfo['tick']['outward_factor'] = 0
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Create a surface at z=0 (x-y plane)
xx, yy = np.meshgrid(np.linspace(-L/2, L/2, 10), np.linspace(-L/2, L/2, 10))
zz = np.zeros_like(xx)
surface = ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

# Initialize velocity for coloring
initial_velocity = compute_lia_velocity(filament, circulation_init, alpha_lia)
speeds = np.linalg.norm(initial_velocity, axis=1)
norm = plt.Normalize(speeds.min(), speeds.max())
cmap = plt.cm.viridis

# Plot the vortex filament with segments colored by velocity
segments = []
for i in range(len(filament)-1):
    segment = ax.plot([filament[i, 0], filament[i+1, 0]],
                      [filament[i, 1], filament[i+1, 1]],
                      [filament[i, 2], filament[i+1, 2]],
                      color=cmap(norm(speeds[i])),
                      lw=2)
    segments.append(segment[0])

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Velocity Magnitude', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Plot the vortex filament
# line, = ax.plot(filament[:, 0], filament[:, 1], filament[:, 2], 
#                 lw=2, color='#4A90E2')

# Initialize velocity arrow (will be updated in animation)
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)
        
    def draw(self, renderer):
        FancyArrowPatch.draw(self, renderer)

# Initial arrow (will be updated in animation)
arrow = Arrow3D([0, 0], [0, 0], [0, 0], 
                mutation_scale=20, lw=2, arrowstyle='-|>', color='red')
ax.add_artist(arrow)

# Add text annotation for simulation info
text_color = 'white'
info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, 
                     color=text_color, fontsize=10)

# -----------------------------
# Animation Update Function
# -----------------------------
def update(frame):
    global filament, sim_time, segments, time_data, metrics_data
    
    # Update simulation time
    sim_time += dt
    
    # Compute self-induced velocity using LIA
    lia_velocity = compute_lia_velocity(filament, circulation_init, alpha_lia)
    
    # Add external flow velocity in the direction specified by flow_direction
    external_flow = np.zeros_like(filament)
    
    # Check for zero flow direction and handle it
    flow_dir_norm = np.linalg.norm(flow_direction)
    if flow_dir_norm > 1e-10:  # Avoid division by zero
        flow_dir_normalized = flow_direction / flow_dir_norm
        for i in range(len(filament)):
            external_flow[i] = flow_velocity_init * flow_dir_normalized
    
    # Total velocity before mutual friction
    total_velocity = lia_velocity + external_flow
    
    # Store the old filament for comparison
    old_filament = filament.copy()
    
    # Update filament position
    filament = filament + total_velocity * dt
    
    # Ensure endpoints stay on the surface (z=0)
    filament[0, 2] = 0
    filament[-1, 2] = 0
    
    # Adjust filament resolution to maintain consistent spatial resolution
    # Only adjust if the filament has moved significantly
    if np.max(np.linalg.norm(filament - old_filament, axis=1)) > delta / 10:
        new_filament = adjust_filament_resolution(filament, delta, max_delta)
        
        # Only update if we got a valid filament back
        if len(new_filament) >= 3:
            filament = new_filament
    
    N = len(filament)  # Update N after resolution adjustment
    
    # Calculate velocity magnitudes for coloring
    # Recompute velocity for the adjusted filament
    lia_velocity = compute_lia_velocity(filament, circulation_init, alpha_lia)
    external_flow = np.zeros_like(filament)
    for i in range(len(filament)):
        external_flow[i] = flow_velocity_init * flow_dir_normalized
    total_velocity = lia_velocity + external_flow
    
    speeds = np.linalg.norm(total_velocity, axis=1)
    
    # Ensure we have valid min/max values for normalization
    if len(speeds) > 0:
        speed_min = speeds.min() if speeds.min() != speeds.max() else 0
        speed_max = speeds.max() if speeds.min() != speeds.max() else 1
        norm = plt.Normalize(speed_min, speed_max)
    else:
        # Fallback if we somehow have no points
        norm = plt.Normalize(0, 1)
    
    # Remove old segments
    for segment in segments:
        segment.remove()
    
    # Create new segments with updated filament
    segments = []
    for i in range(len(filament)-1):
        segment = ax.plot([filament[i, 0], filament[i+1, 0]],
                         [filament[i, 1], filament[i+1, 1]],
                         [filament[i, 2], filament[i+1, 2]],
                         color=cmap(norm(speeds[i])),
                         lw=2)
        segments.append(segment[0])
    
    # Update velocity arrow
    # Use the middle point of the filament for arrow origin
    if len(filament) > 0:
        mid_idx = len(filament) // 2
        mid_point = filament[mid_idx]
        
        # Calculate average velocity for arrow direction and magnitude
        avg_velocity = np.mean(total_velocity, axis=0)
        vel_magnitude = np.linalg.norm(avg_velocity)
        
        if vel_magnitude > 0:
            # Scale arrow length based on velocity magnitude
            arrow_scale = 2.0  # Adjust this factor to make arrow more visible
            arrow_end = mid_point + avg_velocity * arrow_scale / vel_magnitude
            
            # Update arrow
            arrow._verts3d = ([mid_point[0], arrow_end[0]], 
                              [mid_point[1], arrow_end[1]], 
                              [mid_point[2], arrow_end[2]])
            # Make sure the arrow knows which axes it belongs to
            arrow.axes = ax
    
    # Update colorbar
    sm.set_array([])
    if len(speeds) > 0:
        sm.set_clim(speed_min, speed_max)
    
    # Update info text
    info_text.set_text(f"Time: {sim_time:.2f}s\n"
                       f"Circulation: {circulation_init:.2f}\n"
                       f"Flow velocity: {flow_velocity_init:.2f}\n"
                       f"α: {alpha_mf:.3f}, α': {alpha_prime:.3f}\n"
                       f"Points: {N}")
    
    # Calculate metrics and update time series data
    metrics = calculate_metrics(filament, total_velocity)
    
    # Store time and metrics data
    time_data.append(sim_time)
    for key in metrics_data:
        metrics_data[key].append(metrics[key])
    
    # Limit the data points to max_time_points
    if len(time_data) > max_time_points:
        time_data.pop(0)
        for key in metrics_data:
            if len(metrics_data[key]) > 0:  # Check if there are elements to pop
                metrics_data[key].pop(0)
    
    # Update time series plot
    if len(time_data) > 0 and len(metrics_data[current_metric]) > 0:
        time_line.set_data(time_data, metrics_data[current_metric])
    
        # Adjust y-axis limits if needed
        min_val = min(metrics_data[current_metric])
        max_val = max(metrics_data[current_metric])
        
        # Add some padding and handle case where min=max
        if min_val == max_val:
            padding = 0.1 * abs(min_val) if min_val != 0 else 0.1
            y_min = min_val - padding
            y_max = max_val + padding
        else:
            padding = (max_val - min_val) * 0.1
            y_min = min_val - padding
            y_max = max_val + padding
        
        # Update x-axis limits to show the full time range
        x_min = time_data[0]
        x_max = time_data[-1]
        
        # For square axis, make the ranges equal
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # If ranges are very different, adjust to make them more similar
        if x_range > 0 and y_range > 0:  # Ensure positive ranges
            if x_range > y_range:
                # Expand y-range
                center = (y_min + y_max) / 2
                half_range = x_range / 2
                y_min = center - half_range
                y_max = center + half_range
            elif y_range > x_range:
                # Expand x-range
                center = (x_min + x_max) / 2
                half_range = y_range / 2
                x_min = center - half_range
                x_max = center + half_range
            
            # Set limits with a small check to avoid invalid values
            if x_min < x_max and y_min < y_max:
                ax_time.set_xlim(x_min, x_max)
                ax_time.set_ylim(y_min, y_max)
    
    # Update time series title
    ax_time.set_title(f'Time Series: {current_metric}', color='white', fontsize=12)  # Smaller font
    
    return segments + [arrow, info_text, time_line]

# Create the animation object (blit is set to False for 3D animations)
ani = animation.FuncAnimation(fig, update, frames=None, interval=50, blit=False, save_count=50)

# -----------------------------
# Slider and Button Controls
# -----------------------------
# Add sliders for interactive control
slider_color = '#2C2C2C'
slider_text_color = text_color

# Slider for mutual friction coefficient alpha
ax_alpha = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=slider_color)
alpha_slider = Slider(
    ax=ax_alpha,
    label='α (Mutual Friction)',
    valmin=0.0,
    valmax=0.5,
    valinit=alpha_mf,
    color='#4A90E2'
)
alpha_slider.label.set_color(slider_text_color)

# Slider for mutual friction coefficient alpha prime
ax_alpha_prime = plt.axes([0.25, 0.12, 0.65, 0.03], facecolor=slider_color)
alpha_prime_slider = Slider(
    ax=ax_alpha_prime,
    label="α' (Mutual Friction)",
    valmin=0.0,
    valmax=0.5,
    valinit=alpha_prime,
    color='#4A90E2'
)
alpha_prime_slider.label.set_color(slider_text_color)

# Slider for flow velocity
ax_flow = plt.axes([0.25, 0.09, 0.65, 0.03], facecolor=slider_color)
flow_slider = Slider(
    ax=ax_flow,
    label='Flow Velocity',
    valmin=-1.0,
    valmax=1.0,
    valinit=flow_velocity_init,
    color='#4A90E2'
)
flow_slider.label.set_color(slider_text_color)

# Slider for delta (spatial resolution)
ax_delta = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=slider_color)
delta_slider = Slider(
    ax=ax_delta,
    label='δ (Spatial Resolution)',
    valmin=0.01,
    valmax=0.5,
    valinit=delta,
    color='#4A90E2'
)
delta_slider.label.set_color(slider_text_color)

# Flow direction controls (x, y, z components)
ax_flow_x = plt.axes([0.25, 0.03, 0.15, 0.02], facecolor=slider_color)
flow_x_slider = Slider(
    ax=ax_flow_x,
    label='Flow Dir X',
    valmin=-1.0,
    valmax=1.0,
    valinit=flow_direction[0],
    color='#FF5555'
)
flow_x_slider.label.set_color(slider_text_color)

ax_flow_y = plt.axes([0.5, 0.03, 0.15, 0.02], facecolor=slider_color)
flow_y_slider = Slider(
    ax=ax_flow_y,
    label='Flow Dir Y',
    valmin=-1.0,
    valmax=1.0,
    valinit=flow_direction[1],
    color='#55FF55'
)
flow_y_slider.label.set_color(slider_text_color)

ax_flow_z = plt.axes([0.75, 0.03, 0.15, 0.02], facecolor=slider_color)
flow_z_slider = Slider(
    ax=ax_flow_z,
    label='Flow Dir Z',
    valmin=-1.0,
    valmax=1.0,
    valinit=flow_direction[2],
    color='#5555FF'
)
flow_z_slider.label.set_color(slider_text_color)

# Reset button
ax_reset = plt.axes([0.8, 0.01, 0.1, 0.03])
reset_button = Button(ax_reset, 'Reset', color=slider_color, hovercolor='#4A90E2')
reset_button.label.set_color(slider_text_color)

# Add button to cycle through metrics (adjusted position)
ax_cycle = plt.axes([0.9, 0.95, 0.08, 0.03])  # Smaller button
cycle_button = Button(ax_cycle, 'Cycle', color=slider_color, hovercolor='#4A90E2')  # Shorter label
cycle_button.label.set_color(slider_text_color)
cycle_button.label.set_fontsize(9)  # Smaller font

# -----------------------------
# Slider and Button Update Functions
# -----------------------------
def update_mutual_friction(val):
    global alpha_mf, alpha_prime
    alpha_mf = alpha_slider.val
    alpha_prime = alpha_prime_slider.val
    fig.canvas.draw_idle()

def update_flow(val):
    global flow_velocity_init
    flow_velocity_init = flow_slider.val
    fig.canvas.draw_idle()

def update_flow_direction(val):
    global flow_direction
    # Update the flow direction vector
    flow_direction[0] = flow_x_slider.val
    flow_direction[1] = flow_y_slider.val
    flow_direction[2] = flow_z_slider.val
    
    # Ensure the vector is not zero (default to x-direction if all zeros)
    if np.linalg.norm(flow_direction) < 1e-6:
        flow_direction = np.array([1.0, 0.0, 0.0])
        # Update sliders to match
        flow_x_slider.set_val(1.0)
        flow_y_slider.set_val(0.0)
        flow_z_slider.set_val(0.0)
    
    fig.canvas.draw_idle()

def update_delta(val):
    global delta, max_delta, dt
    old_delta = delta
    delta = delta_slider.val
    max_delta = 2 * delta
    
    # Check if the timestep is still valid with the new delta
    is_valid, dt_max = timestep_check(dt, delta, circulation_init, core_radius)
    if not is_valid:
        # Adjust dt to be safe (e.g., 90% of maximum)
        dt = 0.9 * dt_max
        print(f" Adjusting dt to {dt:.4e}")
    
    fig.canvas.draw_idle()

def reset(event):
    global filament, circulation_init, flow_velocity_init, flow_direction, alpha_mf, alpha_prime, delta, max_delta, dt, sim_time, segments, time_data, metrics_data
    
    # Reset parameters to initial values
    circulation_init = 9.97e-4  # Match the initial value at the top of the script
    flow_velocity_init = 0.0
    flow_direction = np.array([1.0, 0.0, 0.0])
    alpha_mf = 0.1
    alpha_prime = 0.05
    delta = 1e-2  # Match the initial value at the top of the script
    max_delta = 2 * delta
    sim_time = 0
    
    # Check timestep validity
    is_valid, dt_max = timestep_check(dt, delta, circulation_init, core_radius)
    if not is_valid:
        dt = 0.9 * dt_max
    
    # Reset sliders
    alpha_slider.reset()
    alpha_prime_slider.reset()
    flow_slider.reset()
    flow_x_slider.reset()
    flow_y_slider.reset()
    flow_z_slider.reset()
    delta_slider.reset()
    
    # Reinitialize vortex filament
    filament = init_vortex_semi_ring(N_points)
    
    # Recompute velocity for coloring
    velocity = compute_lia_velocity(filament, circulation_init, alpha_lia)
    speeds = np.linalg.norm(velocity, axis=1)
    norm = plt.Normalize(speeds.min() if speeds.min() != speeds.max() else 0, 
                         speeds.max() if speeds.min() != speeds.max() else 1)
    
    # Remove old segments
    for segment in segments:
        segment.remove()
    
    # Create new segments
    segments = []
    for i in range(len(filament)-1):
        segment = ax.plot([filament[i, 0], filament[i+1, 0]],
                          [filament[i, 1], filament[i+1, 1]],
                          [filament[i, 2], filament[i+1, 2]],
                          color=cmap(norm(speeds[i])),
                          lw=2)
        segments.append(segment[0])
    
    # Clear time series data
    time_data = []
    for key in metrics_data:
        metrics_data[key] = []
    
    fig.canvas.draw_idle()

def cycle_metric(event):
    global current_metric
    
    # List of available metrics
    metrics_list = list(metrics_data.keys())
    
    # Find current index and cycle to next
    current_index = metrics_list.index(current_metric)
    next_index = (current_index + 1) % len(metrics_list)
    current_metric = metrics_list[next_index]
    
    # Update plot title
    ax_time.set_title(f'Time Series: {current_metric}', color='white', fontsize=12)  # Smaller font
    
    # Update y-axis label based on the metric
    if current_metric == 'point_count':
        ax_time.set_ylabel('Number of Points', color='white', fontsize=10)  # Smaller font
    elif current_metric in ['avg_velocity', 'max_velocity']:
        ax_time.set_ylabel('Velocity Magnitude', color='white', fontsize=10)  # Smaller font
    elif current_metric == 'filament_length':
        ax_time.set_ylabel('Length', color='white', fontsize=10)  # Smaller font
    elif current_metric == 'avg_curvature':
        ax_time.set_ylabel('Curvature', color='white', fontsize=10)  # Smaller font
    elif current_metric == 'kinetic_energy':
        ax_time.set_ylabel('Energy (arb. units)', color='white', fontsize=10)  # Smaller font
    
    fig.canvas.draw_idle()

# Connect sliders and button to update functions
alpha_slider.on_changed(update_mutual_friction)
alpha_prime_slider.on_changed(update_mutual_friction)
flow_slider.on_changed(update_flow)
flow_x_slider.on_changed(update_flow_direction)
flow_y_slider.on_changed(update_flow_direction)
flow_z_slider.on_changed(update_flow_direction)
delta_slider.on_changed(update_delta)
reset_button.on_clicked(reset)
cycle_button.on_clicked(cycle_metric)

# Remove tight_layout() which is causing warnings with 3D plots
# plt.tight_layout()

# Adjust the figure layout manually instead
fig.subplots_adjust(bottom=0.25)  # Make room for sliders at the bottom

plt.show()
