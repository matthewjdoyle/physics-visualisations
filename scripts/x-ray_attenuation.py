import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

"""
Notes
-----

"""

# -----------------------------
# Simulation Parameters
# -----------------------------
I0 = 2000  # Initial X-ray intensity (arbitrary units)
mu_base = 0.15  # Base attenuation coefficient (cm⁻¹) for a reference voltage (100 kVp)
thickness_max = 20  # Maximum tissue thickness to display (cm)
TARGET_ANGLE = 15  # Anode angle in degrees (typical range 12-17°)
TARGET_SIZE = 20  # Size of target visualization in mm
ELECTRON_RANGE = 0.1  # Maximum electron penetration in mm at 150 kVp

# Create a range of tissue thickness values
x = np.linspace(0, thickness_max, 200)
# Effective attenuation: initially, voltage is set to 100, so effective μ = mu_base.
I = I0 * np.exp(-mu_base * x)  # Beer–Lambert law: I = I0 * exp(-μx)

TEXT_POSITION = 10

# -----------------------------
# Set Up the Figure and Plot
# -----------------------------
plt.style.use('dark_background')  # Use dark theme for better contrast
fig = plt.figure(figsize=(15, 6))
fig.patch.set_facecolor('#1C1C1C')  # Dark background

ax1 = fig.add_subplot(121)  # Left subplot for attenuation
ax1.set_facecolor('#1C1C1C')  # Match background color
ax2 = fig.add_subplot(122, projection='3d')  # Right subplot for target
ax2.set_facecolor('#1C1C1C')  # Match background color
plt.subplots_adjust(bottom=0.35, wspace=0.3)

ax1.set_xlim(0, thickness_max)
ax1.set_ylim(0, I0)

# Display initial text info
info_text = ax1.text(0.98, 0.95, f"I₀ = {I0}\nμ₍eff₎ = {mu_base:.2f} cm⁻¹\nTube Voltage = 100 kVp", 
                    transform=ax1.transAxes, 
                    horizontalalignment='right', 
                    verticalalignment='top', 
                    fontsize=10, color='white',
                    bbox=dict(facecolor='#2C2C2C', edgecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# -----------------------------
# Interactive Widgets (Sliders and Button)
# -----------------------------
slider_color = '#4A90E2'  # Nice blue color for sliders
text_color = 'white'
slider_kwargs = dict(
    color=slider_color,
    track_color='#2C2C2C',
    handle_style={'facecolor': slider_color, 'edgecolor': 'white', 'size': 10}
)

# Slider for Attenuation Coefficient (μ_base)
ax_mu = plt.axes([0.15, 0.10, 0.65, 0.03])
mu_slider = Slider(ax_mu, "Base Attenuation μ (cm⁻¹)", 0.05, 1, valinit=mu_base, valstep=0.05, **slider_kwargs)
mu_slider.label.set_color(text_color)

# Slider for Tube Voltage (kVp)
ax_voltage = plt.axes([0.15, 0.05, 0.65, 0.03])
voltage_slider = Slider(ax_voltage, "Tube Voltage (kVp)", 10, 200, valinit=100, valstep=1, **slider_kwargs)
voltage_slider.label.set_color(text_color)

# Slider for Initial Intensity (I0)
ax_I0 = plt.axes([0.15, 0.00, 0.65, 0.03])
I0_slider = Slider(ax_I0, "Initial Intensity (I0)", 50, 2000, valinit=I0, valstep=50, **slider_kwargs)
I0_slider.label.set_color(text_color)

def beam_profile(x,y,z):
    x0 = 0
    y0 = 0
    return np.exp(-((x-x0)**2) / (y+z)**2)

def create_synthetic_volume(voltage, I0):
    # Create a synthetic 3D volume with varying densities
    size = 60
    volume = np.zeros((size, size, size))

    # Simulate a spherical object in the center
    center = size // 2
    radius = size // 4
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - center)**2 + (y - center)**2 + (z - center)**2 < radius**2:
                    volume[x, y, z] = 1  # Assign a density value
                    volume[x,y,z] -= beam_profile(x,y,z)
    
    # Adjust the volume based on voltage and intensity
    volume *= (voltage /50) * (I0 / 1000)
    
    return volume

def create_fixed_volume(voltage, I0):
    # Create a synthetic 3D volume with fixed density (for the underlying object)
    size = 50
    volume = np.zeros((size, size, size))

    center = 30
    radius = size // 4

    
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - center)**2 + (y - center)**2 + (z - center)**2 < radius**2:
                    volume[x, y, z] = 1  # Assign a density value
                    volume[x,y,z] -= beam_profile(x,y,z)
    # Adjust the volume based on intensity and voltage (like the main object)
    # but with a fixed attenuation coefficient
    volume *= (voltage / 50) * (I0 / 1000)
    
    return volume

def update_target_plot(voltage, I0, mu_eff):
    ax2.clear()
    
    # Create a synthetic volume for the main object
    volume = create_synthetic_volume(voltage, I0)
    
    # Create a fixed volume for the underlying object, passing intensity and voltage
    fixed_volume = create_fixed_volume(voltage, I0)
    
    # Calculate opacity based on attenuation coefficient
    # Higher attenuation = lower opacity (as X-rays are more attenuated)
    opacity_main = max(0.2, 1.0 - mu_eff * 0.8)  # Scale opacity inversely with attenuation
    
    try:
        # Render the fixed object first (always visible with constant opacity)
        verts_fixed, faces_fixed, _, _ = measure.marching_cubes(fixed_volume, level=0.4)
        ax2.plot_trisurf(verts_fixed[:, 0], verts_fixed[:, 1], faces_fixed, verts_fixed[:, 2],
                         cmap='viridis', lw=0.5, alpha=0.6)  # Fixed opacity
        
        # Render the main object with variable opacity
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
        ax2.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                         cmap='plasma', lw=1, alpha=opacity_main)
    except:
        ax2.text(0,0,0, "No visible target", color='red', fontsize=12)
    
    # Add an arrow to show incident beam direction
    # ax2.quiver(0, 0, 30, 0, 0, -20, color='cyan', arrow_length_ratio=0.1, linewidth=2)
    
    # Annotations and labels
    ax2.set_xlabel('X', color='white')
    ax2.set_ylabel('Y', color='white')
    ax2.set_zlabel('Z', color='white')
    ax2.set_title(f'X-ray Scan Simulation\n{voltage:.0f} kVp, {I0:.0f} mA', color='white')
    
    # Set consistent view angle
    ax2.view_init(elev=30, azim=45)
    ax2.set_box_aspect([1,1,1])  # Ensure aspect ratio is consistent
    ax2.set_axis_off()

def update(val):
    # Read current slider values
    mu_val = mu_slider.val           # Base μ
    voltage = voltage_slider.val     # Tube voltage in kVp
    I0 = I0_slider.val               # Initial Intensity
    tmax = thickness_max
    
    # Calculate effective attenuation coefficient using a simple model:
    mu_eff = mu_val * (100 / voltage)**0.5
    
    # Update x-range based on new maximum thickness
    x = np.linspace(0, tmax, 200)
    # Calculate transmitted intensity with the effective μ
    I = I0 * np.exp(-mu_eff * x)
    
    l.set_xdata(x)
    l.set_ydata(I)
    
    # Update info text with current parameters and effective μ
    info_text.set_text(f"I₀ = {I0}\nμ₍eff₎ = {mu_eff:.2f} cm⁻¹\nTube Voltage = {voltage} kVp")
    
    # Update target visualization with the effective attenuation coefficient
    update_target_plot(voltage, I0, mu_eff)
    
    fig.canvas.draw_idle()

# Connect sliders to update function
mu_slider.on_changed(update)
voltage_slider.on_changed(update)
I0_slider.on_changed(update)

# Button to add a fixed line
ax_add_line = plt.axes([0.15, 0.15, 0.15, 0.04])
add_line_button = Button(ax_add_line, "Add Fixed Line", color='#2C2C2C', hovercolor='#4A90E2')
add_line_button.label.set_color(text_color)

def add_fixed_line(event):
    current_x = l.get_xdata()
    current_y = l.get_ydata()
    line_color = np.random.rand(3,)  # Generate a random color
    ax1.plot(current_x, current_y, lw=1, color=line_color)

    # Retrieve current parameters
    mu_val = mu_slider.val
    voltage = voltage_slider.val
    I0 = I0_slider.val

    global TEXT_POSITION
    if len(ax1.lines) == 1:
        TEXT_POSITION = 10  # Start at position 10 for the first line
    else:
        TEXT_POSITION += 3  # Increment position by 3 for each new line
        if TEXT_POSITION >= len(current_x) - 3:  # Ensure the position does not exceed the array length
            TEXT_POSITION = 10  # Reset if it does

    # Calculate position for the text based on TEXT_POSITION
    text_x = current_x[TEXT_POSITION]
    text_y = current_y[TEXT_POSITION]

    # Add text annotation near the end of the line
    label = f"I₀ = {I0}, μ = {mu_val:.2f}, V = {voltage} kVp"
    ax1.text(text_x, text_y, label, color=line_color, verticalalignment='bottom', fontsize=8)

    fig.canvas.draw_idle()

add_line_button.on_clicked(add_fixed_line)

# Button to reset to initial values
ax_reset = plt.axes([0.8, 0.15, 0.1, 0.04])
reset_button = Button(ax_reset, "Reset", color='#2C2C2C', hovercolor='#4A90E2')
reset_button.label.set_color(text_color)

# Initialize both plots
l, = ax1.plot(x, I, lw=2, color='blue')
ax1.set_xlabel('Tissue Thickness (cm)', color='white')
ax1.set_ylabel('Transmitted Intensity', color='white')
ax1.set_title('X-ray Attenuation in Tissue', color='white')

# Initialize target visualization
update_target_plot(100, I0, mu_base)

def reset(event):
    mu_slider.reset()
    voltage_slider.reset()
    I0_slider.reset()
    ax2.clear()
    update_target_plot(100, I0, mu_base)
    

reset_button.on_clicked(reset)

plt.show()
