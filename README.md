# physics-visualisations

![Physics Visualisations Banner](Banner_t.gif)

Created by Matthew J. Doyle.

## Overview

This repository contains a set of interactive physics simulations designed for educational purposes. Each simulation visualises different physics phenomena with adjustable parameters, allowing users to explore and understand complex physical concepts through direct manipulation and observation.

## Installation

### Option 1: Executable Files (Windows)

1. Navigate to the `Application` folder in this repository
2. Download the `.exe` file for the simulation you want to use
3. Run the executable on your Windows computer

### Option 2: Python Scripts

To run the simulations from source:

1. Clone this repository:
   ```
   git clone https://github.com/matthewjdoyle/physics-visualisations.git
   cd physics-visualisations
   ```

2. Install dependencies (if required):
   ```
   pip install numpy matplotlib scipy scikit-image
   ```
   
   If you have Anaconda installed, these packages should already be available.

3. Run any simulation:
   ```
   python scripts/ideal_gas_2d.py
   ```

## Simulations


- **ideal_gas_2d.py**: An interactive 2D simulation of an ideal gas in a container. Visualises particle motion, collisions, and thermodynamic properties. Features adjustable parameters for temperature, volume, and particle count to demonstrate gas laws.
  
- **ideal_gas_3d.py**: A 3D extension of the ideal gas simulation, showing particles moving and colliding in a cubic container. Includes interactive controls to modify simulation parameters and observe how changes affect the system's behavior.

<<<<<<< Updated upstream
- **x_ray_attenuation.py**: Simulates X-ray attenuation through different materials and densities. Visualises how X-ray intensity changes as it passes through matter, with adjustable parameters for voltage and initial intensity. Includes 3D visualisation of an X-ray target.

## Other scripts
=======
### Medical Physics

- **x_ray_attenuation.py**: Simulates X-ray attenuation through different materials and densities. Visualises how X-ray intensity changes as it passes through matter, with adjustable parameters for voltage and initial intensity. Includes 3D visualisation of an X-ray target.


### Gravitational Systems

- **nbody_simulation.py**: A 3D N-body gravitational simulation that models the motion of multiple bodies under mutual gravitational attraction. Features interactive controls for the gravitational constant, simulation time-step, and collision detection. Visualises orbital parameters and system energy, allowing exploration of stable and chaotic orbital configurations.

### Wave Physics

- **wave_propagation.py**: Simulates 2D wave propagation in various media with interactive controls for wave parameters (speed, damping, frequency) and medium properties (uniform, gradient, random, lens, barrier). Demonstrates wave phenomena such as reflection, refraction, diffraction, and interference through different boundary conditions and wave source types.

### Vortex Filaments 

- **vortex_semi-ring.py**: Simulates the dynamics of a quantum vortex semi-ring in superfluid helium-4 using the local induction approximation (LIA). Features interactive controls for mutual friction coefficients and external flow velocity. Visualises the 3D motion of the vortex filament, including surface interactions and reconnection events. Includes real-time calculation of filament length, curvature, and kinetic energy.

### Chaos

- **double_pendulum_interactive.py**: An advanced double pendulum simulation with interactive controls for pendulum parameters (lengths, masses, gravity, damping) and initial conditions. Features real-time visualisation of the pendulum motion, energy conservation, and phase space trajectories. Demonstrates chaotic behavior in classical mechanics.

- **lorenz_attractor.py**: An interactive simulation of the Lorenz attractor, a classic example of a chaotic system. Features controls for system parameters $(\sigma, \rho, \beta)$ and visualisation options. Includes real-time calculation of Lyapunov exponents, fixed points, and stability analysis. Demonstrates the butterfly effect.

## Other scripts

### Media Tools
>>>>>>> Stashed changes

- **banner.py**: Creates an animated MP4 banner (shown at the top of this file) featuring four different physics simulations: a double pendulum system, an N-body gravity simulation, wave propagation patterns, and a Lorenz attractor.

## Usage

Each simulation includes interactive controls:

- **Sliders**: Adjust parameters like temperature, volume, or particle count
- **Buttons**: Reset the simulation or trigger specific events
- **Interactive plots**: Some simulations allow direct interaction with the visualization

## Features

- Real-time physics simulations with accurate mathematical models
- Interactive parameter adjustment to explore different physical conditions
- Visual representation of abstract physics concepts
- Educational tool for students and educators

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- SciPy
- scikit-image (for x-ray simulation)


## Contact

For questions, suggestions, or collaborations, please open an issue on this repository, or email enquire.matthewjdoyle@gmail.com

---

*Note: These simulations are designed for educational purposes and may use simplified models of physical phenomena.*

---

[Apache License](LICENSE)

