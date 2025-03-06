# physics-visualisations

![Physics Visualisations Banner](banner_t.gif)

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

2. Install dependencies:
   ```
   pip install numpy matplotlib scipy scikit-image
   ```
   
   Alternatively, if you have Anaconda installed, these packages should already be available.

3. Run any simulation:
   ```
   python scripts/ideal_gas_2d.py
   ```

## Simulations

### Ideal Gas Simulations

- **ideal_gas_2d.py**: An interactive 2D simulation of an ideal gas in a container. Visualises particle motion, collisions, and thermodynamic properties. Features adjustable parameters for temperature, volume, and particle count to demonstrate gas laws.
  
- **ideal_gas_3d.py**: A 3D extension of the ideal gas simulation, showing particles moving and colliding in a cubic container. Includes interactive controls to modify simulation parameters and observe how changes affect the system's behavior.

### X-Ray Physics

- **x_ray_attenuation.py**: Simulates X-ray attenuation through different materials and densities. Visualises how X-ray intensity changes as it passes through matter, with adjustable parameters for voltage and initial intensity. Includes 3D visualisation of an X-ray target.

### Visualization Tools

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

