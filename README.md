# ImplicitLattice: A Function-Defined Modeling Engine

ImplicitLattice is a professional-grade Python framework for generating complex, functionally graded lattice structures using implicit modeling and Signed Distance Fields (SDF). It is designed for additive manufacturing (3D Printing) applications where structural optimization and conformal geometry are critical.

## 🚀 Features

- **TPMS (Triply Periodic Minimal Surfaces):** Native support for Gyroid, Schwarz P, and Diamond lattices defined by periodic trigonometric functions.
- **Functional Grading:** Seamlessly vary lattice thickness across any axis (e.g., Z-linear grading) using continuous mathematical functions.
- **MeshSDF Integration:** Treat any standard STL mesh as an implicit boundary, allowing for perfect conformal trimming of lattices to complex parts.
- **Dual-View Visualization:** Built-in support for full 3D rendering and internal cross-section inspection using PyVista.
- **STL Export:** High-fidelity binary STL export with automatic timestamping for version control.

## 🏗️ Architecture

The engine is built on a modular, object-oriented architecture:

- `core/`: Foundational logic for implicit surfaces, MeshSDF voxelization, and engineering analytics.
- `lattices/`: Library of TPMS primitives and grading decorators.
- `visualization/`: Renderer based on Marching Cubes and PyVista for high-performance 3D display.
- `export/`: Tools for converting implicit fields into manufacturable mesh files.
- `config.py`: Centralized configuration to separate design parameters from engine logic.

## 📊 Engineering Analytics

ImplicitLattice goes beyond geometry by providing real-time engineering feedback:

- **Gibson-Ashby Stiffness Prediction:** Estimates the effective Young's Modulus ($E^*$) of the lattice based on the volume fraction and base material properties.
- **Volume Fraction Calculation:** High-accuracy grid-based and Monte Carlo integration to determine material-to-void ratios.
- **Mass Estimation:** Predicts the final part weight based on the calculated volume and material density ($ho$).

## 🛠️ Quick Start

### 1. Configure Your Design
You can change the entire lattice design without touching the core code. Simply modify `config.py`:

```python
CONFIG = {
    "resolution": 120,            # Mesh fidelity
    "cell_size": 1.2,             # Lattice unit cell size
    "t_min": 0.15,                # Thin region thickness
    "t_max": 0.60,                # Thick region thickness
    "input_stl_path": "your_mesh.stl"
}
```

### 2. Generate and Validate
Run the main application to generate the field, perform analytics, and view the results:

```bash
python main_app.py
```

## 📦 Requirements

- `numpy`
- `pyvista`
- `scikit-image`
- `scipy`

---
*Developed for CADFEA: Advanced Implicit Modeling Research.*
