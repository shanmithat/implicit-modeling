# ImplicitLattice: A Function-Defined Modeling Engine

ImplicitLattice is a professional-grade Python framework for generating complex, functionally graded lattice structures using implicit modeling and Signed Distance Fields (SDF). It is designed for additive manufacturing (3D Printing) applications where structural optimization and conformal geometry are critical.

## Features

- **TPMS (Triply Periodic Minimal Surfaces):** Native support for Gyroid, Schwarz P, and Diamond lattices defined by periodic trigonometric functions.
- **Functional Grading:** Seamlessly vary lattice thickness across any axis (e.g., Z-linear grading) or from an **Attractor Point** (localized reinforcement) using continuous mathematical functions.
- **MeshSDF Integration:** Treat any standard STL mesh as an implicit boundary, allowing for perfect conformal trimming of lattices to complex parts.
- **Dual-View Visualization:** Built-in support for full 3D rendering and internal cross-section inspection using PyVista.
- **STL Export:** High-fidelity binary STL export with automatic timestamping for version control.

## Architecture

The engine is built on a modular, object-oriented architecture:

- `core/`: Foundational logic for implicit surfaces, MeshSDF voxelization, and engineering analytics.
- `lattices/`: Library of TPMS primitives and grading decorators.
- `visualization/`: Renderer based on Marching Cubes and PyVista for high-performance 3D display.
- `export/`: Tools for converting implicit fields into manufacturable mesh files.
- `config.py`: Centralized configuration to separate design parameters from engine logic.

## Mathematical Foundations

### Mathematical Framework
This engine evaluates continuous field equations directly inside a hardware fragment shader pipeline, completely eliminating discrete mesh serialization overhead (STL/OBJ polygon generation loops).

- **Representation:** Functional Implicit Fields via Raymarching Volumetric Signatures.
- **Compute Layer:** WebGL 2.0 / GLSL ES 3.0 context executing out-of-core evaluation on the GPU per-pixel.

### Triply Periodic Minimal Surfaces (TPMS)
The core of this engine relies on TPMS, which are surfaces that are periodic in three independent directions and have a mean curvature of zero. They are modeled as implicit level-set equations $F(x, y, z) = 0$.

#### 1. The Gyroid
The Gyroid is a continuous, non-self-intersecting structure defined by:
$$ \sin(\omega x)\cos(\omega y) + \sin(\omega y)\cos(\omega z) + \sin(\omega z)\cos(\omega x) = t $$
Where $\omega = \frac{2\pi}{L}$ defines the spatial frequency based on the unit cell size $L$, and $t$ acts as the iso-level dictating thickness.

#### 2. The Diamond (Schwarz D)
The Diamond structure provides high stiffness and is defined as:
$$ \sin(\omega x)\sin(\omega y)\sin(\omega z) + \sin(\omega x)\cos(\omega y)\cos(\omega z) + \cos(\omega x)\sin(\omega y)\cos(\omega z) + \cos(\omega x)\cos(\omega y)\sin(\omega z) = t $$

### The Marching Cubes Algorithm
To render and export these implicit fields, ImplicitLattice utilizes the **Marching Cubes (MC)** algorithm via `PyVista/VTK`:
1. **Grid Evaluation:** The 3D bounding box is subdivided into a discrete voxel grid at the specified `resolution`.
2. **Scalar Field Generation:** The TPMS equation (plus any grading/boolean logic) is evaluated at every vertex of the grid to generate a scalar distance field.
3. **Isosurface Extraction:** MC examines each group of 8 adjacent vertices (a cube). By checking which vertices are inside the surface ($>0$) and which are outside ($<0$), the algorithm references a lookup table of 256 possible polygon configurations.
4. **Vertex Interpolation:** The exact positions of the triangles are determined by linear interpolation along the grid edges, resulting in a smooth, high-fidelity mesh suitable for STL export.

## Engineering Analytics

ImplicitLattice goes beyond geometry by providing real-time engineering feedback:

- **Gibson-Ashby Stiffness Prediction:** Estimates the effective Young's Modulus ($E^*$) of the lattice based on the volume fraction and base material properties.
- **Volume Fraction Calculation:** High-accuracy grid-based and Monte Carlo integration to determine material-to-void ratios.
- **Mass Estimation:** Predicts the final part weight based on the calculated volume and material density ($
ho$).

## Quick Start

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

## Requirements

- `numpy`
- `pyvista`
- `scikit-image`
- `scipy`

---
*Developed for CADFEA: Advanced Implicit Modeling Research.*
