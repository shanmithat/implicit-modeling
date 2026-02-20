import sys
import os
import numpy as np
import pyvista as pv
import time
from datetime import datetime

# Ensure the current directory is in the Python path for local module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import core framework
from core.implicit_base import Sphere
from core.mesh_container import MeshSDF
from core.analytics import estimate_volume_fraction, predict_stiffness

# Import lattice structures and grading patterns
from lattices.tpms import Gyroid, Intersection
from lattices.graded_lattice import GradedLattice, linear_z_grading

# Import visualization and export tools
from visualization.renderer import Renderer
from export.stl_exporter import save_mesh_to_stl

def create_demo_container(filename="demo_container.stl"):
    """
    Creates a sample STL container if one doesn't exist.
    A rounded cube provides a professional-looking demonstration.
    """
    if not os.path.exists(filename):
        print(f"Creating demonstration STL container: {filename}")
        # Create a simple rounded box for the demo
        box = pv.Box(bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
        # Triangulate first, then subdivide and smooth
        container = box.triangulate().subdivide(3).smooth(n_iter=100)
        container.save(filename)
    return filename

def main():
    """
    Professional Demonstration Script: Conformal Graded Lattice Generation
    Workflow:
    1. Define Container -> 2. Generate Lattice -> 3. Apply Grading -> 
    4. Conformal Trimming -> 5. Visualization -> 6. Export
    """
    print("="*60)
    print("      CADFEA: Conformal Graded Lattice Demonstration")
    print("="*60)

    # --- 1. SETUP PARAMETERS ---
    RESOLUTION = 100       # High-resolution grid for mesh generation
    CELL_SIZE = 0.8        # Physical size of a single lattice cell
    Z_RANGE = (-1.0, 1.0)  # Vertical range for functional grading
    
    # Resolution Safety Check
    if RESOLUTION > 150:
        print(f"[SAFETY CHECK] WARNING: Resolution {RESOLUTION} is above the recommended threshold (150).")
        print("               This may lead to extreme memory usage or crashes.")
    
    # Frequency = 2*PI / Cell_Size
    GYROID_FREQUENCY = (2 * np.pi) / CELL_SIZE

    # --- 2. LOAD MESH CONTAINER ---
    # We use MeshSDF to treat a 3D model as an implicit boundary.
    container_file = create_demo_container()
    print(f"Loading container: {container_file}...")
    
    start_time = time.time()
    try:
        # use_cache=True enables high-speed voxelization for evaluation
        container = MeshSDF(container_file, use_cache=True, cache_resolution=50)
    except Exception as e:
        print(f"Mesh loading failed: {e}. Falling back to Sphere.")
        container = Sphere(radius=1.2)
    end_time = time.time()
    print(f"[BENCHMARK] (a) Load MeshSDF: {end_time - start_time:.4f} seconds")

    # --- 3. DEFINE GRADED LATTICE ---
    print(f"Generating Gyroid lattice (Frequency: {GYROID_FREQUENCY:.2f})...")
    base_lattice = Gyroid(frequency=GYROID_FREQUENCY)

    # Define linear functional grading: 
    # t=0.1 (Thin) at Z=-1.0 to t=0.5 (Thick) at Z=1.0
    print("Applying functional grading (Z-Linear)...")
    grading_pattern = linear_z_grading(
        z_min=Z_RANGE[0], 
        z_max=Z_RANGE[1], 
        t_min=0.1, 
        t_max=0.5
    )
    graded_lattice = GradedLattice(base_lattice, grading_pattern)

    # --- 4. SYNTHESIS: CONFORMAL TRIMMING ---
    # Intersection operation clips the infinite lattice to the container's bounds.
    print("Synthesizing conformal structure...")
    final_part = Intersection(container, graded_lattice)

    # --- 5. ANALYTICS ---
    # Perform engineering calculations before rendering
    print("Performing engineering analytics...")
    b = container.bounds
    vol_frac = estimate_volume_fraction(final_part, b, samples=50000)
    est_stiffness = predict_stiffness(vol_frac, base_material_modulus=2500) # MPa (Aluminum-ish)
    
    print("-" * 40)
    print(f"  Engineering Stats for Generated Lattice:")
    print(f"  - Estimated Volume Fraction: {vol_frac:.2%}")
    print(f"  - Predicted Stiffness (E*):  {est_stiffness:.2f} MPa")
    print("-" * 40)

    # --- 6. VISUALIZATION ---
    # The renderer automatically handles scalar color-mapping for GradedLattice.
    print(f"Starting renderer (Resolution: {RESOLUTION})...")
    renderer = Renderer()
    
    # Calculate render bounds slightly larger than the container
    render_bounds = (b[0]-0.2, b[1]+0.2, b[2]-0.2, b[3]+0.2, b[4]-0.2, b[5]+0.2)
    
    # We will modify the renderer to return timing info or we will time it here.
    # For now, let's time the whole process.
    final_mesh = renderer.render(
        surface=final_part,
        bounds=render_bounds,
        n_points=RESOLUTION,
        level=0.0
    )

    # --- 7. EXPORT ---
    # Save the resulting high-quality mesh as a binary STL.
    if final_mesh is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"graded_structure_{timestamp}.stl"
        export_path = os.path.join("export", export_filename)
        print(f"Exporting to: {export_path}...")
        save_mesh_to_stl(final_mesh, export_path)
    
    print("="*60)
    print("      Demonstration Complete: Files saved to /export/")
    print("="*60)


if __name__ == "__main__":
    main()
