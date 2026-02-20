import sys
import os
import numpy as np
import pyvista as pv
import time
from datetime import datetime

# Ensure the current directory is in the Python path for local module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import configuration
from config import CONFIG, get_frequency

# Import core framework
from core.implicit_base import Sphere
from core.mesh_container import MeshSDF
from core.analytics import calculate_volume_fraction, gibson_ashby_stiffness, estimate_mass

# Import lattice structures and grading patterns
from lattices.tpms import Gyroid, Intersection
from lattices.graded_lattice import GradedLattice, linear_z_grading

# Import visualization and export tools
from visualization.renderer import Renderer
from export.stl_exporter import save_mesh_to_stl

def create_demo_container(filename=None):
    """
    Creates a sample STL container if one doesn't exist.
    A rounded cube provides a professional-looking demonstration.
    """
    if filename is None:
        filename = CONFIG["input_stl_path"]

    if not os.path.exists(filename):
        print(f"Creating demonstration STL container: {filename}")
        # Create a simple rounded box for the demo
        box = pv.Box(bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
        # Triangulate first, then subdivide and smooth
        container = box.triangulate().subdivide(3).smooth(n_iter=100)
        container.save(filename)
    return filename

def estimate_memory_usage(resolution):
    """
    Estimates the memory usage for a voxel grid of a given resolution.
    Assumes float64 (8 bytes per point).
    """
    total_points = resolution ** 3
    memory_bytes = total_points * 8
    memory_gb = memory_bytes / (1024 ** 3)
    return memory_gb

def main():
    """
    Professional Demonstration Script: Conformal Graded Lattice Generation
    Workflow:
    1. Load Config -> 2. Load Mesh -> 3. Define Graded Lattice -> 
    4. Conformal Trimming -> 5. Analytics -> 6. Visualization -> 7. Export
    """
    print("="*60)
    print("      CADFEA: Conformal Graded Lattice Demonstration")
    print("="*60)

    # --- 1. SETUP PARAMETERS ---
    RESOLUTION = CONFIG["resolution"]
    Z_RANGE = CONFIG["z_range"]
    GYROID_FREQUENCY = get_frequency()
    
    # Memory and Resolution Safety Check
    est_memory = estimate_memory_usage(RESOLUTION)
    print(f"Estimated Voxel Grid Memory: {est_memory:.2f} GB")

    if est_memory > CONFIG["memory_threshold_gb"]:
        print(f"[MEMORY WARNING] Resolution {RESOLUTION} requires {est_memory:.2f} GB.")
        print(f"                 Threshold is set to {CONFIG['memory_threshold_gb']} GB.")
        # Automatic suggestion logic
        safe_res = int((CONFIG["memory_threshold_gb"] * (1024**3) / 8)**(1/3))
        print(f"                 Recommendation: Reduce resolution to < {safe_res}")
        # In a real app, you might ask for confirmation here:
        # choice = input("Do you want to continue? (y/n): ")
        # if choice.lower() != 'y': sys.exit()
    elif RESOLUTION > CONFIG["safety_threshold"]:
        print(f"[SAFETY CHECK] WARNING: Resolution {RESOLUTION} is above the recommended threshold ({CONFIG['safety_threshold']}).")
        print("               This may lead to extreme memory usage or crashes.")
    
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
    print(f"Applying functional grading (Z-Linear: t={CONFIG['t_min']} to t={CONFIG['t_max']})...")
    grading_pattern = linear_z_grading(
        z_min=Z_RANGE[0], 
        z_max=Z_RANGE[1], 
        t_min=CONFIG["t_min"], 
        t_max=CONFIG["t_max"]
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
    vol_frac = calculate_volume_fraction(final_part, b, resolution=CONFIG["analytics_resolution"])
    est_stiffness = gibson_ashby_stiffness(vol_frac, base_material_modulus=CONFIG["base_material_modulus"])
    
    # Calculate bounding box total volume
    total_volume = (b[1]-b[0]) * (b[3]-b[2]) * (b[5]-b[4])
    est_mass = estimate_mass(vol_frac, total_volume, CONFIG["base_material_density"])
    
    print("-" * 40)
    print(f"  Engineering Stats for Generated Lattice:")
    print(f"  - Grid Volume Fraction: {vol_frac:.2%}")
    print(f"  - Effective Stiffness:  {est_stiffness:.2f} MPa (Gibson-Ashby)")
    print(f"  - Estimated Mass:       {est_mass:.4f} g")
    print("-" * 40)

    # --- 6. VISUALIZATION ---
    # The renderer automatically handles scalar color-mapping for GradedLattice.
    print(f"Starting renderer (Resolution: {RESOLUTION})...")
    renderer = Renderer()
    
    # Calculate render bounds slightly larger than the container
    render_bounds = (b[0]-0.2, b[1]+0.2, b[2]-0.2, b[3]+0.2, b[4]-0.2, b[5]+0.2)
    
    final_mesh = renderer.render(
        surface=final_part,
        bounds=render_bounds,
        n_points=RESOLUTION,
        level=CONFIG["level_set"],
        show_section=CONFIG["show_cross_section"]
    )

    # --- 7. EXPORT ---
    # Save the resulting high-quality mesh as a binary STL.
    if final_mesh is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"{CONFIG['export_prefix']}_{timestamp}.stl"
        export_path = os.path.join(CONFIG["export_directory"], export_filename)
        print(f"Exporting to: {export_path}...")
        save_mesh_to_stl(final_mesh, export_path)
    
    print("="*60)
    print("      Demonstration Complete: Files saved to /export/")
    print("="*60)


if __name__ == "__main__":
    main()
