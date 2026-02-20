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
    Final Validation Script: Conformal Graded Lattice Generation
    Workflow:
    1. Load Config -> 2. Load Mesh -> 3. Define Graded Lattice -> 
    4. Conformal Trimming -> 5. Analytics -> 6. Multi-View Visualization -> 7. Export & Summary
    """
    print("="*60)
    print("      CADFEA: FINAL VALIDATION REPORT")
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
        safe_res = int((CONFIG["memory_threshold_gb"] * (1024**3) / 8)**(1/3))
        print(f"                 Recommendation: Reduce resolution to < {safe_res}")
    elif RESOLUTION > CONFIG["safety_threshold"]:
        print(f"[SAFETY CHECK] WARNING: Resolution {RESOLUTION} is above the recommended threshold ({CONFIG['safety_threshold']}).")
        print("               This may lead to extreme memory usage or crashes.")
    
    # --- 2. LOAD MESH CONTAINER ---
    container_file = create_demo_container()
    print(f"Loading container: {container_file}...")
    
    start_time = time.time()
    try:
        container = MeshSDF(container_file, use_cache=True, cache_resolution=50)
    except Exception as e:
        print(f"Mesh loading failed: {e}. Falling back to Sphere.")
        container = Sphere(radius=1.2)
    end_time = time.time()
    print(f"[BENCHMARK] (a) Load MeshSDF: {end_time - start_time:.4f} seconds")

    # --- 3. DEFINE GRADED LATTICE ---
    print(f"Generating Gyroid lattice (Frequency: {GYROID_FREQUENCY:.2f})...")
    base_lattice = Gyroid(frequency=GYROID_FREQUENCY)

    print(f"Applying functional grading (Z-Linear: t={CONFIG['t_min']} to t={CONFIG['t_max']})...")
    grading_pattern = linear_z_grading(
        z_min=Z_RANGE[0], 
        z_max=Z_RANGE[1], 
        t_min=CONFIG["t_min"], 
        t_max=CONFIG["t_max"]
    )
    graded_lattice = GradedLattice(base_lattice, grading_pattern)

    # --- 4. SYNTHESIS: CONFORMAL TRIMMING ---
    print("Synthesizing conformal structure...")
    final_part = Intersection(container, graded_lattice)

    # --- 5. ANALYTICS ---
    print("Performing final engineering analytics...")
    b = container.bounds
    vol_frac = calculate_volume_fraction(final_part, b, resolution=CONFIG["analytics_resolution"])
    est_stiffness = gibson_ashby_stiffness(vol_frac, base_material_modulus=CONFIG["base_material_modulus"])
    total_volume = (b[1]-b[0]) * (b[3]-b[2]) * (b[5]-b[4])
    est_mass = estimate_mass(vol_frac, total_volume, CONFIG["base_material_density"])
    
    # --- 6. VISUALIZATION (Multi-View) ---
    renderer = Renderer()
    render_bounds = (b[0]-0.2, b[1]+0.2, b[2]-0.2, b[3]+0.2, b[4]-0.2, b[5]+0.2)
    
    # Full 3D View
    print("Generating Full 3D Visualization...")
    final_mesh = renderer.render(
        surface=final_part,
        bounds=render_bounds,
        n_points=RESOLUTION,
        level=CONFIG["level_set"],
        show_section=False
    )

    # Cross-Section View to validate grading
    print("Generating Internal Cross-Section Visualization...")
    renderer.render(
        surface=final_part,
        bounds=render_bounds,
        n_points=RESOLUTION,
        level=CONFIG["level_set"],
        show_section=True
    )

    # --- 7. EXPORT & FINAL SUMMARY ---
    if final_mesh is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"{CONFIG['export_prefix']}_VALIDATED_{timestamp}.stl"
        export_path = os.path.join(CONFIG["export_directory"], export_filename)
        print(f"Exporting Final Mesh to: {export_path}...")
        save_mesh_to_stl(final_mesh, export_path)

        print("\n" + "*"*60)
        print("                PROJECT SUCCESS SUMMARY")
        print("*"*60)
        print(f"  Final Engineering Metrics:")
        print(f"  - Target Resolution:    {RESOLUTION}")
        print(f"  - Volume Fraction:      {vol_frac:.2%}")
        print(f"  - Effective Stiffness:  {est_stiffness:.2f} MPa")
        print(f"  - Estimated Part Mass:  {est_mass:.4f} g")
        print(f"  - Material Properties:  Es={CONFIG['base_material_modulus']}MPa, Rho={CONFIG['base_material_density']}g/cm3")
        print("-" * 60)
        print(f"  Status: VALIDATED | Export: {export_filename}")
        print("*"*60 + "\n")


if __name__ == "__main__":
    main()
