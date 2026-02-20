import numpy as np

# CADFEA Configuration Settings
# Modify these values to change the design without touching the core logic.

CONFIG = {
    # --- Grid and Resolution ---
    "resolution": 100,               # High-resolution grid for mesh generation
    "safety_threshold": 150,         # Warn if resolution is above this value
    
    # --- Lattice Geometry ---
    "cell_size": 0.8,                # Physical size of a single lattice cell (mm/cm)
    "z_range": (-1.0, 1.0),          # Vertical range for functional grading
    
    # --- Functional Grading ---
    "t_min": 0.1,                    # Minimum thickness (at z_min)
    "t_max": 0.5,                    # Maximum thickness (at z_max)
    
    # --- Input / Output ---
    "input_stl_path": "demo_container.stl",
    "export_directory": "export",
    "export_prefix": "graded_structure",
    
    # --- Analytics ---
    "mc_samples": 50000,             # Number of Monte Carlo samples for volume fraction
    "base_material_modulus": 2500.0, # MPa (e.g., Aluminum-ish material)
    
    # --- Visualization ---
    "show_edges": True,
    "colormap": "viridis",
    "level_set": 0.0                 # The iso-level to extract the surface at
}

def get_frequency():
    """Calculates frequency based on cell size."""
    return (2 * np.pi) / CONFIG["cell_size"]
