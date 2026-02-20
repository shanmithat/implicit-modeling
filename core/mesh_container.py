import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from core.implicit_base import ImplicitSurface

try:
    import vtk
except ImportError:
    vtk = None

class MeshSDF(ImplicitSurface):
    """
    Wraps a 3D mesh (STL/OBJ) to behave as an implicit surface using a Signed Distance Field (SDF).
    Includes a voxelization mode for high-performance evaluation via interpolation.
    """
    def __init__(self, file_path, use_cache=True, cache_resolution=50):
        """
        Initializes the MeshSDF.
        
        Args:
            file_path (str): Path to the STL/mesh file.
            use_cache (bool): If True, pre-computes a distance grid for fast lookups.
            cache_resolution (int): Resolution of the pre-computed distance grid.
        """
        if vtk is None:
            raise ImportError("VTK is required for MeshSDF. Please install it via pip.")
            
        self.mesh = pv.read(file_path)
        if not isinstance(self.mesh, pv.PolyData):
            self.mesh = self.mesh.extract_surface()
            
        self.implicit_distance = vtk.vtkImplicitPolyDataDistance()
        self.implicit_distance.SetInput(self.mesh)
        
        self.use_cache = use_cache
        self._interpolator = None
        
        if use_cache:
            self._precompute_grid(cache_resolution)

    def _precompute_grid(self, res):
        """Voxelizes the mesh into a distance grid for interpolation."""
        print(f"Voxelizing mesh SDF at resolution {res}...")
        b = self.mesh.bounds
        # Expand bounds slightly to ensure coverage
        padding = 0.1 * max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
        x_range = np.linspace(b[0]-padding, b[1]+padding, res)
        y_range = np.linspace(b[2]-padding, b[3]+padding, res)
        z_range = np.linspace(b[4]-padding, b[5]+padding, res)
        
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        
        # Compute exact distances for the grid points
        dists = np.array([self.implicit_distance.EvaluateFunction(p) for p in pts])
        dists = dists.reshape(res, res, res)
        
        self._interpolator = RegularGridInterpolator((x_range, y_range, z_range), dists, 
                                                   bounds_error=False, fill_value=padding)
        print("Voxelization complete.")

    def evaluate(self, x, y, z):
        """Evaluates distance. Negative is inside, positive is outside."""
        if self.use_cache and self._interpolator:
            # Fast interpolation lookup
            pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
            return self._interpolator(pts).reshape(x.shape)
        else:
            # Slow, exact VTK calculation
            pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
            return np.array([self.implicit_distance.EvaluateFunction(p) for p in pts]).reshape(x.shape)

    @property
    def bounds(self):
        return self.mesh.bounds
