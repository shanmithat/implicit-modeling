import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

class Renderer:
    """
    Handles the visualization of implicit surfaces.
    """
    def render(self, surface, bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5), n_points=100, level=0.0):
        """
        Generates a mesh from an implicit surface and displays it.
        
        Args:
            surface (ImplicitSurface): The implicit surface to render.
            bounds (tuple): A tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
                            defining the volume to mesh.
            n_points (int): The number of points along each axis for the grid.
            level (float): The iso-level to extract the surface at.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        
        # Create a grid of points
        x, y, z = np.mgrid[xmin:xmax:n_points*1j, 
                           ymin:ymax:n_points*1j, 
                           zmin:zmax:n_points*1j]
        
        # Evaluate the implicit function on the grid
        print("Evaluating implicit function on the grid...")
        volume_data = surface.evaluate(x, y, z)
        print("Evaluation complete.")

        # Generate the mesh using marching cubes
        print("Generating mesh using marching cubes...")
        try:
            spacing = ((xmax-xmin)/(n_points-1), (ymax-ymin)/(n_points-1), (zmax-zmin)/(n_points-1))
            verts, faces, _, _ = marching_cubes(volume_data, level=level, spacing=spacing)
            # Correcting vertex positions to match the specified bounds
            verts += [xmin, ymin, zmin]
        except (ValueError, RuntimeError) as e:
            print(f"Marching cubes failed: {e}")
            print("This often means the surface does not intersect the volume. Try adjusting bounds or level.")
            return None
        print("Mesh generation complete.")

        if verts.size == 0 or faces.size == 0:
            print("Warning: No surface generated. The level set might be empty within the given bounds.")
            return None

        # Create a PyVista mesh
        # The faces array from marching_cubes needs to be padded for PyVista
        faces_padded = np.hstack((np.full((faces.shape[0], 1), 3), faces))
        mesh = pv.PolyData(verts, faces_padded)

        # Calculate scalars for coloring if it's a graded lattice
        scalars = None
        thickness_func = None
        
        # Try to find a thickness function in the surface or its sub-surfaces
        if hasattr(surface, 'thickness_function'):
            thickness_func = surface.thickness_function
        elif hasattr(surface, 'surfaces'):
            for s in surface.surfaces:
                if hasattr(s, 'thickness_function'):
                    thickness_func = s.thickness_function
                    break
        
        if thickness_func:
            print("Calculating thickness scalars for coloring...")
            scalars = thickness_func(verts[:, 0], verts[:, 1], verts[:, 2])
            mesh.point_data["Thickness"] = scalars

        # Check if a display is available
        if not pv.system_supports_plotting():
            print("3D plotting is not supported on this system. Cannot open interactive window.")
            return mesh

        # Create a plotter and display the mesh
        print("Opening visualization window...")
        plotter = pv.Plotter()
        if scalars is not None:
            plotter.add_mesh(mesh, scalars="Thickness", cmap='viridis', show_edges=True, smooth_shading=True)
        else:
            plotter.add_mesh(mesh, color='lightblue', show_edges=True, smooth_shading=True)
        
        plotter.show_grid()
        plotter.add_title("Implicit Surface Visualization")
        
        # Use interactive=True to ensure the window stays open
        plotter.show(interactive=True)
        print("Visualization window closed.")
        
        return mesh
