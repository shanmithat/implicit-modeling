import os
import pyvista as pv

def save_mesh_to_stl(mesh, filename):
    """
    Saves a PyVista PolyData mesh to a binary STL file.
    
    Args:
        mesh (pv.PolyData): The mesh object to export.
        filename (str): The target file path for the STL file.
    """
    if not isinstance(mesh, pv.PolyData):
        raise TypeError("Expected a pyvista.PolyData object.")

    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # Ensure the filename ends with .stl
    if not filename.lower().endswith('.stl'):
        filename += '.stl'

    try:
        # PyVista's save method handles binary STL by default for .stl extension
        mesh.save(filename)
        print(f"Successfully saved mesh to: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Failed to save STL file: {e}")
        raise
