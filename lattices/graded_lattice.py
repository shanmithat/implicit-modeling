import numpy as np
from core.implicit_base import ImplicitSurface

class GradedLattice(ImplicitSurface):
    """
    Implements functional grading by modulating the offset level of a base lattice.
    The resulting surface is defined by: base_lattice.evaluate(x, y, z) - thickness_function(x, y, z) = 0.
    """
    def __init__(self, base_lattice, thickness_function):
        """
        Initializes a GradedLattice.
        
        Args:
            base_lattice (ImplicitSurface): The underlying lattice structure (e.g., Gyroid).
            thickness_function (callable): A function that takes (x, y, z) and returns a thickness offset.
        """
        self.base_lattice = base_lattice
        self.thickness_function = thickness_function

    def evaluate(self, x, y, z):
        """
        Evaluates the graded lattice by subtracting the local thickness offset from the base lattice value.
        """
        return self.base_lattice.evaluate(x, y, z) - self.thickness_function(x, y, z)

def linear_z_grading(z_min, z_max, t_min, t_max):
    """
    Creates a linear thickness grading function along the Z-axis.
    
    Args:
        z_min, z_max (float): The Z-range over which to apply the grading.
        t_min, t_max (float): The thickness offsets at z_min and z_max respectively.
    """
    def grading(x, y, z):
        # Clamp z to the range to prevent extrapolation beyond t_min/t_max
        z_clamped = np.clip(z, z_min, z_max)
        return t_min + (t_max - t_min) * (z_clamped - z_min) / (z_max - z_min)
    return grading

def radial_grading(center, r_max, t_inner, t_outer):
    """
    Creates a radial thickness grading function from a center point.
    
    Args:
        center (tuple): The (x, y, z) center of the radial grading.
        r_max (float): The radius at which the grading reaches t_outer.
        t_inner, t_outer (float): The thickness offsets at the center and at r_max.
    """
    center_arr = np.array(center)
    def grading(x, y, z):
        dist = np.sqrt((x - center_arr[0])**2 + (y - center_arr[1])**2 + (z - center_arr[2])**2)
        dist_clamped = np.clip(dist, 0, r_max)
        return t_inner + (t_outer - t_inner) * (dist_clamped / r_max)
    return grading

def point_attractor_grading(points, attractor_pos, radius, t_min, t_max):
    """
    Calculates thickness based on distance to an attractor point.
    Makes the lattice thicker near the attractor_pos and thinner as it moves away.
    
    Args:
        points (tuple/np.ndarray): Either (x, y, z) tuple of coordinate arrays, or (N, 3) array.
        attractor_pos (tuple/list): (x, y, z) position of the attractor point.
        radius (float): Distance over which the grading is applied.
        t_min (float): Minimum thickness (at radius and beyond).
        t_max (float): Maximum thickness (at attractor position).
    """
    if isinstance(points, tuple) and len(points) == 3:
        x, y, z = points
    else:
        points = np.asarray(points)
        if points.ndim == 1:
            x, y, z = points[0], points[1], points[2]
        else:
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

    attractor_arr = np.array(attractor_pos)
    dist = np.sqrt((x - attractor_arr[0])**2 + (y - attractor_arr[1])**2 + (z - attractor_arr[2])**2)
    dist_clamped = np.clip(dist, 0, radius)
    # At dist=0, return t_max. At dist=radius, return t_min.
    return t_max - (t_max - t_min) * (dist_clamped / radius)
