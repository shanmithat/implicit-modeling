import numpy as np
from core.implicit_base import ImplicitSurface

class Gyroid(ImplicitSurface):
    """
    Represents a Gyroid TPMS structure.
    The formula is sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x).
    """
    def __init__(self, frequency=1.0):
        """
        Initializes a Gyroid.
        
        Args:
            frequency (float): Controls the density of the lattice structure.
        """
        self.frequency = frequency

    def evaluate(self, x, y, z):
        """
        Evaluates the Gyroid implicit equation.
        """
        fx = self.frequency * x
        fy = self.frequency * y
        fz = self.frequency * z
        
        return (np.sin(fx) * np.cos(fy) +
                np.sin(fy) * np.cos(fz) +
                np.sin(fz) * np.cos(fx))

class Intersection(ImplicitSurface):
    """
    Represents the intersection of multiple implicit surfaces.
    This is a boolean 'AND' operation, achieved by taking the maximum
    of the individual surface evaluations.
    """
    def __init__(self, *surfaces):
        """
        Initializes an Intersection.
        
        Args:
            *surfaces: A variable number of ImplicitSurface objects.
        """
        if not all(isinstance(s, ImplicitSurface) for s in surfaces):
            raise TypeError("All arguments must be ImplicitSurface objects.")
        self.surfaces = surfaces

    def evaluate(self, x, y, z):
        """
        Evaluates the intersection by taking the maximum of all surfaces.
        """
        evaluations = [s.evaluate(x, y, z) for s in self.surfaces]
        return np.maximum.reduce(evaluations)
