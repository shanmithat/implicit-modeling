import numpy as np
from core.implicit_base import ImplicitSurface

class Gyroid(ImplicitSurface):
    """
    Represents a Gyroid TPMS structure.
    The formula is sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x).
    """
    def __init__(self, frequency=1.0):
        self.frequency = frequency

    def evaluate(self, x, y, z):
        fx, fy, fz = self.frequency * x, self.frequency * y, self.frequency * z
        return (np.sin(fx) * np.cos(fy) +
                np.sin(fy) * np.cos(fz) +
                np.sin(fz) * np.cos(fx))

class Diamond(ImplicitSurface):
    """
    Represents a Diamond (Schwarz D) TPMS structure.
    Formula: sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z) + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z)
    """
    def __init__(self, frequency=1.0):
        self.frequency = frequency

    def evaluate(self, x, y, z):
        fx, fy, fz = self.frequency * x, self.frequency * y, self.frequency * z
        return (np.sin(fx) * np.sin(fy) * np.sin(fz) + 
                np.sin(fx) * np.cos(fy) * np.cos(fz) + 
                np.cos(fx) * np.sin(fy) * np.cos(fz) + 
                np.cos(fx) * np.cos(fy) * np.sin(fz))

class HybridLattice(ImplicitSurface):
    """
    Blends two implicit surfaces using Linear Interpolation (LERP).
    Result = (1 - w) * L1 + w * L2
    """
    def __init__(self, lattice1, lattice2, weight=0.5):
        self.lattice1 = lattice1
        self.lattice2 = lattice2
        self.weight = np.clip(weight, 0.0, 1.0)

    def evaluate(self, x, y, z):
        val1 = self.lattice1.evaluate(x, y, z)
        val2 = self.lattice2.evaluate(x, y, z)
        return (1.0 - self.weight) * val1 + self.weight * val2

class Intersection(ImplicitSurface):
    """
    Boolean 'AND' operation: clips the lattice to the container boundary.
    """
    def __init__(self, *surfaces):
        self.surfaces = surfaces

    def evaluate(self, x, y, z):
        evaluations = [s.evaluate(x, y, z) for s in self.surfaces]
        return np.maximum.reduce(evaluations)
