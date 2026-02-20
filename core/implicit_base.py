from abc import ABC, abstractmethod
import numpy as np

class ImplicitSurface(ABC):
    """
    Abstract base class for an implicit surface.
    The surface is defined by the set of points P where evaluate(P) = 0.
    """
    @abstractmethod
    def evaluate(self, x, y, z):
        """
        Evaluates the implicit function at a given point (x, y, z).
        
        Returns:
            A scalar value. The surface is the zero-level set.
        """
        pass

class Sphere(ImplicitSurface):
    """
    Represents a sphere centered at the origin.
    """
    def __init__(self, radius=1.0, center=(0.0, 0.0, 0.0)):
        """
        Initializes a Sphere.
        
        Args:
            radius (float): The radius of the sphere.
            center (tuple): The (x, y, z) coordinates of the sphere's center.
        """
        self.radius = radius
        self.center = np.array(center)

    def evaluate(self, x, y, z):
        """
        Evaluates the sphere's implicit equation.
        f(x, y, z) = (x-cx)^2 + (y-cy)^2 + (z-cz)^2 - r^2
        """
        # Using individual coordinates to avoid broadcasting issues with stacked arrays
        return (x - self.center[0])**2 + (y - self.center[1])**2 + (z - self.center[2])**2 - self.radius**2
