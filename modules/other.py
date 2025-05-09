from modules.data import Module, debug_decorator
import numpy as np
from utils.utils import get_terrains, get_terrain


class Slope(Module):
    """
    Print slope statistics (in degrees) for each terrain.
    """
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_heap=[],
                 default=None, last=None, **_):

        # Get terrain/terrains
        terrains = get_terrains(
            terrain_temp, terrain_heap, last, remove=False)

        results = {
            'slope_deg_max': [],
            'slope_deg_mean': [],
            'slope_deg_std': [],
        }

        # Loop and calculate slope statistics for each terrain
        for i, terrain in enumerate(terrains):
            height = terrain.array
            dx, dy = terrain.resolution

            grad_y, grad_x = np.gradient(height, dy, dx)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            slope_deg = np.degrees(np.arctan(slope))

            self.info(f"Terrain {i}:")
            self.info(f"  Max slope:  {slope_deg.max():6.2f}°")
            self.info(f"  Mean slope: {slope_deg.mean():6.2f}°")
            self.info(f"  Std slope:  {slope_deg.std():6.2f}°")

            results['slope_deg_max'].append(slope_deg.max())
            results['slope_deg_mean'].append(slope_deg.mean())
            results['slope_deg_std'].append(slope_deg.std())

        return results
