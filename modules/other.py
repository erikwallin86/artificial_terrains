import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

from modules.data import Module, debug_decorator
from utils.utils import get_terrains, get_terrain
from utils.terrains import get_surface_normal


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



class Roughness(Module):
    """
    Calculate terrain roughness by comparing the surface area of the original
    terrain to a smoothed version of the terrain.

    Roughness is computed as:
        roughness = original_surface_area / smoothed_surface_area

    Parameters:
    - sigma_meter (float): The smoothing length in meters (default: 5).
    """
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_heap=[],
                 default=None, last=None, sigma_meter=5, **_):

        from utils.terrains import calculate_surface_area

        # Possibly set sigma (m) using default value
        sigma_meter = default if default is not None else sigma_meter

        terrains = get_terrains(
            terrain_temp, terrain_heap, last, remove=False)

        results = {'roughness': []}

        for i, terrain in enumerate(terrains):
            resolution = terrain.resolution  # (dy, dx)
            sigma_grid = np.divide(sigma_meter, resolution)

            # Apply Gaussian smoothing
            smoothed_array = gaussian_filter(terrain.array, sigma=sigma_grid)

            # Compute surface normals
            area_orig = calculate_surface_area(terrain.array, terrain.resolution)
            area_smooth = calculate_surface_area(smoothed_array, terrain.resolution)

            roughness = area_orig / area_smooth if area_smooth > 0 else np.nan

            self.info(f"Terrain {i}: roughness = {roughness:.4f}")
            results['roughness'].append(roughness)

        return results


class Histograms(Module):
    """
    Plot and save a histogram of mean slope values in degrees.

    Parameters:
    - slope_deg_mean (list or array): List of slope values in degrees.
    """
    create_folder = True

    @debug_decorator
    def __call__(self, slope_deg_mean=None,
                 default=None, last=None, **_):

        print(f"len(slope_deg_mean): {len(slope_deg_mean)}")

        if slope_deg_mean is None or len(slope_deg_mean) == 0:
            print("No slope data provided.")
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(slope_deg_mean, bins=30, color='steelblue', edgecolor='black')
        ax.set_title("Histogram of Mean Slope (degrees)")
        ax.set_xlabel("Slope (degrees)")
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        # Save the figure
        save_path = os.path.join(self.save_dir, f"slope_histogram_{self.file_id}.png")
        fig.savefig(save_path)
        plt.close(fig)

        return {'histogram_path': save_path}


class PlotLine(Module):
    """
    Plot line

    Parameters:
    - slope_deg_mean (list or array): List of slope values in degrees.
    """
    create_folder = True

    @debug_decorator
    def __call__(self, slope_deg_mean=None,
                 default=None, last=None, **_):

        print(f"len(slope_deg_mean): {len(slope_deg_mean)}")

        if slope_deg_mean is None or len(slope_deg_mean) == 0:
            print("No slope data provided.")
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(slope_deg_mean)
        # ax.hist(slope_deg_mean, bins=30, color='steelblue', edgecolor='black')
        ax.set_title("Histogram of Mean Slope (degrees)")
        ax.set_xlabel("x-value (a.u)")
        ax.set_ylabel("Mean slope degree")
        fig.tight_layout()

        # Save the figure
        save_path = os.path.join(self.save_dir, f"slope_plot_{self.file_id}.png")
        fig.savefig(save_path)
        plt.close(fig)

        return {'histogram_path': save_path}
    
