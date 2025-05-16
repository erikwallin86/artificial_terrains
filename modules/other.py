import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq

from modules.data import Module, debug_decorator
from utils.utils import get_terrains, get_terrain
from utils.terrains import get_surface_normal
from utils.plots import new_fig



class SaveData(Module):
    """
    Save selected list or array data to a compressed .npz file.
    """
    create_folder = False

    @debug_decorator
    def __call__(self,
                 filename='data.npz',
                 default=None,
                 folder='SaveData',
                 # parameters to skip:
                 terrain_temp=[], terrain_heap=[],
                 last=None, call_number=None,
                 call_total=None, size=None, ppm=None,
                 extent=None, loop_id=None, loop_id_r=None,
                 **kwargs):
        # use default to update filename or folder
        if default is not None and '.npz' in default:
            filename = default
        if default is not None:
            folder = default
            
        # Use folder
        basename = os.path.basename(self.save_dir)
        if basename != folder:
            dirname = os.path.dirname(self.save_dir)
            self.save_dir = os.path.join(dirname, folder)

        # Create folder if needed
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # Create dict to collect data to save
        data = {}

        # Full path
        filename = os.path.join(self.save_dir, filename)

        for k, v in kwargs.items():
            # Only save list-like objects
            if isinstance(v, (list, tuple, np.ndarray)):
                data[k] = np.array(v)

        # Spara till .npz
        np.savez_compressed(filename, **data)
        self.info(f"Saved to {filename}")


class LoadData(Module):
    """
    Load a .npz file and return the contents as a dictionary of arrays.
    """
    create_folder = False

    @debug_decorator
    def __call__(self,
                 default=None,
                 filename=None,
                 **_,
                 ):
        if default is not None:
            filename = default
        else:
            original_save_dir = os.path.dirname(self.save_dir)
            filename = os.path.join(original_save_dir, 'SaveData/data.npz')

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        # Load file
        data = np.load(filename)

        # Convert to dict
        result = {k: data[k] for k in data.files}
        self.info(f"Loaded from {filename}: keys = {list(result.keys())}")

        return result


class LogData(Module):
    """
    Collect data by appending to lists, then save to .npz file on delete.
    """
    create_folder = True

    def __init__(self, *args, **kwargs):
        self.data = {}  # key: list of values
        self.filename = None
        super().__init__(*args, **kwargs)

    @debug_decorator
    def __call__(
            self, default=None, filename='data.npz',
            # parameters to skip:
            terrain_temp=[], terrain_heap=[],
            last=None, call_number=None,
            call_total=None, size=None, ppm=None,
            extent=None, loop_id=None, loop_id_r=None,
            **kwargs,
    ):

        # Append each kwarg value to the internal data lists
        for k, v in kwargs.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

        # Possibly set the filename
        if self.filename is None:
            if default is None:
                self.filename = filename
            if default is not None:
                self.filename = default

    def save(self):
        # Convert lists to arrays and save
        to_save = {k: np.array(v) for k, v in self.data.items()}
        filename = os.path.join(self.save_dir, self.filename)
        np.savez(filename, **to_save)
        self.info(f"Saved log data to {filename}")

    def __del__(self):
        # Save on delete
        if self.data:
            self.save()


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


class CombineRoughness(Module):
    """
    Test to combine roughness values in some heuristic way

    """
    list_or_results = []
    create_folder = False

    @debug_decorator
    def __call__(self, roughness=None, weights=0,
                 default=None, last=None, **kwargs):

        # Calculate 'combined roughness'
        result = 1
        for r, w_r in zip(roughness, weights):
            result += (np.sqrt(r)-1)*np.absolute(w_r)

        self.info(f"heuristic_combined_roughness:{result}")

        if 'heuristic_combined_roughness' in kwargs:
            print("###################################")
            kwargs['heuristic_combined_roughness'].append(result)
        else:
            return {'heuristic_combined_roughness': [result]}


class PlotRoughness(Module):
    """
    Plot roughness comparison

    """
    create_folder = True

    @debug_decorator
    def __call__(self, roughness=None, heuristic_combined_roughness=None,
                 default=None, last=None, **_):

        fig, ax = new_fig()
        ax.scatter(roughness, heuristic_combined_roughness)
        # ax.plot(slope_deg_mean)
        # ax.hist(slope_deg_mean, bins=30, color='steelblue', edgecolor='black')
        # ax.set_title("Histogram of Mean Slope (degrees)")
        ax.set_xlabel("Roughness")
        ax.set_ylabel("Heuristic combined roughness")
        fig.tight_layout()

        # Save the figure
        filename = f"roughness_v_heuristic_{self.file_id}.png"
        filename = os.path.join(self.save_dir, filename)
        fig.savefig(filename)


class PlotRoughnessForLengthscale(Module):
    """
    Plot typical roughness for different length-scales
    """
    create_folder = True

    @debug_decorator
    def __call__(self, roughness=None,
                 default=None, last=None,
                 num_octaves=10,
                 **_):

        roughness = np.array(roughness)

        # Reshape to get length-scale on separate axis
        roughness = roughness.reshape(-1, num_octaves)
        print(f"roughness.shape:{roughness.shape}")
        fig, ax = new_fig()

        for i in range(roughness.shape[0]):
            ax.plot(roughness[i, :])

        ax.set_xlabel("Length-scale")
        ax.set_ylabel("Roughness")
        fig.tight_layout()

        # Save the figure
        filename = f"roughness_for_lengthscales_{self.file_id}.png"
        filename = os.path.join(self.save_dir, filename)
        fig.savefig(filename)


class SetRoughness(Module):
    """
    Adjust weights so that the proxy roughness matches the target.
    """
    create_folder = False

    @debug_decorator
    def __call__(self, roughness=None, weights=None, target_roughness=1.05,
                 default=None, **_):

        # Set target from default
        target_roughness = default if default is not None else target_roughness

        if roughness is None or weights is None:
            raise ValueError("Both roughness and weights must be provided.")

        roughness = np.asarray(roughness)
        weights = np.asarray(weights)

        delta_r = np.sqrt(roughness) - 1
        print(f"delta_r:{delta_r}")
        print(f"np.sum(delta_r):{np.sum(delta_r)}")

        S = np.sum(np.abs(weights) * delta_r)
        # Calculate proxy roughness, \tilde{R}
        R_tilde = 1 + S
        print(f"R_tilde:{R_tilde}")

        # Difference to desired (proxy) roughness
        diff = target_roughness - R_tilde
        print(f"diff:{diff}")
        # Negative sign --> we want our proxy roughness to decrease

        # Calculate vector to add or subtract to weights
        # jag kallar den ibland 'a'.
        delta_w = delta_r/np.sum(delta_r)
        print(f"delta_w:{delta_w}")
        print(f"np.sum(delta_w):{np.sum(delta_w)}")

        # Calculate new weights
        lambda_factor = (target_roughness - R_tilde)/np.sum(delta_r * delta_w)
        print(f"lambda_factor:{lambda_factor}")

        # Test with clip
        adjusted_weights = weights + delta_w * lambda_factor
        print(f"adjusted_weights:{adjusted_weights}")
        adjusted_weights = np.clip(adjusted_weights, 0, None)
        print(f"adjusted_weights:{adjusted_weights}")

        return {'weights': adjusted_weights}


class SetRoughness2(Module):
    """
    Adjust weights so that the proxy roughness matches the target.

    The proxy roughness is defined as:
        R_proxy = 1 + sum_i (sqrt(r_i) - 1) * w_i

    We solve for lambda in the following,
        R_target = 1 + sum_i (sqrt(r_i) - 1) * w_i * r_i^λ

    Where:
        - r_i is the roughness of terrain i
        - w_i is the original weight of terrain i
        - λ (lambda) is a scalar exponent we solve for
        - The adjusted weight is: w_i_target = w_i * r_i^λ

    This allows both positive and negative weights, and ensures that weights
    scale toward or away from zero depending on how much each terrain contributes
    to the roughness deviation.
    """
    create_folder = False

    @debug_decorator
    def __call__(self, roughness=None, weights=None, target_roughness=1.0,
                 default=None, **_):

        # Use `default` if `target_roughness` is not explicitly provided
        target_roughness = default if default is not None else target_roughness

        # Check input validity
        if roughness is None or weights is None:
            raise ValueError("Both roughness and weights must be provided.")

        # Convert to numpy arrays for vectorized computation
        roughness = np.asarray(roughness)
        weights = np.asarray(weights)

        # Precompute delta_r = sqrt(r) - 1
        delta_r = np.sqrt(roughness) - 1

        # Base for exponentiation (a_i), set to roughness r_i
        base_r = roughness

        # Define proxy roughness function of λ:
        #     R(λ) = 1 + sum_i (sqrt(r_i) - 1) * w_i * r_i^λ
        # For lambda=0 this is the proxy roughness, otherwise it's a scaled one?
        def proxy_roughness(lambda_):
            scaled = np.absolute(weights) * base_r**lambda_
            return 1 + np.sum(delta_r * scaled)

        # Define root-finding function: f(λ) = R(λ) - target_roughness
        def f(lambda_):
            return proxy_roughness(lambda_) - target_roughness

        # Solve for λ numerically using Brent's method
        lambda_solution = brentq(f, -1000, 100)

        self.info(f"lambda_solution:{lambda_solution}")

        # Compute new weights: w_i_new = w_i * r_i^λ
        adjusted_weights = weights * base_r**lambda_solution
        self.info(f"adjusted_weights:{adjusted_weights}")

        return {'weights': adjusted_weights}


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
    

class PlotPersistanceVRoughness(Module):
    """
    """
    create_folder = True

    @debug_decorator
    def __call__(self, roughness=None, persistance=None,
                 default=None, last=None, **_):

        fig, ax = new_fig()

        ax.scatter(persistance, roughness)

        ax.set_xlabel("Persistance (a.u.)")
        ax.set_ylabel("Roughness (a.u.)")
        fig.tight_layout()

        filename = f"persistance_v_roughness{self.file_id}.png"
        filename = os.path.join(self.save_dir, filename)
        fig.savefig(filename)
