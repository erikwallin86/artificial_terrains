import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq

from modules.data import Module, debug_decorator
from utils.utils import get_terrains, get_terrain
from utils.plots import new_fig, save_all_axes


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
                 overwrite=False,
                 **kwargs,
                 ):
        if default is not None:
            filename = default
        else:
            original_save_dir = os.path.dirname(self.save_dir)
            for name in ['SaveData/data.npz', 'LogData/data.npz']:
                filename = os.path.join(original_save_dir, name)
                if os.path.isfile(filename):
                    break

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        # Load file
        data = np.load(filename, allow_pickle=True)

        # Convert to dict
        result = {}
        for k, v in data.items():
            if not overwrite and k in kwargs:
                # Update name to avoid overwrite
                k = f'{k}_2'
            result[k] = v

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
            last=None,
            call_total=None, size=None,
            extent=None, loop_id=None, loop_id_r=None,
            # don't skip
            # call_number=None,
            # ppm=None,
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
        to_save = {}
        for k, v in self.data.items():
            # Only save if it's a proper NumPy array
            try:
                arr = np.array(v)
                # if arr.dtype is not object:
                if not np.issubdtype(arr.dtype, np.object_):
                    to_save[k] = arr
            except Exception:
                # Skip entries that can't be converted to a NumPy array
                pass

            # Second attempt, try to concatenate arrays/lists of different size
            if k not in to_save:
                try:
                    arr = np.concatenate(v)
                    # if arr.dtype is not object:
                    if not np.issubdtype(arr.dtype, np.object_):
                        to_save[k] = arr
                except Exception:
                    pass

        filename = os.path.join(self.save_dir, self.filename)
        np.savez(filename, **to_save)
        self.info(f"Saved log data to {filename}")

    def __del__(self):
        # Save on delete
        if self.data:
            self.save()


class MakeData1D(Module):
    """
    Split all arrays with shape (N, D) into D separate 1D arrays named
    key_0, key_1, ..., key_{D-1}.
    """
    create_folder = False

    @debug_decorator
    def __call__(
            self,
            # parameters to skip:
            terrain_temp=[], terrain_heap=[],
            last=None, call_number=None,
            call_total=None, size=None, ppm=None,
            extent=None, loop_id=None, loop_id_r=None,
            **kwargs,
    ):
        result = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] > 1:
                for i in range(v.shape[1]):
                    result[f"{k}_{i}"] = v[:, i]
                self.info(f"Split '{k}' into {v.shape[1]} separate 1D arrays.")
            else:
                result[k] = v  # Keep unchanged if not 2D or not wide enough

        return result


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

        from utils.terrains import get_surface_area

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
            area_orig = get_surface_area(terrain.array, terrain.resolution)
            area_smooth = get_surface_area(smoothed_array, terrain.resolution)

            roughness = area_orig / area_smooth if area_smooth > 0 else np.nan

            self.info(f"Terrain {i}: roughness = {roughness:.4f}")
            results['roughness'].append(roughness)

        return results


class Slope(Module):
    """
    Compute mean slope and gradient for terrain arrays.

    Parameters:
    - sigma_meter (float): Smoothing length in meters (default: 0).
    """
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_heap=[],
                 default=None, last=None, sigma_meter=0, **_):

        # Possibly override sigma with `default`
        sigma_meter = default if default is not None else sigma_meter

        terrains = get_terrains(
            terrain_temp, terrain_heap, last, remove=False)

        results = {
            'slope_deg': [],  # mean slope, in degrees
            'mean_gradient': [],  # mean gradient (dz/dx and dz/dy)
        }

        for i, terrain in enumerate(terrains):
            # Get terrain resolution
            resolution = terrain.resolution  # (dy, dx)

            # Calculate gradient, with or without smoothing
            if sigma_meter > 0:
                sigma_grid = np.divide(sigma_meter, resolution)
                # Apply Gaussian smoothing
                smoothed = gaussian_filter(terrain.array, sigma=sigma_grid)
                # Compute gradients in physical units
                dz_dx, dz_dy = np.gradient(smoothed, *resolution)
            else:
                # Without smoothing
                dz_dx, dz_dy = np.gradient(terrain.array, *resolution)

            # Get mean derivatives in 'metre' scale.
            mean_dz_dx = np.mean(dz_dx)
            mean_dz_dy = np.mean(dz_dy)

            mean_gradient = (mean_dz_dx, mean_dz_dy)
            norm = np.linalg.norm(mean_gradient)
            mean_slope_deg = np.atan(norm)*180/np.pi

            # # Test: Use edge-values instead
            # mean_height_at_xmin = np.mean(terrain.array[0, :])
            # mean_height_at_xmax = np.mean(terrain.array[-1, :])
            # mean_height_at_ymin = np.mean(terrain.array[:, 0])
            # mean_height_at_ymax = np.mean(terrain.array[:, -1])
            # marginal_mean_heights = [
            #     mean_height_at_xmin, mean_height_at_xmax,
            #     mean_height_at_ymin, mean_height_at_ymax]
            # # Get extent
            # [xmin, xmax, ymin, ymax] = terrain.extent
            # # Remove one cell-length in each direction, to get the same value
            # # as mean of derivative above.
            # dz_dx2 = (mean_height_at_xmax-mean_height_at_xmin)/(xmax-xmin - resolution[0])
            # dz_dy2 = (mean_height_at_ymax-mean_height_at_ymin)/(ymax-ymin - resolution[1])

            self.info(f"Terrain {i}: slope = {mean_slope_deg:.4f} (deg)")

            results['slope_deg'].append(mean_slope_deg)
            results['mean_gradient'].append(mean_gradient)

        return results


class SurfaceStructure(Module):
    """
    Compute surface-structure measures
    """
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_heap=[],
                 position=None, height=None,
                 default=None, last=None, sigma_meter=0, **_):

        terrain = get_terrain(
            terrain_temp, terrain_heap, remove=False)

        # Count number of rocks larger than some cutoffs
        a = np.sum(height > 0.1)
        b = np.sum(height > 0.3)
        c = np.sum(height > 0.5)
        d = np.sum(height > 0.7)
        print(f"(a, b, c, d):{(a, b, c, d)}")

        # Categorize as number of obstaces per ha (100x100m)
        per_ha_factor = 100*100/np.multiply(*terrain.size)

        results = {
            'h_20': (a-b)*per_ha_factor,
            'h_40': (b-c)*per_ha_factor,
            'h_60': (c-d)*per_ha_factor,
            'h_80': (d-0)*per_ha_factor,
            'h_sum': a*per_ha_factor,
        }

        # Calculate surface structure
        from utils.utils import calc_ytstruktur_2
        h_20 = results['h_20']
        h_40 = results['h_40']
        h_60 = results['h_60']
        h_80 = results['h_80']
        Y = calc_ytstruktur_2(h_20, h_40, h_60, h_80)
        self.info(f"Y:{Y}")

        return results


class CombineRoughness(Module):
    """
    Combine roughness values and weights into a proxy roughness estimate.
    """
    list_or_results = []
    create_folder = False

    @debug_decorator
    def __call__(self, roughness=None, weights=0,
                 default=None, last=None, **kwargs):
        """
        Parameters:
        - roughness: Array-like of roughness values.
        - weights: Array-like of weights for each roughness value.
        """
        r_i = roughness
        w_i = weights

        alpha = 1.0  # Currently hardcoded

        def model(alpha, beta=2):
            base = (np.power(r_i, alpha)-1) * np.power(np.abs(w_i), beta)
            r_pred = 1 + np.power(np.sum(base), 1 / alpha)
            return r_pred

        result = model(alpha)

        self.info(f"proxy roughness:{result}")
        return {'proxy_roughness': result}


class CombineSlope(Module):
    """
    Combine slope gradients into a single representative slope estimate.

    Takes weighted mean gradients (dz/dx, dz/dy) and computes the resulting
    overall slope angle in degrees.
    """
    list_or_results = []
    create_folder = False

    @debug_decorator
    def __call__(self, mean_gradient=None, weights=0,
                 default=None, last=None, **kwargs):
        """
        Parameters:
        - mean_gradient: List of (dz/dx, dz/dy) tuples.
        - weights: List or array of weights for each gradient.
        """
        combined_gradient = np.zeros(2)

        for (dz_dx, dz_dy), weight in zip(mean_gradient, weights):
            # Scale slopes by weight
            dz_dx *= weight
            dz_dy *= weight

            # Add to combined vector
            combined_gradient += np.array([dz_dx, dz_dy])

        self.info(f"combined_gradient:{combined_gradient}")
        norm = np.linalg.norm(combined_gradient)
        combined_slope_deg = np.atan(norm)*180/np.pi
        self.info(f"combined_slope_deg:{combined_slope_deg}")

        return {'combined_slope_deg': combined_slope_deg}


class OptimizeRoughness(Module):
    """
    Find the optimal roughness proxy

    """
    create_folder = True

    @debug_decorator
    def __call__(self, mean_gradient=None, weights=0,
                 default=None, last=None, **kwargs):
        from scipy.optimize import minimize_scalar

        original_save_dir = os.path.dirname(self.save_dir)

        file1 = os.path.join(original_save_dir, "LogData/data.npz")
        data1 = np.load(file1)

        file2 = os.path.join(original_save_dir, "LogData/data2.npz")
        data2 = np.load(file2)

        w_i = data1['weights']
        r_i = data1['roughness']

        r_obs = data2['roughness']

        def model(alpha, beta=2):
            # r_pred = 1 + np.sum((np.power(r_i, alpha)-1) * np.abs(w_i), axis=1)
            # beta = alpha
            base = (np.power(r_i, alpha)-1) * np.power(np.abs(w_i), beta)
            r_pred = 1 + np.power(np.sum(base, axis=1), 1 / alpha)

            # r_pred = 1.0 + np.sum((np.power(r_i-1, alpha)) * np.abs(w_i), axis=1)

            return r_pred

        # Loss function to minimize
        def loss(alpha):
            r_pred = model(alpha)
            loss = np.sum((r_obs - r_pred)**2)
            print(f"(alpha, loss):{(alpha, loss)}")

            return loss

        y = []
        # Set the limits of the plots *and the optimization*
        x = np.linspace(0.55, 2, 100)

        for x_ in x:
            y.append(loss(x_))
        y = np.array(y)

        # Plot loss function as a function of alpha
        fig, ax = new_fig()
        ax.plot(x, y)
        filename = os.path.join(self.save_dir, 'loss_as_function_of_alpha.png')
        fig.savefig(filename)

        # Minimize the loss
        res = minimize_scalar(loss, bounds=(x[0], x[-1]), method='bounded')
        if res.success:
            alpha = res.x
            print(f"Optimal alpha: {alpha}")
            print(f"loss(alpha):{loss(alpha)}")
        else:
            print("Optimization failed.")

        # Plot roughness v pred roughness
        fig, ax = new_fig()
        r_pred = model(alpha)
        ax.scatter(r_obs, r_pred)
        ax.set_xlabel('r-obs')
        ax.set_ylabel('r-pred')
        ax.grid()
        ax.plot([1, 1.15], [1, 1.15])
        filename = os.path.join(self.save_dir, 'pred_v_obs.png')
        fig.savefig(filename)



class SetRoughness(Module):
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
        r_i = np.asarray(roughness)
        weights = np.asarray(weights)

        # Define proxy roughness function
        def proxy_roughness(lambda_, alpha=1, beta=2):
            # Scale weights
            w_i = np.absolute(weights) * r_i**lambda_

            # Calculate proxy roughness using scaled weights
            base = (np.power(r_i, alpha)-1) * np.power(np.abs(w_i), beta)
            r_pred = 1 + np.power(np.sum(base), 1 / alpha)

            return r_pred

        # Define root-finding function: f(λ) = R(λ) - target_roughness
        def f(lambda_):
            return proxy_roughness(lambda_) - target_roughness

        # Solve for λ numerically using Brent's method
        lambda_solution = brentq(f, -1000, 100)

        self.info(f"lambda_solution:{lambda_solution}")

        # Compute new weights: w_i_new = w_i * r_i^λ
        adjusted_weights = weights * r_i**lambda_solution
        self.info(f"adjusted_weights:{adjusted_weights}")

        return {'weights': adjusted_weights}


class SetSlope(Module):
    """
    Adjust weights so that the proxy slope matches the target.
    """
    create_folder = False

    @debug_decorator
    def __call__(self, slope_deg=None, mean_gradient=None, weights=None,
                 target_slope_deg=5, default=None, **_):

        # Use `default` if `target_slope_deg` is not explicitly provided
        target_slope_deg = default if default is not None else target_slope_deg

        target_slope = np.tan(np.radians(target_slope_deg))

        gradients = np.asarray(mean_gradient)
        weights = np.asarray(weights)

        # Norm (magnitude) of each gradient
        norms = np.linalg.norm(gradients, axis=1)

        # Simple rescale
        norms = 1+norms/np.max(norms)

        # Define proxy slope as function of λ
        def proxy_slope(lambda_):
            scaled_weights = weights * norms**lambda_
            combined_gradient = np.sum(gradients.T * scaled_weights, axis=1)
            return np.linalg.norm(combined_gradient)

        # Root function: difference from target slope
        def f(lambda_):
            return proxy_slope(lambda_) - target_slope
        try:
            lambda_solution = brentq(f, -10, 100)
        except ValueError:
            self.warning("Brentq failed: returning original weights")
            return {'weights': weights}

        adjusted_weights = weights * norms**lambda_solution

        self.info(f"lambda_solution: {lambda_solution:.4f}")
        self.info(f"adjusted_weights: {adjusted_weights}")

        return {'weights': adjusted_weights}


class PlotScatter(Module):
    """
    Plot scatter plots of all pairs of data with equal length.
    Skips plotting when keys are identical (k1 == k2) or lengths differ.
    """
    create_folder = True

    @debug_decorator
    def __call__(self,
                 default=None,
                 # parameters to skip:
                 terrain_temp=[], terrain_heap=[],
                 last=None, call_number=None,
                 call_total=None, size=None, ppm=None,
                 extent=None, loop_id=None, loop_id_r=None,
                 # plot-parameters
                 exportmode=False, dpi=200, overwrite=False,
                 color=None, cmap=None,
                 grid=False,
                 **kwargs):

        # Loop over all key-value pairs in kwargs
        for k1, v1 in kwargs.items():
            for k2, v2 in kwargs.items():
                # Skip plotting identical variables or mismatched lengths
                # if k1 == k2 or len(v1) != len(v2):
                if k1 == k2:
                    continue

                # Generate filename and save figure
                filename = f"{k1}_v_{k2}{self.file_id}.png"
                filename = os.path.join(self.save_dir, filename)
                if os.path.isfile(filename) and not overwrite:
                    continue

                try:
                    # Create figure and axis
                    fig, ax = new_fig()

                    if color is not None and color in kwargs:
                        color_data = kwargs[color]
                        sc = ax.scatter(v1, v2, c=color_data, cmap=cmap, alpha=1)
                        fig.colorbar(sc, ax=ax, label=color)
                    else:
                        ax.scatter(v1, v2)

                    # Label axes
                    ax.set_xlabel(k1)
                    ax.set_ylabel(k2)

                    # Apply tight layout
                    fig.tight_layout()

                    if grid:
                        ax.grid()
                    fig.savefig(filename)
                    if exportmode:
                        save_all_axes(fig, filename, delta=0.0, dpi=dpi)

                except Exception:
                    # Skip silently on error (could add logging if needed)
                    pass


class PlotHistogram(Module):
    """
    Plot histogram for each entry in kwargs.
    Accepts optional `bins` and `range` for histogram computation.
    """
    create_folder = True

    @debug_decorator
    def __call__(self,
                 default=None,
                 # parameters to skip:
                 terrain_temp=[], terrain_heap=[],
                 last=None, call_number=None,
                 call_total=None, size=None, ppm=None,
                 extent=None, loop_id=None, loop_id_r=None,
                 # plot parameters
                 bins=10, range=None,
                 **kwargs):

        # Use `default` as bins if specified
        bins = default if default is not None else bins

        for k, v in kwargs.items():
            try:
                # Create figure and axis
                fig, ax = new_fig()

                # Compute histogram
                hist, bin_edges = np.histogram(v, bins, range)

                # Compute bin centers
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                # Plot as bar chart using bin centers
                ax.bar(bin_centers, hist, width=0.9*np.diff(bin_edges))

                # Label axes
                ax.set_xlabel(k)
                ax.set_ylabel('Occurrences')

                # Apply tight layout
                fig.tight_layout()

                # Generate filename and save figure
                filename = f"{k}{self.file_id}.png"
                filename = os.path.join(self.save_dir, filename)
                fig.savefig(filename)

            except Exception:
                # Silently skip any error (consider logging)
                pass


class PlotLines(Module):
    """
    Plot line graphs for each entry in kwargs.

    Each value `v` is assumed to be 2D (n_lines x n_points). Lines are
    plotted by transposing the array so each row becomes a line on the plot.

    The x-axis represents the index of each point.
    """
    create_folder = True

    @debug_decorator
    def __call__(self,
                 default=None,
                 # parameters to skip:
                 terrain_temp=[], terrain_heap=[],
                 last=None, call_number=None,
                 call_total=None, size=None, ppm=None,
                 extent=None, loop_id=None, loop_id_r=None,
                 **kwargs):

        for k, v in kwargs.items():
            try:
                # Create figure and axis
                fig, ax = new_fig()

                # Plot transposed array (each row becomes a line)
                ax.plot(v.T)

                # Label axes
                ax.set_xlabel('Index')
                ax.set_ylabel(k)

                # Apply tight layout
                fig.tight_layout()

                # Generate filename and save figure
                filename = f"{k}{self.file_id}.png"
                filename = os.path.join(self.save_dir, filename)
                fig.savefig(filename)

            except Exception:
                # Silently skip any error (consider logging)
                pass
