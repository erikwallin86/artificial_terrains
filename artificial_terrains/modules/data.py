import os
import numpy as np
from utils.terrains import Terrain
from utils.debug import debug
from utils.utils import Distribution
from utils.plots import save_all_axes
import colorcet as cc
import time
import functools


def debug_decorator(func):
    '''
    Wrapper around calls to print some debug info
    and measure execution time.
    '''
    @functools.wraps(func)
    def wrapper(self, call_number=None, call_total=None, *args, **kwargs):
        if call_number is not None and call_total is not None:
            self.logger.info(f"Run {self.name} {call_number+1}/{call_total}")

        if call_total is not None and call_total > 1:
            self.file_id = f'_{call_number:05d}'
        else:
            self.file_id = ''

        # Possibly debug input
        debug(self, kwargs, call_number, call_total, 'input')

        # ---- timing starts here ----
        start = time.perf_counter()

        result = func(
            self, *args,
            call_number=call_number,
            call_total=call_total,
            **kwargs)

        elapsed = time.perf_counter() - start
        # ---- timing ends here ----

        # Log timing
        self.logger.info(f"  took {elapsed:.3f} s")

        # Possibly debug output
        debug(self, result, call_number, call_total, 'output')

        return result

    return wrapper


class Module():
    create_folder = True
    '''
    Args:
    '''
    def __init__(self, save_dir=None, logger=None):
        self.save_dir = save_dir
        self.logger = logger

        self.save_dir_original = save_dir

        self.save_dir = os.path.join(self.save_dir, self.__class__.__name__)
        if self.create_folder:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)

    def __call__(
            self, **kwargs):
        # Return kwargs which are not 'input'/'data'
        return kwargs

    def info(self, logging_string):
        ''' 'Shorter' method for logger.info, with indent. '''
        self.logger.info(f"  {logging_string}")

    def debug(self, logging_string):
        ''' 'Shorter' method for logger.debug, with indent. '''
        self.logger.debug(f"  {logging_string}")

    def start(self, **kwargs):
        # Store kwargs
        self.kwargs = kwargs
        # Initialize loop generator
        self.loop_generator_instance = self.loop_generator()

    @property
    def name(self):
        return self.__class__.__name__

    def loop_generator(self):
        """
        Default-beteende: returnera bara en gång, samma som __call_
        """
        yield self.__call__(**self.kwargs)

    def __iter__(self):
        """
        Gör modulen itererbar
        """
        return self.loop_generator_instance


class Print(Module):
    ''' Print array info '''
    create_folder = False

    @debug_decorator
    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            # if isinstance(value, np.ndarray):
            #     print(f"len(value):{len(value)}")

            # Print
            self.logger.info(f"{key}: {value}")


class GridSize(Module):
    ''' Test '''
    create_folder = False

    @debug_decorator
    def __call__(self, grid_size=100, default=None, **kwargs):
        grid_size = default if default is not None else grid_size

        if isinstance(grid_size, list):
            grid_size_x, grid_size_y = grid_size
        elif isinstance(grid_size, str) and 'x' in default:
            grid_size_x, grid_size_y = map(float, default.lower().split('x'))
        else:
            grid_size_x, grid_size_y = (grid_size, grid_size)

        grid_size = [grid_size_x, grid_size_y]

        self.info(f"Set grid_size:{grid_size}")

        return {
            'grid_size': grid_size,
            'resolution': None,  # Not valid?
            }


class Size(Module):
    ''' Test '''
    create_folder = False

    @debug_decorator
    def __call__(self, size=50, default=None, **kwargs):
        # göra så att 'default' parametern alltid står först
        # och göra det här automatiskt
        size = default if default is not None else size

        if isinstance(size, list):
            size_x, size_y = size
        elif isinstance(size, str) and 'x' in default:
            size_x, size_y = map(float, default.lower().split('x'))
        else:
            size_x, size_y = (size, size)

        size = [size_x, size_y]

        self.info(f"Set size:{size} [m]")

        return {
            'size': size,
            'extent': [-size[0]/2, size[0]/2, -size[1]/2, size[1]/2],  # test
            }


class Extent(Module):
    ''' Extent '''
    create_folder = False

    @debug_decorator
    def __call__(self, extent=[-25, 25, -25, 25], default=None, **kwargs):
        extent = default if default is not None else extent

        self.info(f"Set extent:{extent} [m]")
        from utils.terrains import extent_to_size

        return {
            'size': extent_to_size(extent),
            'extent': extent,
            }


class Location(Module):
    ''' Update location in extent '''
    create_folder = False

    @debug_decorator
    def __call__(self, location=[0, 0], extent=None, default=None, **kwargs):
        location = default if default is not None else location

        from utils.terrains import extent_to_location
        current_location = extent_to_location(extent)

        # Calculate diff
        diff = [location[0] - current_location[0],
                location[1] - current_location[1]]

        # And shift the extent
        extent = [extent[0]+diff[0],
                  extent[1]+diff[0],
                  extent[2]+diff[1],
                  extent[3]+diff[1]
                  ]

        return {
            'extent': extent,
            }


class Resolution(Module):
    '''  '''
    create_folder = False

    @debug_decorator
    def __call__(self, resolution=5, default=None, **kwargs):
        resolution = default if default is not None else resolution

        self.info(f"Set points-per-meter:{resolution}")

        return {
            'resolution': resolution,
            'grid_size': None,
            }


class Seed(Module):
    """
    Sets a seed

    If `default` is an integer, it overrides `seed`.
    If `default` is 'persistent_random', a random seed is generated once and reused.

    Returns:
        dict: {'seed': resolved seed value}
    """
    create_folder = False

    @debug_decorator
    def __call__(self, seed=0, default=None, **kwargs):
        # Override with default if specified
        if default is not None and isinstance(default, int):
            seed = default
        elif default == 'persistent_random':
            try:
                seed = self.seed
            except AttributeError:
                import random
                seed = random.randint(0, 2**32 - 1)
                self.seed = seed

        self.info(f"Set seed: {seed}")

        return {'seed': seed}


class Folder(Module):
    ''' Set 'folder', affecting e.g Save, and Plot '''
    create_folder = False

    @debug_decorator
    def __call__(self, folder=None, default=None, **kwargs):
        # göra så att 'default' parametern alltid står först
        # och göra det här automatiskt
        folder = default if default is not None else folder

        self.info(f"Set 'folder':{folder}")

        return {
            'folder': folder,
            }


class DebugPlot(Module):
    ''' Do debug plot '''
    create_folder = False

    @debug_decorator
    def __call__(self, default=None, filename='debug_plot.png', **kwargs):
        from utils.plots import debug_plot_horizontal

        filename = default if default is not None else filename
        filename = os.path.join(self.save_dir_original, filename)
        debug_plot_horizontal(filename, **kwargs)


class Hypercube(Module):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self, n=2, seed=None,
                 l_bounds=[1, 1, 0.1], u_bounds=[10, 10, 0.3],
                 default=None, call_number=None, call_total=None, **pipe):

        n = default if default is not None else n

        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=3)
        sample = sampler.random(n=n)
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

        # Create empty pipes list
        pipes = []

        for i, sample in enumerate(sample_scaled):
            new_pipe = pipe.copy()
            new_pipe['weights'] = sample
            if seed is not None:
                new_pipe['seed'] = seed
                seed += 1
            pipes.append(new_pipe)

        if len(pipes) == 0:
            raise ValueError("Empty pipe")
        elif len(pipes) == 1:
            # Just return pipe
            return pipes[0]
        else:
            return pipes


class Y(Module):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self, Y=2, seed=None,
                 default=None, call_number=None, call_total=None, **pipe):

        Y = default if default is not None else Y
        if Y == 1:
            pipe['weights'] = [1, 1, 0.05]
            pipe['rock_heights'] = [0.2, 0.2, 0.2, 0.2]

        if Y == 2:
            pipe['weights'] = [3, 3, 0.1]
        if Y == 3:
            pipe['weights'] = [5, 4, 0.1]
        if Y == 4:
            pipe['weights'] = [10, 6, 0.2]
        if Y == 5:
            pipe['weights'] = [15, 8, 0.3]

        return pipe


class PlotObstacles(Module):
    ''' Plot obstacles '''
    @debug_decorator
    def __call__(self, size=None, position=[[0, 0]], height=[1], yaw_deg=[0],
                 width=[5], aspect=[1], pitch_deg=[0],
                 filename=None, default=None,
                 exportmode=False, dpi=200,
                 **kwargs):
        # Setup filename
        filename = default if default is not None else filename
        if filename is None:
            filename = f'obstacles{self.file_id}.png'

        filename = os.path.join(self.save_dir, filename)

        from utils.obstacles import Obstacles
        from utils.plots import plot_obstacles
        obstacles = Obstacles()

        obstacles.position = np.array(position).T
        obstacles.height = np.array(height).reshape(1, -1)
        obstacles.yaw_deg = np.array(yaw_deg).reshape(1, -1)
        obstacles.width = np.array(width).reshape(1, -1)
        obstacles.aspect = np.array(aspect).reshape(1, -1)
        obstacles.pitch_deg = np.array(pitch_deg).reshape(1, -1)

        extent = [-size/2, size/2, -size/2, size/2]  # TODO: Temporary fix

        fig, ax = plot_obstacles(
            obstacles, xlim=extent[:2], ylim=extent[:-2],
            color=obstacles.height, cmap=cc.cm.coolwarm,
        )
        fig.savefig(filename)
        if exportmode:
            save_all_axes(fig, filename, delta=0.0, dpi=dpi)
            filename = filename.replace('.png', f'_{0:01d}.png')
            # Possibly use as image texture
            return {'texture': filename}


class SaveObstacles(Module):
    ''' Save obstacles '''
    @debug_decorator
    def __call__(self, size=None, position=[[0, 0]], height=[1], yaw_deg=[0],
                 width=[5], aspect=[1], pitch_deg=[0],
                 filename='obstacles.npz', default=None,
                 overwrite=False,
                 **kwargs):
        # Setup filename
        filename = default if default is not None else filename
        filename = os.path.join(self.save_dir, filename)

        # Skip if already exists, unless 'overwrite'
        if os.path.exists(filename) and not overwrite:
            return

        from utils.obstacles import Obstacles
        obstacles = Obstacles()

        obstacles.position = np.array(position).T
        obstacles.height = np.array(height).reshape(1, -1)
        obstacles.yaw_deg = np.array(yaw_deg).reshape(1, -1)
        obstacles.width = np.array(width).reshape(1, -1)
        obstacles.aspect = np.array(aspect).reshape(1, -1)
        obstacles.pitch_deg = np.array(pitch_deg).reshape(1, -1)

        if '.yaml' in filename:
            obstacles.save_yaml(filename)
        else:
            obstacles.save_numpy(filename)


class Set(Module):
    create_folder = False
    '''
    Set a parameter as Set:parameter=value
    '''
    @debug_decorator
    def __call__(self, default, **_):
        import ast

        # Split the string into key and value
        key, value = default.split('=')
        # Use ast.literal_eval to safely evaluate the value
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass

        # Create the dictionary
        result_dict = {key: value}

        return result_dict


class SetDistribution(Module):
    create_folder = False
    '''
    Set a distribution as SetDistribution:parameter=distribution(*args)
    '''
    @debug_decorator
    def __call__(self, default, **_):
        # Parse distribution, and return
        from utils.utils import parse_and_assign_distribution
        varname, dist_obj = parse_and_assign_distribution(default)

        return {f'{varname}_distribution': dist_obj}


class Sample(Module):
    create_folder = False
    '''
    Sample a parameter from a distribution
    '''
    @debug_decorator
    def __call__(self, default, **kwargs):
        import ast
        parameter_name = default

        # Get corresponding distribution
        distribution = kwargs[f'{parameter_name}_distribution']

        return {parameter_name: distribution(size=1)}


class SetLogger(Module):
    """Activate and configure the shared logger."""
    create_folder = False

    def __call__(self, default=None, level="info", **_):
        import logging
        # import sys
        logger = logging.getLogger("artificial_terrains")

        level = default if default is not None else level

        # Remove any old handlers to avoid duplicates
        # logger.handlers.clear()

        # Normalize level string → numeric constant
        levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
            "off": logging.CRITICAL + 1,
        }
        level_value = levels.get(str(level).lower(), logging.INFO)
        logger.setLevel(level_value)

        # Create console handler
        # handler = logging.StreamHandler(sys.stdout)
        # formatter = logging.Formatter("%(levelname)s: %(message)s")
        # handler.setFormatter(formatter)
        # logger.addHandler(handler)
        # logger.info(f"Logger activated (level={level.upper()})")


class Random(Module):
    create_folder = False
    '''
    Set random position, height, width, yaw, etc.
    '''
    @debug_decorator
    def __call__(
            self, number_of_values=3,
            size=None, grid_size=None, resolution=None, extent=None,
            position_x_distribution=None,
            position_y_distribution=None,
            height_distribution=Distribution('uniform', 1, 5),
            width_distribution=Distribution('uniform', 2, 10),
            aspect_distribution=Distribution('uniform', 0.5, 1.5),
            yaw_deg_distribution=Distribution('uniform', 0, 360),
            pitch_deg_distribution=Distribution('uniform', 0, 30),
            position_probability_2d=None,
            terrain_temp=[],
            seed=None,
            default=None, **kwargs):

        to_generate = ['position', 'height', 'yaw_deg', 'width', 'aspect',
                       'pitch_deg']

        if seed is not None:
            np.random.seed(seed=seed)

        pipe = {}

        # Parse default input
        if default is not None and isinstance(default, (int, float)):
            number_of_values = default
        if default is not None and isinstance(default, (str)):
            if '[' not in default:
                to_generate = [default]
            else:
                to_generate = default

        from utils.artificial_shapes import determine_extent_and_resolution
        extent, (N_x, N_y) = determine_extent_and_resolution(
            resolution, size, grid_size, extent)

        # Setup distributions that depend on other parameters
        if position_x_distribution is None:
            position_x_distribution = Distribution('uniform', extent[0], extent[1])
        if position_y_distribution is None:
            position_y_distribution = Distribution('uniform', extent[2], extent[3])

        # test. Return weights given Random:weights
        if default == 'weights':
            num_octaves = len(terrain_temp)
            weights = self.get_weights(num_octaves)
            return {'weights': weights}

        # Generate position
        if 'position' in to_generate:
            pos_x = position_x_distribution(size=number_of_values)
            pos_y = position_y_distribution(size=number_of_values)
            pipe['position'] = np.stack([pos_x, pos_y], axis=-1)

            # Generate from position_probability_2d if given
            if position_probability_2d is not None:
                pipe['position'] = self.points_from_probability(
                    position_probability_2d, number_of_values)

        name_to_distribution_mapping = {
            'width': width_distribution,
            'height': height_distribution,
            'aspect': aspect_distribution,
            'yaw_deg': yaw_deg_distribution,
            'pitch_deg': pitch_deg_distribution,
        }

        # Loop through parameters, and run their distribution function
        for name, distribution in name_to_distribution_mapping.items():
            if name in to_generate:
                lookup_name = f'{name}_lookup_function'
                if lookup_name in kwargs:
                    # Get values using lookup array and list of positions
                    from utils.interpolator import Interpolator
                    terrain = kwargs[lookup_name]
                    interpolator = Interpolator(terrain.array, terrain.extent)
                    position = pipe['position']
                    sampled_values = interpolator.interpolator(position)
                    pipe[name] = sampled_values

                else:
                    # Simply call the distribution
                    pipe[name] = distribution(size=number_of_values)

        return pipe

    def points_from_probability(self, prob: Terrain, number_of_values):
        from utils.noise import generate_points_from_2D_prob
        # Generate integers between (0, 0) and prob.array.shape
        points = generate_points_from_2D_prob(prob.array, number_of_values)
        # Add uniform random noise in [0, 1)
        points = points + np.random.uniform(size=points.shape)

        # Map to extent
        x = np.interp(points[0], [0, prob.array.shape[0]], prob.extent[:2])
        y = np.interp(points[1], [0, prob.array.shape[1]], prob.extent[2:4])

        # Merge into combined array again
        position = np.array([x, y]).T

        return position

    def get_weights(self, num_octaves, start=128, amplitude_start=10,
                    persistance=0.60, random_amp=0.5, random_sign=True, **_):
        ''' Generate a list of amplitudes '''
        # Generate amplitude list
        amplitude_start_log = np.log(amplitude_start)/np.log(persistance)
        amplitude_end_log = amplitude_start_log+num_octaves-1
        amplitude_list = np.logspace(amplitude_start_log, amplitude_end_log,
                                     num_octaves, base=persistance)

        # Possibly multiply amplitudes by random perturbations
        if random_amp:
            random_factor = np.random.normal(
                loc=1.0, scale=random_amp, size=amplitude_list.shape)
            amplitude_list *= random_factor
        if random_sign:
            random_sign = np.random.choice([-1, 1], size=amplitude_list.shape)
            amplitude_list *= random_sign

        return amplitude_list


class LoadObstacles(Module):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self,
                 filename='obstacles.npz',
                 extent=None,
                 default=None,
                 call_number=None,
                 call_total=None,
                 exportmode=None,
                 overwrite=False,
                 **pipe):
        filename = default if default is not None else filename

        from utils.obstacles import Obstacles

        if 'yaml' in filename:
            obstacles = Obstacles.from_yaml(filename)
        else:
            obstacles = Obstacles.from_numpy(filename)

        # Filter
        # pick = (obstacles.height > 0.29).squeeze()
        # obstacles = obstacles[pick]

        # Add to pipe, in same manner as the 'Random' module does

        pipe['position'] = obstacles.position.T
        # For these, we remove one dimension
        pipe['width'] = obstacles.width[0, :]
        pipe['height'] = obstacles.height[0, :]
        pipe['yaw_deg'] = obstacles.yaw_deg[0, :]
        pipe['aspect'] = obstacles.aspect[0, :]
        pipe['pitch_deg'] = obstacles.pitch_deg[0, :]

        return pipe


class RemoveDistantObstacles(Module):
    """
    Filter obstacles based on their distance

    Parameters:
    - distance_limit (float): The maximum distance an obstacle can be from
      the extent to be kept. Obstacles beyond this limit are removed.
    - extent (list): A list or tuple defining the bounds of the region in
      which obstacles should be kept. Should be in the format [xmin, xmax, ymin, ymax].
    - default (optional): If set, it overrides the default distance_limit.
    """
    create_folder = False

    @debug_decorator
    def __call__(self,
                 distance_limit=100,
                 extent=None,
                 default=None,
                 **pipe):
        from utils.terrains import distances_to_extent
        # Use the provided default distance_limit if set, otherwise use the
        # given value for distance_limit.
        distance_limit = default if default is not None else distance_limit

        # Calculate the closest distance to the extent for all positions
        distances = distances_to_extent(pipe['position'], extent)

        # Determine which obstacles are within the distance limit
        keep = (distances <= distance_limit)

        result = {}
        # Filter the obstacle data based on the 'keep' boolean mask
        for name in [
                'position', 'width', 'height',
                'yaw_deg', 'aspect', 'pitch_deg']:
            result[name] = pipe[name][keep]

        return result
