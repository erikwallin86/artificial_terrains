from .module import Module, debug_decorator
import numpy as np
from ..utils.terrains import Terrain

FEATURE_KEYS = ('position', 'height', 'yaw_deg', 'width', 'aspect', 'pitch_deg')


def iter_kwargs(kwargs):
    """Yield kwargs for each generated feature."""
    feature_lengths = [len(kwargs[key]) for key in FEATURE_KEYS if key in kwargs]
    max_length = max(feature_lengths, default=1)

    for index in range(max_length):
        yielded_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (str, dict, Terrain)):
                yielded_kwargs[key] = value
                continue

            try:
                values = list(value)
            except TypeError:
                yielded_kwargs[key] = value
                continue

            if index < len(values):
                yielded_kwargs[key] = values[index]

        yield yielded_kwargs


class Generative(Module):
    create_folder = False
    """Generate one or more terrains from a shape function."""

    @debug_decorator
    def __call__(
            self, function_name=None, terrain_temp=None, default=None,
            size=None, grid_size=None, resolution=None, extent=None,
            reset_input=True,
            **kwargs):
        """Sample kwargs and evaluate the selected shape function."""
        terrain_temp = [] if terrain_temp is None else terrain_temp
        functions = [
            'gaussian', 'step', 'donut', 'plane', 'sphere', 'cube',
            'smoothstep', 'sine', 'crater',
        ]
        if function_name is None:
            function_name = np.random.choice(functions)

        assert function_name in functions

        from ..utils.artificial_shapes import determine_extent_and_resolution
        extent, (N_x, N_y) = determine_extent_and_resolution(
            resolution, size, grid_size, extent)

        feature_defaults = {
            'position': [[0, 0]],
            'height': [],
            'yaw_deg': [],
            'width': [],
            'aspect': [],
            'pitch_deg': [],
        }
        kwargs = {**feature_defaults, **kwargs}

        if np.array(kwargs['position']).ndim == 1:
            kwargs['position'] = [kwargs['position']]

        for key in FEATURE_KEYS:
            if key == 'position':
                continue
            if np.array(kwargs[key]).ndim == 0:
                kwargs[key] = [kwargs[key]]

        from ..utils.artificial_shapes import get_meshgrid
        from ..utils.artificial_shapes import FUNCTIONS
        function_factory = FUNCTIONS[function_name]

        # Iterate feature-wise kwargs, while repeating scalar context unchanged.
        for feature_kwargs in iter_kwargs(kwargs):
            self.logger.debug(f"kwargs:{feature_kwargs}")

            function_callable = function_factory(**feature_kwargs)
            X, Y = get_meshgrid(extent, N_x, N_y)
            heights_array = function_callable(X, Y)

            terrain = Terrain.from_array(heights_array, extent=extent)
            terrain_temp.append(terrain)

        pipe = {'terrain_temp': terrain_temp}

        if reset_input:
            pipe['position'] = [[0, 0]]
            for parameter in FEATURE_KEYS:
                if parameter == 'position':
                    continue
                pipe[parameter] = []

        return pipe


class Gaussian(Generative):
    def __call__(self, *args, **kwargs):
        return super().__call__(
            *args, function_name='gaussian', **kwargs)


class Step(Generative):
    def __call__(self, *args, **kwargs):
        return super().__call__(
            *args, function_name='step', **kwargs)


class Donut(Generative):
    def __call__(self, *args, **kwargs):
        return super().__call__(
            *args, function_name='donut', **kwargs)


class Plane(Generative):
    def __call__(self, *args, **kwargs):
        return super().__call__(
            *args, function_name='plane', **kwargs)


class Sphere(Generative):
    def __call__(self, *args, **kwargs):
        return super().__call__(
            *args, function_name='sphere', **kwargs)


class Cube(Generative):
    def __call__(self, *args, **kwargs):
        return super().__call__(
            *args, function_name='cube', **kwargs)


class SmoothStep(Generative):
    def __call__(self, *args, **kwargs):
        return super().__call__(
            *args, function_name='smoothstep', **kwargs)


class Sine(Generative):
    def __call__(self, call_number=None, call_total=None, *args, **kwargs):
        return super().__call__(
            *args, function_name='sine', call_number=call_number,
            call_total=call_total, **kwargs)


class Crater(Generative):
    def __call__(self, *args, **kwargs):
        return super().__call__(
            *args, function_name='crater', **kwargs)


class Function(Module):
    create_folder = False
    '''
    Execute given x, y function
    '''
    @debug_decorator
    def __call__(
            self,
            expression='x',
            terrain_temp=None, default=None, overwrite=None,
            size=None, grid_size=None, resolution=None, extent=None,
            **kwargs):
        '''
        '''
        expression = default if default is not None else expression

        terrain_temp = [] if terrain_temp is None else terrain_temp

        # Setup sizes etc.
        from ..utils.artificial_shapes import determine_extent_and_resolution
        # Get size and resolution from any two tuples: resolution, size, resolution
        extent, (N_x, N_y) = determine_extent_and_resolution(
            resolution, size, grid_size, extent)

        from ..utils.artificial_shapes import get_meshgrid

        # Get meshgrid
        X, Y = get_meshgrid(extent, N_x, N_y)

        safe_locals = {
            'x': X,
            'y': Y,
            'r': np.sqrt(np.square(X) + np.square(Y)),
            'np': np  # Optional: allow np.sin, np.sqrt, etc.
        }

        # Evaluate the expression
        heights_array = eval(expression, {"__builtins__": {}}, safe_locals)

        terrain = Terrain.from_array(heights_array, extent=extent)
        # desctiption = f'{kwargs.__repr__()}'
        desctiption = f'{self.name}{self.file_id}_{kwargs.__repr__()}'
        terrain_temp.append(terrain)

        pipe = {
            'terrain_temp': terrain_temp,
        }

        return pipe
