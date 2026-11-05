from .module import Module, debug_decorator
from typing import Any, Literal
import numpy as np
from ..utils.terrains import Terrain

FunctionName = Literal[
    'Gaussian', 'Step', 'Donut', 'Plane', 'Sphere', 'Cube',
    'SmoothStep', 'Sine', 'Crater', 'SmallLunarCrater', 'Path']
PathDirection = Literal['vertical', 'horizontal', 'sine', 'random']
PathPoints = list[tuple[float, float]] | list[list[float]]
# One value, or one value per generated feature
Feature = float | list[float] | None
Position = tuple[float, float] | list[float] | list[list[float]]

FEATURE_KEYS: tuple[str, ...] = (
    'position', 'height', 'yaw_deg', 'width', 'aspect', 'pitch_deg')


def iter_kwargs(kwargs: dict[str, Any]):
    """Yield kwargs for each generated feature."""
    feature_lengths = [len(kwargs[key]) for key in FEATURE_KEYS if key in kwargs]
    max_length = max(feature_lengths, default=1)

    for index in range(max_length):
        yielded_kwargs = {}
        for key, value in kwargs.items():
            if key in {'points', 'cross_section'} or isinstance(
                    value, (str, dict, Terrain)):
                yielded_kwargs[key] = value
                continue

            try:
                values = list(value)
            except TypeError:
                yielded_kwargs[key] = value
                continue

            # Only the six feature keys determine how many calls to make.
            # Other iterables are sampled at the same index only when they
            # match that feature count; otherwise they are passed through.
            if key not in FEATURE_KEYS and len(values) != max_length:
                yielded_kwargs[key] = value
                continue

            if index < len(values):
                yielded_kwargs[key] = values[index]

        yield yielded_kwargs


class Generative(Module):
    create_folder = False
    """Generate one or more terrains from a shape function."""
    # Set by subclasses; None means a random function is chosen
    function_name = None

    @debug_decorator
    def __call__(
            self, default=None, *,
            position: Position = [[0, 0]],
            height: Feature = [],
            yaw_deg: Feature = [],
            width: Feature = [],
            aspect: Feature = [],
            pitch_deg: Feature = [],
            function_name: FunctionName | None = None,
            terrain_temp=None, size=None, grid_size=None, resolution=None,
            extent=None, reset_input=True, **kwargs):
        """Sample kwargs and evaluate the selected shape function."""
        terrain_temp = [] if terrain_temp is None else terrain_temp
        functions = [
            'Gaussian', 'Step', 'Donut', 'Plane', 'Sphere', 'Cube',
            'SmoothStep', 'Sine', 'Crater', 'SmallLunarCrater',
        ]
        if function_name is None:
            function_name = self.function_name
        if function_name is None:
            function_name = np.random.choice(functions)

        assert function_name in functions + ['Path']

        from ..utils.artificial_shapes import determine_extent_and_resolution
        extent, (N_x, N_y) = determine_extent_and_resolution(
            resolution, size, grid_size, extent)

        kwargs = {
            'position': position,
            'height': height,
            'yaw_deg': yaw_deg,
            'width': width,
            'aspect': aspect,
            'pitch_deg': pitch_deg,
            **kwargs,
        }

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
    function_name = 'Gaussian'


class Step(Generative):
    function_name = 'Step'


class Donut(Generative):
    function_name = 'Donut'


class Plane(Generative):
    function_name = 'Plane'


class Sphere(Generative):
    function_name = 'Sphere'


class Cube(Generative):
    function_name = 'Cube'


class SmoothStep(Generative):
    function_name = 'SmoothStep'


class Sine(Generative):
    function_name = 'Sine'


class Crater(Generative):
    function_name = 'Crater'


class SmallLunarCrater(Generative):
    function_name = 'SmallLunarCrater'


class Path(Module):
    """Set polyline geometry for a later path-based generating module."""
    create_folder = False

    @debug_decorator
    def __call__(
            self,
            default: PathDirection | PathPoints | None = None, *,
            direction: PathDirection | PathPoints = 'vertical',
            points: PathPoints | None = None, extent=None,
            amplitude=None, wavelength=None, heading=0.0, angle=None,
            phase=0.0, samples=64, curvature=None, seed=None):
        direction = default if default is not None else direction
        if points is None and not isinstance(direction, str):
            points = direction

        if points is None:
            if extent is None:
                raise ValueError("Path requires the current terrain extent.")
            x_min, x_max, y_min, y_max = extent
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            if direction == 'vertical':
                points = [(center_x, y_min), (center_x, y_max)]
            elif direction == 'horizontal':
                points = [(x_min, center_y), (x_max, center_y)]
            elif direction == 'sine':
                samples = int(samples)
                if samples < 2:
                    raise ValueError("Path samples must be at least two.")
                size_x = x_max - x_min
                size_y = y_max - y_min
                amplitude = (
                    0.1 * min(size_x, size_y)
                    if amplitude is None else amplitude
                )
                heading = heading if angle is None else angle
                heading_rad = np.deg2rad(heading)
                tangent = np.array([np.cos(heading_rad), np.sin(heading_rad)])
                normal = np.array([-tangent[1], tangent[0]])
                half_span = 0.5 * (
                    abs(tangent[0]) * size_x + abs(tangent[1]) * size_y)
                wavelength = half_span if wavelength is None else wavelength
                if amplitude < 0 or wavelength <= 0:
                    raise ValueError(
                        "Sine path amplitude must be non-negative and wavelength positive."
                    )
                along = np.linspace(-half_span, half_span, samples)
                offsets = amplitude * np.sin(
                    2 * np.pi * along / wavelength + phase)
                center = np.array([center_x, center_y])
                points = center + along[:, None] * tangent + offsets[:, None] * normal
            elif direction == 'random':
                samples = int(samples)
                if samples < 2:
                    raise ValueError("Path samples must be at least two.")
                size_x = x_max - x_min
                size_y = y_max - y_min
                path_length = np.hypot(size_x, size_y)
                step = path_length / (samples - 1)
                curvature = (
                    1.0 / max(size_x, size_y)
                    if curvature is None else curvature
                )
                if curvature < 0:
                    raise ValueError("Random path curvature must be non-negative.")

                rng = np.random.default_rng(seed)
                points = np.empty((samples, 2), dtype=float)
                points[0] = rng.uniform([x_min, y_min], [x_max, y_max])
                heading_rad = rng.uniform(-np.pi, np.pi)
                for index in range(1, samples):
                    heading_rad += rng.uniform(-curvature, curvature) * step
                    points[index] = points[index - 1] + step * np.array([
                        np.cos(heading_rad), np.sin(heading_rad)])
            else:
                raise ValueError(
                    "Path direction must be 'vertical', 'horizontal', 'sine', 'random', "
                    "or explicit points."
                )

        path_points = np.asarray(points, dtype=float)
        if (
                path_points.ndim != 2
                or path_points.shape[1] != 2
                or len(path_points) < 2
                or not np.all(np.isfinite(path_points))):
            raise ValueError("Path points must contain at least two finite (x, y) pairs.")
        if np.any(np.sum(np.diff(path_points, axis=0)**2, axis=1) == 0):
            raise ValueError("Consecutive path points must be distinct.")

        return {'path_points': path_points.tolist()}


class WheelTracks(Generative):
    """Generate two wheel ruts along the path stored in the pipe."""

    def __call__(
            self, default: float | None = None, *, path_points=None,
            gauge: float = 1.6, track_width: float = 0.6,
            depth: float = 0.08, **kwargs):
        if path_points is None:
            raise ValueError("WheelTracks requires Path to run first.")
        gauge = default if default is not None else gauge
        if gauge <= track_width:
            raise ValueError("Wheel-track gauge must be greater than track width.")
        if track_width <= 0 or depth <= 0:
            raise ValueError("Wheel-track width and depth must be positive.")

        half_gauge = gauge / 2
        half_width = track_width / 2
        cross_section = [
            (-half_gauge - half_width, 0.0),
            (-half_gauge, -depth),
            (-half_gauge + half_width, 0.0),
            (half_gauge - half_width, 0.0),
            (half_gauge, -depth),
            (half_gauge + half_width, 0.0),
        ]
        return super().__call__(
            function_name='Path',
            points=path_points,
            cross_section=cross_section,
            **kwargs,
        )


class Function(Module):
    create_folder = False
    '''
    Execute given x, y function
    '''
    @debug_decorator
    def __call__(
            self, default=None, *, expression='x', terrain_temp=None,
            overwrite=None, size=None, grid_size=None, resolution=None,
            extent=None, **kwargs):
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
