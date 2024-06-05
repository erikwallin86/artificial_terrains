from datahandlers.data import DataHandler, debug_decorator
import numpy as np
from utils.terrains import Terrain


class Generative(DataHandler):
    create_folder = False
    '''
    Generative base
    '''
    @debug_decorator
    def __call__(
            self, function_name=None, terrain_temp=[], default=None,
            position=[[0, 0]], height=[], width=[],
            aspect=[], yaw_deg=[], pitch_deg=[],
            size=None, resolution=None, ppm=None,
            reset_input=True,
            **kwargs):
        '''

        Args:
          position: list of positions, where position is [x, y] list
          heights: list of heights, where height is float in [-infty, infty]
          yaws: list of yaw, where yaw is float in [0, 2*np.pi]
        '''
        functions = ['gaussian', 'step', 'donut', 'plane', 'sphere', 'cube',
                     'smoothstep', 'sine']
        if function_name is None:
            function_name = np.random.choice(functions)

        assert function_name in functions

        # Format arguments
        if np.array(position).ndim == 1:
            position = [position]
        if np.array(height).ndim == 0:
            height = [height]
        if np.array(yaw_deg).ndim == 0:
            yaw_deg = [yaw_deg]
        if np.array(width).ndim == 0:
            width = [width]
        if np.array(aspect).ndim == 0:
            aspect = [aspect]
        if np.array(pitch_deg).ndim == 0:
            pitch_deg = [pitch_deg]

        # Setup sizes etc.
        from utils.artificial_shapes import determine_size_and_resolution
        # Get size and resolution from any two tuples: ppm, size, resolution
        size_x, size_y, N_x, N_y = determine_size_and_resolution(
            ppm, size, resolution)

        properties = {
            'position': position,
            'height': height,
            'yaw_deg': yaw_deg,
            'width': width,
            'aspect': aspect,
            'pitch_deg': pitch_deg,
        }

        keys = list(properties.keys())

        from itertools import zip_longest
        from utils.artificial_shapes import get_meshgrid
        from utils.artificial_shapes import FUNCTIONS

        for values in zip_longest(*properties.values()):
            # Create a dictionary for each group, excluding any None values
            kwargs = {k: v for k, v in zip(keys, values) if v is not None}
            print(f"kwargs:{kwargs}")

            # Setup function callable
            function_callable = FUNCTIONS[function_name](**kwargs)

            # Get meshgrid
            X, Y = get_meshgrid(size_x, size_y, N_x, N_y)

            # Get heights from function callable
            heights_array = function_callable(X, Y)

            # Make terrain, and add to dict
            terrain = Terrain.from_array(heights_array)
            # desctiption = f'{kwargs.__repr__()}'
            desctiption = f'{self.name}{self.file_id}_{kwargs.__repr__()}'
            terrain_temp.append(terrain)

        print(f"### OUTPUT len(terrain_temp):{len(terrain_temp)}")

        pipe = {
            'terrain_temp': terrain_temp,
        }

        if reset_input:
            pipe['position'] = None
            pipe['height'] = None
            pipe['yaw_deg'] = None
            pipe['width'] = None
            pipe['aspect'] = None
            pipe['pitch_deg'] = None

        return pipe

        # if len(terrain_temp) > 1:
        #     return {'terrain_temp': terrain_temp}
        # else:
        #     return {'terrain': terrain}


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
