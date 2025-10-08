from modules.data import Module, debug_decorator
import numpy as np
import os
from utils.utils import get_terrains, get_terrain


class Negate(Module):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 default=None, last=None, **_):
        terrains = get_terrains(
            terrain_temp, terrain_prim, last, remove=False,
            print_fn=self.info)

        # Negate the terrains
        for terrain in terrains:
            terrain.array = -terrain.array


class Scale(Module):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 factor=2, default=None, last=None, **_):
        factor = default if default is not None else factor

        terrains = get_terrains(
            terrain_temp, terrain_prim, last, remove=False,
            print_fn=self.info)

        # Scale the terrains
        for terrain in terrains:
            terrain.array *= factor


class Add(Module):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 term=1, default=None, last=None, **_):
        term = default if default is not None else term

        terrains = get_terrains(
            terrain_temp, terrain_prim, last, remove=False,
            print_fn=self.info)

        # Scale the terrains
        for terrain in terrains:
            terrain.array += term


class Absolute(Module):
    ''' Apply the absolute operation on the terrain '''

    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 default=None, last=None, **_):
        terrains = get_terrains(
            terrain_temp, terrain_prim, last, remove=False,
            print_fn=self.info)

        # Scale the terrains
        for terrain in terrains:
            terrain.array = np.absolute(terrain.array)


class Clip(Module):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 clip_min=0, clip_max=1,
                 default=None, last=None, **_):
        ''' Clip terrains '''

        terrains = get_terrains(
            terrain_temp, terrain_prim, last, remove=False,
            print_fn=self.info)

        # Clip
        for terrain in terrains:
            terrain.array = np.clip(terrain.array, clip_min, clip_max)


class Around(Module):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 around=1, default=None, last=None, **_):
        ''' Move terrain in z to place mean at some value '''
        # Get value from 'default'
        around = default if default is not None else around

        terrains = get_terrains(
            terrain_temp, terrain_prim, last, remove=False,
            print_fn=self.info)

        # Move to place mean around 1
        for terrain in terrains:
            terrain.array = terrain.array + around - np.mean(terrain.array)


class Around1(Around):
    def __call__(self, *args, around=None, **kwargs):
        super().__call__(*args, around=1, **kwargs)


class AsProbability(Module):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 default=None, last=None, **_):
        ''' Use terrain as a 2d probability '''
        # Get and remove terrain from dict/heap
        terrain = get_terrain(
            terrain_temp, terrain_prim, print_fn=self.info)

        # Clip to remove negative values
        terrain.array = np.clip(terrain.array, 0, None)
        # Normalize
        terrain.array = terrain.array/np.sum(terrain.array)

        return {'position_probability_2d': terrain}


class AsLookupFor(Module):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 parameter='width',
                 default=None, last=None, **_):
        ''' Use terrain as function f(x, y) to sample values given positions '''
        # Use default input for the parameter
        parameter = default if default is not None else parameter

        # Get and remove terrain from dict/heap
        terrain = get_terrain(
            terrain_temp, terrain_prim, print_fn=self.info)

        return {f'{parameter}_lookup_function': terrain}


class AsFactor(Module):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 default=None, last=None, **_):
        ''' Use terrain as a 2d factor '''
        # Get and remove terrain from dict/heap
        terrain = get_terrain(
            terrain_temp, terrain_prim, print_fn=self.info, remove=True)

        return {'factor': terrain.array}


class BezierRemap(Module):
    ''' Nonlinear remapping of values using bezier curves

    Uses a 3rd order bezier curve, with p0=(0, 0) and p3=(1, 1).
    p1, and p2 are free, and are used to shape the bezier curve.
    p1 and p2 are set given 1, 2, or 4 input numbers.

    Args:
      bezier_args: (float, or list of length 1, 2, or 4)
        a float in [-1, 1]:  p1=p2=[a, 1-a]
        [a, b] floats in [-1, 1]:  p1=p2=[a, b]
        [a, b, c, d] floats in [-1, 1]:  p1=[a, b], p2=[c, d]

    '''
    create_folder = True

    @debug_decorator
    def __call__(self, bezier_args=0.5, terrain_temp=[], terrain_prim=[],
                 default=None, plot=False, call_number=None, call_total=None,
                 last=None, **pipe):

        bezier_args = default if default is not None else bezier_args
        # Make sure a is a list
        if not isinstance(bezier_args, list):
            bezier_args = [bezier_args]

        terrains = get_terrains(
            terrain_temp, terrain_prim, last, remove=False,
            print_fn=self.info)

        # Get bezier curve
        t_array = np.linspace(0, 1, 50)
        xp, fp, points = self.bezier_function(t_array, *bezier_args)

        # plot
        if plot:
            from utils.plots import new_fig
            fig, ax = new_fig()
            ax.plot(xp, fp)
            filename = os.path.join(self.save_dir, 'bezier.png')
            ax.scatter(*points.T)
            for i, point in enumerate(points):
                ax.annotate(i, point)
            fig.savefig(filename)

        # Apply nonlinear bezier remapping
        for terrain in terrains:
            # Normalize array
            z_min = np.min(terrain.array)
            z_max = np.max(terrain.array)
            normalized_array = (terrain.array - z_min) / (z_max - z_min)

            # Use bezier curve to do nonlinear remapping
            transformed_array = np.interp(normalized_array, xp, fp)

            # Rescale to original limits
            rescaled_array = transformed_array * (z_max - z_min) + z_min

            # Assign to terrain
            terrain.array = rescaled_array

    @staticmethod
    def bezier_function(t, *args, p0=[0, 0], p1=None, p2=None, p3=[1, 1]):
        if p1 is None or p2 is None:
            # Set from a
            if len(args) == 1:
                a = args[0]
                p1 = [a, 1-a]
                p2 = [a, 1-a]

            elif len(args) == 2:
                a, b = args
                p1 = [a, b]
                p2 = [a, b]
            elif len(args) == 4:
                a, b, c, d = args
                p1 = [a, b]
                p2 = [c, d]

        p0 = np.array(p0).reshape(-1, 2)
        p1 = np.array(p1).reshape(-1, 2)
        p2 = np.array(p2).reshape(-1, 2)
        p3 = np.array(p3).reshape(-1, 2)

        t = t.reshape(-1, 1)

        curve = (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3
        xp = curve[:, 0]
        fp = curve[:, 1]

        points = [p0.squeeze(), p1.squeeze(), p2.squeeze(), p3.squeeze()]

        return np.array(xp), np.array(fp), np.array(points)


class Smooth(Module):
    """
    Perform Gaussian smoothing
    """
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 default=None, last=None, sigma_meter=5, **_):
        from scipy.ndimage import gaussian_filter

        # Possibly set sigma (m) using default value
        sigma_meter = default if default is not None else sigma_meter
        self.info(f"smooth using sigma: {sigma_meter} (m)")

        # Get terrains
        terrains = get_terrains(
            terrain_temp, terrain_prim, last, remove=False,
            print_fn=self.info)

        # Loop the terrains, and perform Gaussian smoothing
        for terrain in terrains:
            resolution = terrain.resolution  # (dy, dx)
            sigma_grid = np.divide(sigma_meter, resolution)

            terrain.array = gaussian_filter(terrain.array, sigma=sigma_grid)
