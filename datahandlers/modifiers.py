from datahandlers.data import DataHandler, debug_decorator
import numpy as np
import os
from utils.utils import get_terrains, get_terrain


class Negate(DataHandler):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_dict={}, terrain_heap=[],
                 default=None, last=None, **_):
        terrains = get_terrains(
            terrain_dict, terrain_heap, last, remove=False)

        # Negate the terrains
        for terrain in terrains:
            terrain.array = -terrain.array


class Around(DataHandler):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_dict={}, terrain_heap=[],
                 around=1, default=None, last=None, **_):
        ''' Move terrain in z to place mean at some value '''
        # Get value from 'default'
        around = default if default is not None else around

        terrains = get_terrains(
            terrain_dict, terrain_heap, last, remove=False)

        # Move to place mean around 1
        for terrain in terrains:
            terrain.array = terrain.array + around - np.mean(terrain.array)


class Around1(Around):
    def __call__(self, *args, around=None, **kwargs):
        super().__call__(*args, around=1, **kwargs)


class AsProbability(DataHandler):
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_dict={}, terrain_heap=[],
                 default=None, last=None, **_):
        ''' Use terrain as a 2d probability '''
        # Get and remove terrain from dict/heap
        terrain = get_terrain(
            terrain_dict, terrain_heap, print_fn=print)

        # Clip to remove negative values
        terrain.array = np.clip(terrain.array, 0, None)
        # Normalize
        terrain.array = terrain.array/np.sum(terrain.array)

        return {'position_probability_2d': terrain}


class BezierRemap(DataHandler):
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
    def __call__(self, bezier_args=0.5, terrain_dict={},
                 default=None, plot=False, call_number=None, call_total=None,
                 last=None, **pipe):

        bezier_args = default if default is not None else bezier_args
        # Make sure a is a list
        if not isinstance(bezier_args, list):
            bezier_args = [bezier_args]

        if len(terrain_dict) > 1:
            # Work with the terrain_dict
            terrains = list(terrain_dict.values())
        elif 'terrain' in pipe and 'terrain_heap' in pipe:
            # Work with terrain + terrain-heap
            terrains = [pipe['terrain']] + pipe['terrain_heap'][::-1]
            # Possibly only work with last N terrains
            if last is not None:
                terrains = terrains[:last]
            # Remove any used terrains from terrain_heap.
            pipe['terrain_heap'] = [terrain for terrain in pipe['terrain_heap']
                                    if terrain not in terrains]
            if len(pipe['terrain_heap']) > 0:
                self.info(f"{len(pipe['terrain_heap'])} terrains remaining")

        # Remove terrain, as this now is first in terrains
        pipe['terrain'] = None

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
