import os
import numpy as np
from utils.terrains import Terrain


def debug_decorator(func):
    '''
    Wrapper around calls to print some debug info
    '''
    def wrapper(self, call_number=None, call_total=None, *args, **kwargs):
        if call_number is not None and call_total is not None:
            self.logger.info(f"Run {self.name} {call_number+1}/{call_total}")

        if call_total is not None and call_total > 1:
            self.file_id = f'_{call_number:05d}'
        else:
            self.file_id = ''

        result = func(
            self, *args,
            call_number=call_number,
            call_total=call_total,
            **kwargs)
        return result
    return wrapper


class DataHandler():
    create_folder = True
    '''
    Args:
    '''
    def __init__(self, save_dir=None, logger=None):
        self.save_dir = save_dir
        self.logger = logger

        self.save_dir = os.path.join(self.save_dir, self.__class__.__name__)
        if self.create_folder:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)

    def __call__(
            self, **kwargs):
        # Return kwargs which are not 'input'/'data'
        return kwargs

    def info(self, info_string):
        ''' 'Shorter' method for logger.info, with indent. '''
        self.logger.info(f"  {info_string}")

    @property
    def name(self):
        return self.__class__.__name__


class Print(DataHandler):
    ''' Print array info '''
    create_folder = False

    @debug_decorator
    def __call__(self, **kwargs):
        print(f"kwargs:{kwargs}")


class Resolution(DataHandler):
    ''' Test '''
    create_folder = False

    @debug_decorator
    def __call__(self, resolution=100, default=None, **kwargs):
        # göra så att 'default' parametern alltid står först
        # och göra det här automatiskt
        resolution = default if default is not None else resolution

        return {
            'N': resolution,
            }


class Size(DataHandler):
    ''' Test '''
    create_folder = False

    @debug_decorator
    def __call__(self, size=50, default=None, **kwargs):
        # göra så att 'default' parametern alltid står först
        # och göra det här automatiskt
        size = default if default is not None else size

        self.info(f"Set size:{size} [m]")

        return {
            'size': size,
            }


class PPM(DataHandler):
    ''' Test '''
    create_folder = False

    @debug_decorator
    def __call__(self, ppm=5, default=None, **kwargs):
        # göra så att 'default' parametern alltid står först
        # och göra det här automatiskt
        ppm = default if default is not None else ppm

        self.info(f"Set points-per-meter:{ppm}")

        return {
            'ppm': ppm,
            }


class Seed(DataHandler):
    ''' Test '''
    create_folder = False

    @debug_decorator
    def __call__(self, seed=0, default=None, **kwargs):
        # göra så att 'default' parametern alltid står först
        # och göra det här automatiskt
        seed = default if default is not None else seed

        self.info(f"Set seed:{seed}")

        return {
            'seed': seed,
            }


class MakeFlat(DataHandler):
    ''' Make basic terrains '''
    @debug_decorator
    def __call__(self, resolution=100, size=50, **kwargs):
        from utils.terrains import Terrain
        terrain = Terrain.from_array(
            np.zeros((resolution, resolution)), size=(size, size))
        filename = 'terrain.npz'
        filename = os.path.join(self.save_dir, filename)
        terrain.save(filename)


class Basic(DataHandler):
    create_folder = False

    ''' Make basic terrains

    Second version, test new interface
    '''
    @debug_decorator
    def __call__(self, **kwargs):
        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        scale_list = [400, 32, 0.5]

        terrain_dict = {}

        info_dict = {}
        for i, x in enumerate(scale_list):
            scaling = 1.75/(1+x/68)
            simplex_noise = 1/scaling * get_simplex2(
                **kwargs,
                scale_x=x, scale_y=x,
                center=False, info_dict=info_dict,
                logger_fn=self.info,
            )

            terrain = Terrain.from_array(simplex_noise, **info_dict)
            terrain_dict[x] = terrain

        return {
            'terrain_dict': terrain_dict,
            }


class Octaves(DataHandler):
    create_folder = False
    ''' Make terrain by combining octaves

    Args:
      num_octaves: (int) number of octaves/terrains to generate
      start (float): start scale
      amplitude_start (float): amplitude for start octave
      persistance (float): factor by which amplitude is scaled for each octave
    '''
    @debug_decorator
    def __call__(self, num_octaves=10, start=128, persistance=0.60,
                 amplitude_start=10, default=None, random_amp=0.5, **kwargs):
        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        # Get number of octaves from possible default input
        num_octaves = default if default is not None else num_octaves

        # Generate scale list
        start_log2 = np.log2(start)
        end_log2 = start_log2+1 - num_octaves
        scale_list = np.logspace(start_log2, end_log2, num_octaves, base=2)

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

        terrain_dict = {}

        info_dict = {}
        for i, x in enumerate(scale_list):
            scaling = 1.75/(1+x/68)
            simplex_noise = 1/scaling * get_simplex2(
                **kwargs,
                scale_x=x, scale_y=x,
                center=False, info_dict=info_dict,
                logger_fn=self.info,
            )

            terrain = Terrain.from_array(simplex_noise, **info_dict)
            terrain_dict[x] = terrain

        return {
            'terrain_dict': terrain_dict,
            # Pass weights, for use in WeightedSum
            'weights': amplitude_list,
            }


class Hypercube(DataHandler):
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


class Y(DataHandler):
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


class Combine(DataHandler):
    ''' Combined basic terrains within a terrain_dict '''
    create_folder = False

    @debug_decorator
    def __call__(self, operation='Add', terrain_dict={},
                 default=None, call_number=None, call_total=None,
                 last=None, **pipe):
        operations = ['Add', 'Max', 'Min', 'Prod']
        operation = default if default is not None else operation
        assert operation in operations, f'operation not in {operations}'

        if len(terrain_dict) > 1:
            # Work with the terrain_dict
            terrains = list(terrain_dict.values())
            self.info(f"{operation} {len(terrains)} terrains from 'dict'")
        elif 'terrain' in pipe and 'terrain_heap' in pipe:
            # Work with terrain + terrain-heap
            terrains = [pipe['terrain']] + pipe['terrain_heap'][::-1]

            # Possibly only work with last N terrains
            if last is not None:
                terrains = terrains[:last]

            self.info(f"{operation} {len(terrains)} terrains from 'heap'")
            # Remove any used terrains from terrain_heap.
            pipe['terrain_heap'] = [terrain for terrain in pipe['terrain_heap']
                                    if terrain not in terrains]
            if len(pipe['terrain_heap']) > 0:
                self.info(f"{len(pipe['terrain_heap'])} terrains remaining")

            # Remove terrain, as this now is first in terrains
            pipe['terrain'] = None

        # Apply any of the operations
        if operation == 'Add':
            array = np.sum(terrains, axis=0)
            terrain = Terrain.from_array(
                array, size=terrains[0].size, extent=terrains[0].extent)
        elif operation == 'Max':
            array = np.maximum.reduce(terrains)
            terrain = Terrain.from_array(
                array, size=terrains[0].size, extent=terrains[0].extent)
        elif operation == 'Min':
            array = np.minimum.reduce(terrains)
            terrain = Terrain.from_array(
                array, size=terrains[0].size, extent=terrains[0].extent)
        elif operation == 'Prod':
            array = np.prod(terrains, axis=0)
            terrain = Terrain.from_array(
                array, size=terrains[0].size, extent=terrains[0].extent)
        else:
            raise AttributeError

        # Possibly move existing terrain to terrain_heap
        if 'terrain' in pipe and pipe['terrain'] is not None:
            if 'terrain_heap' in pipe and isinstance(pipe['terrain_heap'], list):
                pipe['terrain_heap'].append(pipe['terrain'])
            else:
                pipe['terrain_heap'] = [pipe['terrain']]
        # Reset any terrain_dict
        pipe['terrain_dict'] = {}
        # Set the new terrain
        pipe['terrain'] = terrain

        return pipe


class CombineLast(Combine):
    ''' Combine, with only the last two terrains,

    'terrain' and the latest from the 'heap'
    '''
    def __call__(self, *args, last=None, **kwargs):
        return super(CombineLast, self).__call__(*args, last=2, **kwargs)


class WeightedSum(DataHandler):
    ''' Weighted sum of basic terrains within a terrain_dict '''
    create_folder = False

    @debug_decorator
    def __call__(self, weights=[5, 8, 0.1], terrain_dict={},
                 call_number=None, call_total=None,
                 default=None, **pipe):

        weights = default if default is not None else weights
        weights = np.array(weights).reshape(-1, 1, 1)

        self.info(f"Use weights:{weights.squeeze()}")

        terrains = list(terrain_dict.values())
        array = np.sum(np.multiply(terrains, weights), axis=0)
        terrain = Terrain.from_array(
            array, size=terrains[0].size, extent=terrains[0].extent)

        # Possibly move existing terrain to terrain_heap
        if 'terrain' in pipe and pipe['terrain'] is not None:
            if 'terrain_heap' in pipe and isinstance(pipe['terrain_heap'], list):
                pipe['terrain_heap'].append(pipe['terrain'])
            else:
                pipe['terrain_heap'] = [pipe['terrain']]
        # Reset any terrain_dict
        pipe['terrain_dict'] = {}
        # Set the new terrain
        pipe['terrain'] = terrain

        return pipe


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


class MakeObstacles(DataHandler):
    ''' Make basic terrains '''
    @debug_decorator
    def __call__(self, resolution=100, size=50, **kwargs):
        from utils.noise import get_simplex
        from utils.obstacles import Obstacles
        from utils.plots import plot_obstacles

        x = 1
        N = resolution
        scaling = 1.75/(1+x*0.75)

        simplex_noise = 1/scaling * get_simplex(
            Nx=N, Ny=N, scale_x=N*x, scale_y=N*x)

        obstacles = Obstacles(Y=3)
        filename = 'obstacles.png'
        filename = os.path.join(self.save_dir, filename)
        plot_obstacles(obstacles, filename)
        obstacles.save_numpy(filename.replace('.png', '.npz'))


class Rocks(DataHandler):
    create_folder = False

    ''' Test to make some rocks '''
    @debug_decorator
    def __call__(self,
                 rock_size=[0.5, 1, 2, 4],
                 rock_heights=None,
                 fraction=0.8,
                 default=None,
                 **kwargs):
        rock_size = default if default is not None else rock_size
        if rock_heights is None:
            rock_heights = np.divide(rock_size, 2)
            # [0.25, 0.5, 1, 2],

        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        terrain_dict = {}
        for i, (x, height) in enumerate(zip(rock_size, rock_heights)):
            info_dict = {}
            scaling = 1.75/(1+x*0.75)
            simplex_noise = 1/scaling * get_simplex2(
                scale_x=x, scale_y=x,
                info_dict=info_dict,
                center=False,
                logger_fn=self.info,
                **kwargs,
            )

            max_value = np.max(simplex_noise)
            # Pick all that is not holes/rocks
            pick = (simplex_noise < max_value*fraction)
            # Normalize, and remove all that is not holes/rocks
            simplex_noise = np.divide(
                simplex_noise - max_value*fraction, max_value*(1-fraction))
            simplex_noise[pick] = 0
            # Scale heights
            simplex_noise = simplex_noise * height

            terrain = Terrain.from_array(simplex_noise, **info_dict)
            terrain_dict[x] = terrain

        return {
            'terrain_dict': terrain_dict,
            }


class Holes(DataHandler):
    create_folder = False

    ''' Test to make some rocks '''
    @debug_decorator
    def __call__(self, size_list=[0.5, 1, 2, 4],
                 fraction=0.8,
                 default=None,
                 **kwargs):
        size_list = default if default is not None else size_list

        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        terrain_dict = {}
        for i, x in enumerate(size_list):
            info_dict = {}
            scaling = 1.75/(1+x*0.75)
            simplex_noise = 1/scaling * get_simplex2(
                scale_x=x, scale_y=x,
                info_dict=info_dict,
                center=False,
                logger_fn=self.info,
                **kwargs,
            )

            min_value = np.min(simplex_noise)
            # Pick all that is not holes/rocks
            pick = (simplex_noise > min_value*fraction)
            # Normalize, and remove all that is not holes/rocks
            simplex_noise = np.divide(
                simplex_noise - min_value*fraction, -min_value*(1-fraction))
            simplex_noise[pick] = 0
            # Scale heights
            simplex_noise = simplex_noise * (x/8)

            terrain = Terrain.from_array(simplex_noise, **info_dict)
            terrain_dict[x] = terrain

        return {
            'terrain_dict': terrain_dict,
            }


class MakePerturbations(DataHandler):
    ''' Make perturbations around some value '''
    @debug_decorator
    def __call__(self, around=0, default=None, resolution=100, size=50,
                 **kwargs):
        from utils.noise import get_simplex2
        from utils.plots import plot_image
        from utils.terrains import Terrain

        around = default if default is not None else around

        N = resolution

        slope_list = [8]
        mid_list = np.logspace(-2, 1, 4, base=2)
        small_list = np.logspace(-6, -3, 4, base=2)

        list_mapping = {
            'slope_list': slope_list,
            'mid_list': mid_list,
            'small_list': small_list,
            }

        for name, scaling_list in list_mapping.items():
            for i, x in enumerate(scaling_list):
                scaling = 1.75/(1+x*0.75)
                simplex_noise = 1/scaling * get_simplex2(
                    N_x=N, N_y=N, scale_x=x, scale_y=x, seed=1702291141015233793,
                    center=False)

                # Move to be around 'around'
                simplex_noise = np.interp(
                    simplex_noise, [-1, 1], [around-1, around+1])

                filename = f'around_{around}_{name}_{i:05d}.png'
                filename = os.path.join(self.save_dir, filename)
                plot_image(simplex_noise, filename=filename, vmin=around-1, vmax=around+1)

                terrain = Terrain.from_array(simplex_noise, size=(size, size))
                terrain.save(filename.replace('.png', '.npz'))


class MakeOctaves(DataHandler):
    ''' Make octaves '''
    @debug_decorator
    def __call__(self, resolution=100, size=50, **kwargs):
        from utils.noise import get_simplex2
        from utils.plots import plot_image
        from utils.terrains import Terrain

        N = resolution

        mid_list = np.logspace(-10, 3, 14, base=2)

        list_mapping = {
            'octaves_list': mid_list,
            }

        for name, scaling_list in list_mapping.items():
            for i, x in enumerate(scaling_list):
                scaling = 1.75/(1+x*0.75)
                simplex_noise = 1/scaling * get_simplex2(
                    N_x=N, N_y=N, scale_x=x, scale_y=x)
                filename = f'{name}_{i:05d}.png'
                filename = os.path.join(self.save_dir, filename)
                plot_image(simplex_noise, filename=filename, vmin=-1, vmax=1)
                # scale by 10
                simplex_noise = simplex_noise*10

                terrain = Terrain.from_array(simplex_noise, size=(size, size))
                terrain.save(filename.replace('.png', '.npz'))


class MakeBands(DataHandler):
    ''' Make bands '''
    @debug_decorator
    def __call__(self, resolution=100, size=50, value=0.1, default=None, **kwargs):
        from utils.noise import get_simplex2
        from utils.plots import plot_image
        from utils.terrains import Terrain

        value = default if default is not None else value

        N = resolution

        mid_list = np.logspace(-10, 3, 14, base=2)

        for i, x in enumerate(mid_list):
            scaling = 1.75/(1+x*0.75)
            simplex_noise = 1/scaling * get_simplex2(
                N_x=N, N_y=N, scale_x=x, scale_y=x)
            filename = f'band_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)

            zeros1 = (simplex_noise < value)
            zeros2 = (simplex_noise > -value)
            zeros = np.logical_and(zeros1, zeros2)

            plot_image(zeros, filename=filename, vmin=-1, vmax=1)
            simplex_noise = simplex_noise*10

            terrain = Terrain.from_array(simplex_noise, size=(size, size))
            terrain.save(filename.replace('.png', '.npz'))


class MakeEdgy(DataHandler):
    ''' Make something more edgy? '''
    @debug_decorator
    def __call__(self, resolution=100, size=50, value=0.1, default=None, dpi=400, **kwargs):
        from utils.noise import get_simplex2
        from utils.plots import plot_image
        from utils.terrains import Terrain

        value = default if default is not None else value

        N = resolution

        # mid_list = np.logspace(-10, 3, 14, base=2)
        mid_list = np.logspace(-5, 3, 9, base=2)

        for i, x in enumerate(mid_list):
            scaling = 1.75/(1+x*0.75)
            simplex_noise = 1/scaling * get_simplex2(
                N_x=N, N_y=N, scale_x=x, scale_y=x)
            filename = f'edgy_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)

            from scipy import ndimage

            antal_strukturer = 1/x  # antal strukturer
            L = resolution / antal_strukturer * 2

            simplex_noise = ndimage.zoom(simplex_noise, 1/L, order=1)
            simplex_noise = ndimage.zoom(simplex_noise, L, order=1)

            fig, ax = plot_image(simplex_noise, vmin=-1, vmax=1)
            fig.savefig(filename, dpi=dpi)

            simplex_noise = simplex_noise*10

            terrain = Terrain.from_array(simplex_noise, size=(size, size))
            terrain.save(filename.replace('.png', '.npz'))


class Test(DataHandler):
    ''' Test to add terrains of same scale '''
    @debug_decorator
    def __call__(self, resolution=100, size=50, **kwargs):
        from utils.noise import get_simplex
        from utils.plots import plot_image
        from utils.terrains import Terrain

        N = resolution

        magnitudes = []
        scaling_list = np.linspace(0.1, 0.5, 50)

        list_of_noise = []
        for i, x in enumerate(scaling_list):
            scaling = 1.75/(1+x*0.75)
            simplex_noise = 1/scaling * get_simplex(
                Nx=N, Ny=N, scale_x=N*x, scale_y=N*x)
            list_of_noise.append(simplex_noise)
            filename = f'test_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)
            plot_image(simplex_noise, filename=filename, vmin=-1, vmax=1)
            # scale by 10
            simplex_noise = simplex_noise*10
            terrain = Terrain.from_array(simplex_noise, size=(size, size))
            terrain.save(filename.replace('.png', '.npz'))

            # plot sum
            filename = f'sum_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)
            result = np.sum(list_of_noise, axis=0)
            magnitude = np.max(result) - np.min(result)
            magnitudes.append(magnitude)
            plot_image(result, filename=filename)
        print(f"magnitudes:{magnitudes}")

        from utils.plots import new_fig
        fig, ax = new_fig()
        ax.plot(magnitudes)
        fig.savefig(os.path.join(self.save_dir, 'magnitudes.png'))


class TestScaling(DataHandler):
    ''' '''
    @debug_decorator
    def __call__(self, resolution=100, size=50, **kwargs):
        from utils.noise import get_simplex2

        xs = np.logspace(-10, 18, 39, base=2)
        N = resolution

        test = []
        heights = []

        simplex_noise_list = []
        for i, x in enumerate(xs):
            print(f"(i, x):{(i, x)}")
            t = 1.75/(1+x/68)
            test.append(t)

            simplex_noise = get_simplex2(N_x=N, N_y=N, scale_x=x, scale_y=x)
            heights.append(np.max(simplex_noise)-np.min(simplex_noise))
            simplex_noise_list.append(simplex_noise)

        from utils.plots import new_fig
        fig, ax = new_fig()
        ax.plot(xs, heights, label='height')
        ax.plot(xs, test, label='test')
        ax.plot(xs, np.ones_like(xs), label='Unity')

        ax.plot(xs, np.divide(heights, test), label='heights/test')

        ax.set_xscale('log', base=2)
        ax.legend()
        filename = 'scaling.png'
        filename = os.path.join(self.save_dir, filename)
        fig.savefig(filename)


class TestBrick(DataHandler):
    ''' Test to generate terrains which can be 'bricked' together '''
    @debug_decorator
    def __call__(self, resolution=100, size=50, default=None, dpi=400, **kwargs):
        from utils.noise import get_simplex2
        from utils.plots import plot_image
        from utils.terrains import Terrain

        # res = [100, 200, 500]
        # sizes = [50, 75, 100]

        dxs = [0, 0, 0.1, 0.1]
        dys = [0, 0.1, 0, 0.1]

        x = 1/8

        info_dict = {}
        for i, (dx, dy) in enumerate(zip(dxs, dys)):
            scaling = 1.75/(1+x*0.75)
            simplex_noise = 1/scaling * get_simplex2(
                size=(size, size),
                scale_x=x, scale_y=x,
                seed=1702291141015233793,
                info_dict=info_dict,
                dx=dx, dy=dy,
            )
            print(f"info_dict:{info_dict}")
            filename = f'testbrick_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)

            fig, ax = plot_image(simplex_noise, vmin=-1, vmax=1,
                                 extent=info_dict['extent'])
            fig.savefig(filename, dpi=dpi)

            simplex_noise = simplex_noise*10

            terrain = Terrain.from_array(simplex_noise, **info_dict)
            terrain.save(filename.replace('.png', '.npz'))


class TestSimplex(DataHandler):
    ''' Test Simplex noise

    Find places which are always close to zero, to visualize the
    underlying stucture

    '''
    @debug_decorator
    def __call__(self, resolution=100, size=50, **kwargs):
        from utils.noise import get_simplex
        from utils.plots import plot_image

        N = resolution

        L = 0.1
        scaling_list = [L, L, L, L, L, L, L, L, L, L, L, L]
        list_of_zeros = []
        for i, x in enumerate(scaling_list):
            scaling = 1.75/(1+x*0.75)
            simplex_noise = 1/scaling * get_simplex(
                Nx=N, Ny=N, scale_x=N*x, scale_y=N*x)
            zeros1 = (simplex_noise < 0.1)
            zeros2 = (simplex_noise > -0.1)
            # zeros = (simplex_noise == 0)
            zeros = np.logical_and(zeros1, zeros2)
            list_of_zeros.append(zeros)

            print(f"np.sum(zeros):{np.sum(zeros)}")
            filename = f'zeros_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)
            plot_image(zeros, filename, vmin=-1, vmax=1)

        test = np.prod(list_of_zeros, axis=0)
        filename = f'test_{i:05d}.png'
        filename = os.path.join(self.save_dir, filename)
        plot_image(test, filename, vmin=-1, vmax=1)
