import os
import numpy as np
from utils.terrains import Terrain
from utils.debug import debug


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

        # Possibly debug
        debug(self, kwargs, call_number, call_total, 'input')

        result = func(
            self, *args,
            call_number=call_number,
            call_total=call_total,
            **kwargs)

        # Possibly debug
        debug(self, result, call_number, call_total, 'output')

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

        self.save_dir_original = save_dir

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
            'ppm': None,
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
            'resolution': None,
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


class Folder(DataHandler):
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


class DebugPlot(DataHandler):
    ''' Do debug plot '''
    create_folder = False

    @debug_decorator
    def __call__(self, default=None, filename='debug_plot.png', **kwargs):
        from utils.plots import debug_plot_horizontal

        filename = default if default is not None else filename
        filename = os.path.join(self.save_dir_original, filename)
        debug_plot_horizontal(filename, **kwargs)


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


class MakeObstacles(DataHandler):
    ''' Make basic terrains '''
    @debug_decorator
    def __call__(self, resolution=None, size=None, ppm=None, **kwargs):
        from utils.noise import get_simplex
        from utils.obstacles import Obstacles
        from utils.plots import plot_obstacles

        from utils.artificial_shapes import determine_size_and_resolution
        # Get size and resolution from any two tuples: ppm, size, resolution
        size_x, size_y, N_x, N_y = determine_size_and_resolution(
            ppm, size, resolution)

        x = 1
        N = resolution
        scaling = 1.75/(1+x*0.75)

        simplex_noise = 1/scaling * get_simplex(
            Nx=N, Ny=N, scale_x=N*x, scale_y=N*x)

        obstacles = Obstacles(Y=3)

        # Fix using extent...
        diff = np.array([size_x/2, size_y/2]).reshape(2, 1)
        obstacles.position = np.subtract(obstacles.position, diff)

        filename = 'obstacles.png'
        filename = os.path.join(self.save_dir, filename)
        plot_obstacles(obstacles, filename)
        obstacles.save_numpy(filename.replace('.png', '.npz'))

        # Filter out small obstacles
        pick = (obstacles.height > 0.29).squeeze()
        obstacles = obstacles[pick]

        pipe = {}
        pipe['position'] = obstacles.position.T
        # For these, we remove one dimension
        pipe['width'] = obstacles.width[0, :]
        pipe['height'] = obstacles.height[0, :]
        pipe['yaw_deg'] = obstacles.yaw_deg[0, :]
        pipe['aspect'] = obstacles.aspect[0, :]
        pipe['pitch_deg'] = obstacles.pitch_deg[0, :]

        return pipe


class PlotObstacles(DataHandler):
    ''' Plot obstacles '''
    @debug_decorator
    def __call__(self, size=None, position=[[0, 0]], height=[1], yaw_deg=[0],
                 width=[5], aspect=[1], pitch_deg=[0],
                 filename='obstacles.png', default=None,
                 **kwargs):
        # Setup filename
        filename = default if default is not None else filename
        filename = os.path.join(self.save_dir_original, filename)

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

        plot_obstacles(obstacles, filename, xlim=extent[:2], ylim=extent[:-2])


class SaveObstacles(DataHandler):
    ''' Save obstacles '''
    @debug_decorator
    def __call__(self, size=None, position=[[0, 0]], height=[1], yaw_deg=[0],
                 width=[5], aspect=[1], pitch_deg=[0],
                 filename='obstacles.npz', default=None,
                 **kwargs):
        # Setup filename
        filename = default if default is not None else filename
        filename = os.path.join(self.save_dir, filename)

        from utils.obstacles import Obstacles
        obstacles = Obstacles()

        obstacles.position = np.array(position).T
        obstacles.height = np.array(height).reshape(1, -1)
        obstacles.yaw_deg = np.array(yaw_deg).reshape(1, -1)
        obstacles.width = np.array(width).reshape(1, -1)
        obstacles.aspect = np.array(aspect).reshape(1, -1)
        obstacles.pitch_deg = np.array(pitch_deg).reshape(1, -1)

        obstacles.save_numpy(filename)


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


class Random(DataHandler):
    create_folder = False
    '''
    Set random position, height, width, yaw, etc.
    '''
    @debug_decorator
    def __call__(
            self, number_of_values=3,
            position=[[0, 0]], height=[1], yaw=[0],
            size=None,
            default=None, **kwargs):

        to_generate = ['position', 'height', 'yaw_deg', 'width', 'aspect', 'pitch_deg']

        pipe = {}

        if default is not None and isinstance(default, (int, float)):
            number_of_values = default

        # Generate position
        if 'position' in to_generate:
            pipe['position'] = np.random.uniform(
                low=-size/2, high=size/2, size=(number_of_values, 2))

        # Generate yaw
        if 'yaw_deg' in to_generate:
            pipe['yaw_deg'] = np.random.uniform(
                low=0, high=360, size=(number_of_values))

        # Generate pitch
        if 'pitch_deg' in to_generate:
            pipe['pitch_deg'] = np.random.uniform(
                low=0, high=30, size=(number_of_values))

        # Generate width
        if 'width' in to_generate:
            pipe['width'] = np.random.uniform(
                low=0, high=10, size=(number_of_values))

        # Generate height
        if 'height' in to_generate:
            pipe['height'] = np.random.uniform(
                low=0, high=5, size=(number_of_values))

        # Generate aspect
        if 'aspect' in to_generate:
            pipe['aspect'] = np.random.uniform(
                low=0.5, high=1.5, size=(number_of_values))

        print(f"pipe:{pipe}")

        return pipe


class LoadObstacles(DataHandler):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self,
                 filename='obstacles.npz',
                 default=None,
                 call_number=None,
                 call_total=None,
                 **pipe):
        filename = default if default is not None else filename

        from utils.obstacles import Obstacles
        obstacles = Obstacles.from_numpy(filename)

        # Filter
        pick = (obstacles.height > 0.29).squeeze()
        obstacles = obstacles[pick]

        # Add to pipe, in same manner as the 'Random' datahandler does

        pipe['position'] = obstacles.position.T
        # For these, we remove one dimension
        pipe['width'] = obstacles.width[0, :]
        pipe['height'] = obstacles.height[0, :]
        pipe['yaw_deg'] = obstacles.yaw_deg[0, :]
        pipe['aspect'] = obstacles.aspect[0, :]
        pipe['pitch_deg'] = obstacles.pitch_deg[0, :]

        return pipe


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
