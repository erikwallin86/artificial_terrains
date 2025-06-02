from modules.data import Module, debug_decorator
import numpy as np


class Basic(Module):
    create_folder = False

    ''' Make basic terrains
    '''
    @debug_decorator
    def __call__(self, terrain_temp=None, scale_list=[400, 32, 0.5],
                 default=None, seed=None, **kwargs):
        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        terrain_temp = [] if terrain_temp is None else terrain_temp

        # Possibly replace scaling list from 'default'
        if isinstance(default, list):
            scale_list = default
        elif isinstance(default, (float, int)):
            scale_list = [default]

        # Make sure scale_list is a list
        if isinstance(scale_list, (float, int)):
            scale_list = [scale_list]

        info_dict = {}
        for i, x in enumerate(scale_list):
            scaling = 1.75/(1+x/68)
            simplex_noise = 1/scaling * get_simplex2(
                **kwargs,
                scale_x=x, scale_y=x,
                info_dict=info_dict,
                logger_fn=self.info,
                seed=seed,
            )

            if seed is not None:
                seed += 1

            terrain = Terrain.from_array(simplex_noise, **info_dict)
            terrain_temp.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            }


class Octaves(Module):
    create_folder = False
    ''' Make terrain by combining octaves

    Args:
      num_octaves: (int) number of octaves/terrains to generate
      start (float): start scale
      amplitude_start (float): amplitude for start octave
      persistance (float): factor by which amplitude is scaled for each octave
    '''
    @debug_decorator
    def __call__(
            self, terrain_temp=None, default=None,
            num_octaves=10,
            start=128,
            persistance=0.60,
            amplitude_start=10,
            random_amp=0.5,
            random_sign=True,
            only_generate_weights=False,
            **kwargs
    ):
        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        terrain_temp = [] if terrain_temp is None else terrain_temp

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
        if random_sign:
            random_sign = np.random.choice([-1, 1], size=amplitude_list.shape)
            amplitude_list *= random_sign

        if not only_generate_weights:
            info_dict = {}
            for i, x in enumerate(scale_list):
                scaling = 1.75/(1+x/68)
                simplex_noise = 1/scaling * get_simplex2(
                    **kwargs,
                    scale_x=x, scale_y=x,
                    center=True, info_dict=info_dict,
                    logger_fn=self.info,
                )

                terrain = Terrain.from_array(simplex_noise, **info_dict)
                terrain_temp.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            # Pass weights, for use in WeightedSum
            'weights': amplitude_list,
            }


class Rocks(Module):
    create_folder = False

    ''' Test to make some rocks '''
    @debug_decorator
    def __call__(self,
                 rock_size=[0.5, 1, 2, 4],
                 rock_heights=None,
                 fraction=0.8,
                 default=None,
                 terrain_temp=None,
                 **kwargs):
        rock_size = default if default is not None else rock_size

        # Make sure 'rock-size' is a list
        if not isinstance(rock_size, list):
            rock_size = [rock_size]

        if rock_heights is None:
            rock_heights = np.divide(rock_size, 2)
            # [0.25, 0.5, 1, 2],

        terrain_temp = [] if terrain_temp is None else terrain_temp

        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        for i, (x, height) in enumerate(zip(rock_size, rock_heights)):
            info_dict = {}

            simplex_noise = get_simplex2(
                scale_x=x, scale_y=x,
                info_dict=info_dict,
                center=True,
                logger_fn=self.info,
                random_shift=False,
                **kwargs,
            )

            # max_value = np.max(simplex_noise)
            max_value = np.sqrt(3)/2

            # Pick all that is not holes/rocks
            pick = (simplex_noise < max_value*fraction)
            # Normalize, and remove all that is not holes/rocks
            simplex_noise = np.divide(
                simplex_noise - max_value*fraction, max_value*(1-fraction))
            simplex_noise[pick] = 0
            # Scale heights
            simplex_noise = simplex_noise * height

            terrain = Terrain.from_array(simplex_noise, **info_dict)
            terrain_temp.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            }


class Holes(Module):
    create_folder = False

    ''' Test to make some rocks '''
    @debug_decorator
    def __call__(self,
                 size_list=[0.5, 1, 2, 4],
                 heights=None,
                 fraction=0.8,
                 default=None,
                 terrain_temp=None,
                 **kwargs):
        size_list = default if default is not None else size_list

        if heights is None:
            heights = np.divide(size_list, 4)

        terrain_temp = [] if terrain_temp is None else terrain_temp

        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        for i, (x, height) in enumerate(zip(size_list, heights)):
            info_dict = {}
            scaling = 1.75/(1+x*0.75)
            simplex_noise = 1/scaling * get_simplex2(
                scale_x=x, scale_y=x,
                info_dict=info_dict,
                center=True,
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
            simplex_noise = simplex_noise * height

            terrain = Terrain.from_array(simplex_noise, **info_dict)
            terrain_temp.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            }
