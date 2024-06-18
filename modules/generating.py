from modules.data import Module, debug_decorator
import numpy as np


class Basic(Module):
    create_folder = False

    ''' Make basic terrains
    '''
    @debug_decorator
    def __call__(self, terrain_temp=[], scale_list=[400, 32, 0.5],
                 default=None, **kwargs):
        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        # Possibly replace scaling list from 'default'
        if isinstance(default, list):
            scale_list = default
        elif isinstance(default, (float, int)):
            scale_list = [default]

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
            self, terrain_temp=[], default=None,
            num_octaves=10,
            start=128,
            persistance=0.60,
            amplitude_start=10,
            random_amp=0.5,
            random_sign=True,
            **kwargs
    ):
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
        if random_sign:
            random_sign = np.random.choice([-1, 1], size=amplitude_list.shape)
            amplitude_list *= random_sign

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
                 terrain_temp=[],
                 **kwargs):
        rock_size = default if default is not None else rock_size
        if rock_heights is None:
            rock_heights = np.divide(rock_size, 2)
            # [0.25, 0.5, 1, 2],

        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        for i, (x, height) in enumerate(zip(rock_size, rock_heights)):
            info_dict = {}
            scaling = 1.75/(1+x*0.75)
            simplex_noise = 1/scaling * get_simplex2(
                scale_x=x, scale_y=x,
                info_dict=info_dict,
                center=True,
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
            terrain_temp.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            }


class Holes(Module):
    create_folder = False

    ''' Test to make some rocks '''
    @debug_decorator
    def __call__(self, size_list=[0.5, 1, 2, 4],
                 fraction=0.8,
                 default=None,
                 terrain_temp=[],
                 **kwargs):
        size_list = default if default is not None else size_list

        from utils.noise import get_simplex2
        from utils.terrains import Terrain

        for i, x in enumerate(size_list):
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
            simplex_noise = simplex_noise * (x/8)

            terrain = Terrain.from_array(simplex_noise, **info_dict)
            terrain_temp.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            }