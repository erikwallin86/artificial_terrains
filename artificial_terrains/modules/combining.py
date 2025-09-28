from modules.data import Module, debug_decorator
import numpy as np
from utils.terrains import Terrain


class Combine(Module):
    ''' Combined terrains '''
    create_folder = False

    @debug_decorator
    def __call__(self, operation='Add', terrain_temp=None, terrain_prim=None,
                 default=None, last=None, **_):
        '''
        Combine terrains in terrain_temp (or terrain_prim) using 'operation'

        Args:
          operation (string): What operation to perform
          last (int): Use last N terrains. If None, then all are used

        '''
        # Initialize lists
        terrain_temp = [] if terrain_temp is None else terrain_temp
        terrain_prim = [] if terrain_prim is None else terrain_prim

        operations = ['Add', 'Max', 'Min', 'Prod']
        operation = default if default is not None else operation
        assert operation in operations, f'operation not in {operations}'

        if last is not None and last > 0:
            # When e.g. taking the last two, the slice is [-2:]
            # So here we make sure this is negative
            last = -last

        from utils.utils import get_terrains
        terrains = get_terrains(
            terrain_temp, terrain_prim, last=last, remove=True,
            print_fn=self.info)

        # Turn into arrays. Workaround to work in Windows/Ubuntu
        terrain_arrays = [terrain.array for terrain in terrains]

        # Apply any of the operations
        if operation == 'Add':
            array = np.sum(terrain_arrays, axis=0)
            terrain = Terrain.from_array(
                array, size=terrains[0].size, extent=terrains[0].extent)
        elif operation == 'Max':
            array = np.maximum.reduce(terrain_arrays)
            terrain = Terrain.from_array(
                array, size=terrains[0].size, extent=terrains[0].extent)
        elif operation == 'Min':
            array = np.minimum.reduce(terrain_arrays)
            terrain = Terrain.from_array(
                array, size=terrains[0].size, extent=terrains[0].extent)
        elif operation == 'Prod':
            array = np.prod(terrain_arrays, axis=0)
            terrain = Terrain.from_array(
                array, size=terrains[0].size, extent=terrains[0].extent)
        else:
            raise AttributeError

        # Add terrain to heap
        terrain_prim.append(terrain)

        # Return updated heap/dict
        return {
            'terrain_temp': terrain_temp,
            'terrain_prim': terrain_prim,
        }


class CombineLast(Combine):
    ''' Combine, with only the last two terrains,

    'terrain' and the latest from the 'heap'
    '''
    def __call__(self, *args, last=None, **kwargs):
        return super(CombineLast, self).__call__(*args, last=2, **kwargs)


class ToPrimary(Combine):
    ''' Move terrains to primary '''
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=[], terrain_prim=[],
                 default=None, last=None, **_):
        from utils.utils import get_terrains
        terrain = get_terrains(terrain_temp, terrain_prim, last)
        terrain_prim.extend(terrain)

        return {
            'terrain_temp': terrain_temp,
            'terrain_prim': terrain_prim,
        }


class WeightedSum(Module):
    ''' Weighted sum of terrains '''
    create_folder = False

    @debug_decorator
    def __call__(self, weights=[5, 8, 0.1], terrain_temp=None,
                 terrain_prim=None, last=None,
                 overwrite=False,
                 default=None, **_):

        # Initialize lists
        terrain_temp = [] if terrain_temp is None else terrain_temp
        terrain_prim = [] if terrain_prim is None else terrain_prim

        weights = default if default is not None else weights
        weights = np.array(weights).reshape(-1, 1, 1)

        self.info(f"Use weights:{weights.squeeze()}")

        # Get terrains
        from utils.utils import get_terrains
        terrains = get_terrains(
            terrain_temp, terrain_prim, last=last, remove=True,
            print_fn=self.info)

        # Get arrays
        terrain_arrays = [terrain.array for terrain in terrains]

        array = np.sum(np.multiply(terrain_arrays, weights), axis=0)
        terrain = Terrain.from_array(
            array, size=terrains[0].size, extent=terrains[0].extent)

        terrain_prim.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            'terrain_prim': terrain_prim,
            }


class Compose(Module):
    ''' Combined two (and only) terrains using a mask (factor) + operation '''
    create_folder = False

    @debug_decorator
    def __call__(self, operation='Over',
                 terrain_temp=None, terrain_prim=None,
                 factor=None, default=None, last=None, **_):
        '''
        Combine terrains two terrains using some operation and mask (factor)

        Args:
          operation (string): What operation to perform
        '''
        # Initialize lists
        terrain_temp = [] if terrain_temp is None else terrain_temp
        terrain_prim = [] if terrain_prim is None else terrain_prim

        operations = ['Over', 'Under']
        operation = default if default is not None else operation
        assert operation in operations, f'{operation} not in {operations}'

        from utils.utils import get_terrains
        terrains = get_terrains(
            terrain_temp, terrain_prim, last=2, remove=True,
            print_fn=self.info)

        # Turn into arrays. Workaround to work in Windows/Ubuntu
        array_0 = terrains[0].array
        array_1 = terrains[1].array

        # clip factor to [0, 1]
        factor = np.clip(factor, 0, 1)

        if operation == 'Over':
            array = array_1*factor + array_0*(1-factor)
        if operation == 'Under':
            array = array_0*factor + array_1*(1-factor)

        terrain = Terrain.from_array(
            array, size=terrains[0].size, extent=terrains[0].extent)

        # Add terrain to heap
        terrain_prim.append(terrain)

        # Return updated primary/temporary
        return {
            'terrain_temp': terrain_temp,
            'terrain_prim': terrain_prim,
        }


class Stack(Module):
    ''' Place terrains side by side '''
    create_folder = False

    @debug_decorator
    def __call__(self, terrain_temp=None, terrain_prim=None,
                 default=None, last=None, **_):
        '''
        Operates on the 'most recent' list.
        '''
        # Initialize lists
        terrain_temp = [] if terrain_temp is None else terrain_temp
        terrain_prim = [] if terrain_prim is None else terrain_prim

        if last is not None and last > 0:
            # When e.g. taking the last two, the slice is [-2:]
            # So here we make sure this is negative
            last = -last

        from utils.utils import get_terrains
        terrains = get_terrains(
            terrain_temp, terrain_prim, last=last, remove=True,
            print_fn=self.info)

        from utils.terrains import assign_grid_indices
        from utils.terrains import merge_terrain_blocks

        # Construct grid-indices from terrain extents
        grid_indices = assign_grid_indices(terrains)
        index_to_terrain = {}
        for grid_index, terrain in zip(grid_indices, terrains):
            index_to_terrain[grid_index] = [terrain]

        # Add the merged terrain to the primary list
        merged_terrain_list = merge_terrain_blocks(index_to_terrain)
        terrain_prim.extend(merged_terrain_list)

        # We need to return these in case we have emptited some list
        # (made it a new list?), as then the existing list is not 'updated'
        return {
            'terrain_temp': terrain_temp,
            'terrain_prim': terrain_prim,
        }
