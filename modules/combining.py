from modules.data import Module, debug_decorator
import numpy as np
from utils.terrains import Terrain


class Combine(Module):
    ''' Combined terrains '''
    create_folder = False

    @debug_decorator
    def __call__(self, operation='Add', terrain_temp=None, terrain_heap=None,
                 default=None, last=None, **_):
        '''
        Combine terrains in terrain_temp (or terrain_heap) using 'operation'

        Args:
          operation (string): What operation to perform
          last (int): Use last N terrains. If None, then all are used

        '''
        # Initialize lists
        terrain_temp = [] if terrain_temp is None else terrain_temp
        terrain_heap = [] if terrain_heap is None else terrain_heap

        operations = ['Add', 'Max', 'Min', 'Prod']
        operation = default if default is not None else operation
        assert operation in operations, f'operation not in {operations}'

        if last is not None and last > 0:
            # When e.g. taking the last two, the slice is [-2:]
            # So here we make sure this is negative
            last = -last

        from utils.utils import get_terrains
        terrains = get_terrains(
            terrain_temp, terrain_heap, last=last, remove=True,
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
        terrain_heap.append(terrain)

        # Return updated heap/dict
        return {
            'terrain_temp': terrain_temp,
            'terrain_heap': terrain_heap,
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
    def __call__(self, terrain_temp=[], terrain_heap=[],
                 default=None, last=None, **_):
        from utils.utils import get_terrains
        terrain = get_terrains(terrain_temp, terrain_heap, last)
        terrain_heap.extend(terrain)

        return {
            'terrain_temp': terrain_temp,
            'terrain_heap': terrain_heap,
        }


class WeightedSum(Module):
    ''' Weighted sum of terrains '''
    create_folder = False

    @debug_decorator
    def __call__(self, weights=[5, 8, 0.1], terrain_temp=None,
                 terrain_heap=None, last=None,
                 default=None, **_):

        # Initialize lists
        terrain_temp = [] if terrain_temp is None else terrain_temp
        terrain_heap = [] if terrain_heap is None else terrain_heap

        weights = default if default is not None else weights
        weights = np.array(weights).reshape(-1, 1, 1)

        self.info(f"Use weights:{weights.squeeze()}")

        # Get terrains
        from utils.utils import get_terrains
        terrains = get_terrains(
            terrain_temp, terrain_heap, last=last, remove=True,
            print_fn=self.info)

        # Get arrays
        terrain_arrays = [terrain.array for terrain in terrains]

        array = np.sum(np.multiply(terrain_arrays, weights), axis=0)
        terrain = Terrain.from_array(
            array, size=terrains[0].size, extent=terrains[0].extent)

        terrain_heap.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            'terrain_heap': terrain_heap,
            }
