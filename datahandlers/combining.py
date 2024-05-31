from datahandlers.data import DataHandler, debug_decorator
import numpy as np
from utils.terrains import Terrain


class Combine(DataHandler):
    ''' Combined basic terrains within a terrain_dict '''
    create_folder = False

    @debug_decorator
    def __call__(self, operation='Add', terrain_dict={}, terrain_heap=[],
                 default=None, last=None, **_):
        '''
        Combine terrains in terrain_dict (or terrain_heap) using 'operation'

        Args:
          operation (string): What operation to perform
          last (int): Use last N terrains. If None, then all are used

        '''
        operations = ['Add', 'Max', 'Min', 'Prod']
        operation = default if default is not None else operation
        assert operation in operations, f'operation not in {operations}'

        if last is not None and last > 0:
            # When e.g. taking the last two, the slice is [-2:]
            # So here we make sure this is negative
            last = -last

        if len(terrain_dict) > 0:
            # Work with the terrain_dict. Trick to slice dict using last
            sliced_terrain_dict = dict(list(terrain_dict.items())[last:])
            terrains = list(sliced_terrain_dict.values())

            # Remove any used terrains from the terrain_dict
            terrain_dict = {k: v for k, v in terrain_dict.items()
                            if k not in sliced_terrain_dict}

            self.info(f"{operation} {len(terrains)} terrains from 'dict'")
        elif len(terrain_heap) > 0:
            # Get terrains from heap (if last=None then all are used)
            terrains = terrain_heap[last:]
            self.info(f"{operation} {len(terrains)} terrains from 'heap'")

            # Remove any used terrains from terrain_heap.
            terrain_heap = [terrain for terrain in terrain_heap
                            if terrain not in terrains]
            if len(terrain_heap) > 0:
                self.info(f"{len(terrain_heap)} terrains remaining")

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
            'terrain_dict': terrain_dict,
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
    def __call__(self, terrain_dict={}, terrain_heap=[],
                 default=None, last=None, **_):
        from utils.utils import get_terrains
        terrain = get_terrains(terrain_dict, terrain_heap, last)
        terrain_heap.extend(terrain)

        return {
            'terrain_dict': terrain_dict,
            'terrain_heap': terrain_heap,
        }


class WeightedSum(DataHandler):
    ''' Weighted sum of basic terrains within a terrain_dict '''
    create_folder = False

    @debug_decorator
    def __call__(self, weights=[5, 8, 0.1], terrain_dict={},
                 terrain_heap=[],
                 default=None, **_):

        weights = default if default is not None else weights
        weights = np.array(weights).reshape(-1, 1, 1)

        self.info(f"Use weights:{weights.squeeze()}")

        terrains = list(terrain_dict.values())
        terrain_arrays = [terrain.array for terrain in terrains]

        array = np.sum(np.multiply(terrain_arrays, weights), axis=0)
        terrain = Terrain.from_array(
            array, size=terrains[0].size, extent=terrains[0].extent)

        terrain_heap.append(terrain)

        return {
            'terrain_dict': {},
            'terrain_heap': terrain_heap,
            }
