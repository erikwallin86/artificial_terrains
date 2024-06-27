from modules.data import Module, debug_decorator
import numpy as np
from utils.terrains import Terrain


class Split(Module):
    ''' 
    Split into pipes
    '''
    create_folder = False

    @debug_decorator
    def __call__(self, nx=2, ny=2, seed=None,
                 extent=None,
                 terrain_temp=None, terrain_heap=None,
                 default=None, call_number=None, call_total=None, **pipe):

        # n = default if default is not None else n
        if isinstance(default, list):
            [nx, ny] = default
        elif isinstance(default, (float, int)):
            nx, ny = defult, default

        # x and y indices of splits
        x = np.arange(nx)
        y = np.arange(ny)

        # All combinations of x and y indices
        from itertools import product
        combinations = list(product(x, y))

        # How extent splits in each dimension
        x_extent_split = np.linspace(extent[0], extent[1], nx + 1)
        y_extent_split = np.linspace(extent[2], extent[3], ny + 1)

        # Create empty pipes list
        pipes = []

        from utils.terrains import split_terrain_blocks, extent_to_size
        if terrain_temp is not None:
            terrain_temp_blocks = split_terrain_blocks(terrain_temp, nx, ny)
        if terrain_heap is not None:
            terrain_heap_blocks = split_terrain_blocks(terrain_heap, nx, ny)

        for i, split_index in enumerate(combinations):
            # Copy pipe
            new_pipe = pipe.copy()
            # and add split-index
            new_pipe['split_index'] = split_index

            # Get indices of combination
            a, b = split_index
            # Get new x and y extents
            extent_x = list(x_extent_split[a:a+2])
            extent_y = list(y_extent_split[b:b+2])
            # And combine to new extent
            sub_extent = extent_x + extent_y
            new_pipe['extent'] = sub_extent
            new_pipe['size'] = extent_to_size(sub_extent)
            if terrain_temp:
                array_list = list(terrain_temp_blocks[:, a, b])
                terrain_list = [Terrain.from_array(array, extent=sub_extent) for array in array_list]
                new_pipe['terrain_temp'] = terrain_list
            if terrain_heap:
                array_list = list(terrain_heap_blocks[:, a, b])
                terrain_list = [Terrain.from_array(array, extent=sub_extent) for array in array_list]
                new_pipe['terrain_heap'] = terrain_list

            if seed is not None:
                new_pipe['seed'] = seed
                # seed += 1
            pipes.append(new_pipe)

        if len(pipes) == 0:
            raise ValueError("Empty pipe")
        elif len(pipes) == 1:
            # Just return pipe
            return pipes[0]
        else:
            return pipes


class Join(Module):
    ''' 
    Join multiple pipes
    '''
    create_folder = False

    @debug_decorator
    def __call__(self, 
                 terrain_temp=None, terrain_heap=None,
                 extent=None, split_index=None,
                 default=None, call_number=None, call_total=None,
                 index_to_terrain_temp={},
                 index_to_terrain_heap={},
                 extent_list=[],
                 **pipe):

        # If only one call, do nothing
        if call_total == 1:
            return

        # Append terrain lists to dict
        index_to_terrain_temp[split_index] = terrain_temp
        index_to_terrain_heap[split_index] = terrain_heap
        extent_list.append(extent)

        # Early return on all but last call
        if call_number + 1 != call_total:
            return 'remove'

        # On last call, place collected terrains side by side using np.block
        from utils.terrains import merge_terrain_blocks
        if list(index_to_terrain_temp.values())[0]:
            terrain_temp = merge_terrain_blocks(index_to_terrain_temp)
            pipe['terrain_temp'] = terrain_temp

        if list(index_to_terrain_heap.values())[0]:
            terrain_heap = merge_terrain_blocks(index_to_terrain_heap)
            pipe['terrain_heap'] = terrain_heap

        # Merge the extent
        extent_start = extent_list[0]
        extent_end = extent_list[-1]
        merged_extent = [extent_start[0], extent_end[1], extent_start[2], extent_end[3]]
        pipe['extent'] = merged_extent

        pipe['split_index'] = None

        return pipe
