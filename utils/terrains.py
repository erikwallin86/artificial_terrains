import numpy as np


class Terrain():
    def __init__(self):
        self.i = 0

    @classmethod
    def from_numpy(cls, filename):
        ''' Load numpy terrain '''
        terrain_obj = cls()
        terrain_obj.array, info_dict = load_hf(filename)
        terrain_obj.size = info_dict['size']
        terrain_obj.extent = info_dict['extent']
        terrain_obj.kwargs = {}  # TODO: Added as a fix. But not in use I think

        return terrain_obj

    @classmethod
    def from_geotiff(cls, filename):
        ''' Load geotiff terrain '''
        raise NotImplementedError

    @classmethod
    def from_las(cls, filename):
        ''' Generate from point cloud?? '''
        raise NotImplementedError

    @classmethod
    def from_dem(cls, filename):
        ''' Load 'digital elevation model' '''
        raise NotImplementedError

    @classmethod
    def from_array(cls, array, size=None, extent=None, **kwargs):
        ''' Use array '''
        terrain_obj = cls()
        terrain_obj.array = array.astype(np.float64)

        if extent is None:
            extent = [-size[0]/2, size[0]/2, -size[1]/2, size[1]/2]
        elif size is None:
            size = [extent[1]-extent[0], extent[3]-extent[2]]
        terrain_obj.extent = extent
        terrain_obj.size = size

        # TODO: I do nothing with the kwargs,
        # but they might containt information I want to save with the array

        # TODO: La till det här, men det lirar inte exakt med hur save funkar
        # Ska man ge kwargs save, eller här, eller båda?
        terrain_obj.kwargs = kwargs

        return terrain_obj

    def save(self, filename, **kwargs):
        '''
        Save
        '''
        if not filename.endswith('.npz'):
            filename += '.npz'

        size_array = np.array(self.size).astype(np.float64)
        try:
            extent = self.extent
        except AttributeError:
            extent = [-self.size[0]/2, self.size[0]/2,
                      -self.size[1]/2, self.size[1]/2]

        with open(filename, 'wb') as f:
            np.savez(f, size=size_array, extent=extent, heights=self.array,
                     **self.kwargs, **kwargs)

    @property
    def position(self):
        '''Return the (x, y) center of the terrain from its extent'''
        x_min, x_max, y_min, y_max = self.extent
        return [(x_min + x_max) / 2, (y_min + y_max) / 2]

    @property
    def resolution(self):
        """
        Return the resolution (dx, dy) of the terrain.

        dx = (x_max - x_min) / (width - 1)
        dy = (y_max - y_min) / (height - 1)
        """
        x_min, x_max, y_min, y_max = self.extent
        height, width = self.array.shape
        dx = (x_max - x_min) / (width - 0)
        dy = (y_max - y_min) / (height - 0)

        return (dx, dy)

    # TODO: Use this instead of setting self.size etc.
    # @property
    # def size(self):
    #     '''Return (width, height) of the terrain from extent'''
    #     x_min, x_max, y_min, y_max = self.extent
    #     return [x_max - x_min, y_max - y_min]

    def __repr__(self):
        return f'Terrain(array={self.array.shape}, size={self.size}, extent={self.extent})'

    def __array__(self):
        # https://stackoverflow.com/questions/43493256/python-how-to-implement-a-custom-class-compatible-with-numpy-functions
        return self.array

    def __array_wrap__(self, obj, context=None):
        print("############### WRAP ############")
        # Make sure to return terrain object
        # Works with np 'ufunc's
        # obj is the numpy array.
        # np.maximum, but not np.maximum.reduce
        # np.add, but not np.sum1
        #

        return Terrain.from_array(obj, self.size, self.extent)


def load_hf(filename):
    '''
    Load height field from numpy binary file
    Returns an array of heights and an array specifying the size
    '''
    with open(filename, 'rb') as f:
        try:
            loaded_data = np.load(f, allow_pickle=True)
        except AttributeError:
            pass
        info_dict = {}
        for key in loaded_data.files:
            if key == 'heights':
                array = loaded_data['heights']
            else:
                info_dict[key] = loaded_data[key]

        return array, info_dict


def extent_to_size(extent):
    return [extent[1]-extent[0], extent[3]-extent[2]]


def merge_terrain_blocks(index_to_terrain_list):
    """
    Merges terrain arrays from a dictionary of indexed terrain lists into larger terrain arrays.

    Args:
        index_to_terrain_list (dict): A dictionary where keys are tuple indices (i, j) and 
                                      values are lists of Terrain objects at those indices.

    Returns:
        list: A list of merged Terrain objects, each representing a layer of the input terrain arrays.
    """
    # Determine the number of terrain layers and grid dimensions
    num_layers = len(next(iter(index_to_terrain_list.values())))
    last_index = list(index_to_terrain_list.keys())[-1]
    grid_rows = last_index[0] + 1
    grid_cols = last_index[1] + 1

    # Determine new extent of the merged terrain
    extent_start = list(index_to_terrain_list.values())[0][0].extent
    extent_end = list(index_to_terrain_list.values())[-1][0].extent
    merged_extent = [extent_start[0], extent_end[1], extent_start[2], extent_end[3]]

    # Prepare a 3D list to store terrain arrays
    terrain_layers = [[[None for _ in range(grid_cols)] for _ in range(grid_rows)] for _ in range(num_layers)]

    # Populate the 3D list with terrain arrays
    for (row, col), terrain_list in index_to_terrain_list.items():
        for layer_idx, terrain in enumerate(terrain_list):
            terrain_layers[layer_idx][row][col] = terrain.array

    # Convert the arrays to the desired format
    merged_terrain_list = []
    for layer_arrays in terrain_layers:
        # Merge blocks into a single array
        merged_array = np.block(layer_arrays)
        # Create a Terrain object
        merged_terrain = Terrain.from_array(merged_array, extent=merged_extent)
        merged_terrain_list.append(merged_terrain)

    return merged_terrain_list


def split_terrain_blocks(terrain_list, num_splits_row, num_splits_col):
    """
    Splits a given array into sub-arrays over two dimensions.

    Args:
        array (np.ndarray): The input array to be split into blocks.
        num_splits_row (int): The number of splits along the row dimension.
        num_splits_col (int): The number of splits along the column dimension.

    Returns:
        list: A list of lists, where each inner list contains sub-arrays split along the column dimension.
    """
    list_of_blocks = []

    for terrain in terrain_list:
        array = terrain.array
        # Split the array into sub-arrays along the row dimension
        row_splits = np.array_split(array, num_splits_row, axis=0)
        # Split each of the row sub-arrays into sub-arrays along the column dimension
        blocks = [np.array_split(row_split, num_splits_col, axis=1) for row_split in row_splits]
        list_of_blocks.append(blocks)

    blocks = np.array(list_of_blocks)

    return blocks


def assign_grid_indices(terrains):
    centers = [t.position for t in terrains]
    x_coords = sorted(set(x for x, _ in centers))
    y_coords = sorted(set(y for _, y in centers), reverse=False)

    x_map = {x: i for i, x in enumerate(x_coords)}
    y_map = {y: j for j, y in enumerate(y_coords)}

    grid_indices = [(x_map[x], y_map[y]) for x, y in centers]

    return grid_indices


def distances_to_extent(points, extent):
    """
    Calculate the shortest distance from each point to a rectangular extent.

    Parameters:
    - points: (N, 2) array of (x, y) coordinates
    - xmin, xmax: float, horizontal bounds of the extent
    - ymin, ymax: float, vertical bounds of the extent

    Returns:
    - distances: (N,) array of Euclidean distances to the rectangle
    """
    xmin, xmax, ymin, ymax = extent

    x = points[:, 0]
    y = points[:, 1]

    dx = np.where(x < xmin, xmin - x, np.where(x > xmax, x - xmax, 0))
    dy = np.where(y < ymin, ymin - y, np.where(y > ymax, y - ymax, 0))

    return np.sqrt(dx**2 + dy**2)


def get_surface_normal(hf_array, size):
    '''
    Get surface normal from array

    Args:
      hf_array:
      size: physical size (float, float)

    Return: array with surface normals
    '''
    # https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
    spacing = np.divide(size, hf_array.shape)

    zy, zx = np.gradient(hf_array, *spacing)
    normal = np.dstack((-zx, -zy, np.ones_like(hf_array)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    return normal


def calculate_surface_area(heightmap: np.ndarray, resolution: tuple[float, float]) -> float:
    """
    Estimate the surface area of a terrain heightmap.

    Parameters:
    - heightmap: 2D numpy array of shape (H, W), representing elevation values.
    - resolution: tuple (dy, dx), the spacing between pixels in y and x directions (in meters).

    Returns:
    - surface_area: float, estimated 3D surface area in square meters.
    """
    dy, dx = resolution
    dz_dy, dz_dx = np.gradient(heightmap, dy, dx)

    # Local area element from the surface normal magnitude approximation
    area_elements = np.sqrt(1 + dz_dx**2 + dz_dy**2) * dx * dy

    return np.sum(area_elements)
