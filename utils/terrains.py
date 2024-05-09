import numpy as np
from utils.noise import get_simplex


class Terrain():
    def __init__(self):
        self.i = 0
        pass
        # simplex_noise = get_simplex(Nx=200, Ny=200)
        # TODO: Use Grid class?

    # @classmethod
    # def from_noise(cls, dict_of_arrays):
    #     obstacles_obj = cls()
    #     obstacles_obj.set_from_dict_of_arrays(dict_of_arrays)
    #     return obstacles_obj

    @classmethod
    def from_noise(cls, ):
        ''' Create terrain noise '''
        # TODO: Apply noise to construct terrain
        raise NotImplementedError

    @classmethod
    def from_function(cls, function):
        ''' Create terrain from function '''
        # TODO: Loop all points and call function for height
        raise NotImplementedError

    @classmethod
    def from_numpy(cls, filename):
        ''' Load numpy terrain '''
        terrain_obj = cls()
        # terrain_obj.array, terrain_obj.size = load_hf(filename)
        terrain_obj.array, info_dict = load_hf2(filename)
        terrain_obj.size = info_dict['size']
        terrain_obj.extent = info_dict['extent']

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
    def from_array(cls, array, size=(50.0, 50.0), extent=None, **kwargs):
        ''' Use array '''
        terrain_obj = cls()
        terrain_obj.array = array.astype(np.float64)
        terrain_obj.size = size

        if extent is None:
            extent = [-size[0]/2, size[0]/2, -size[1]/2, size[1]/2]
        terrain_obj.extent = extent

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

    def __repr__(self):
        return f'Terrain(array={self.array.shape}, size={self.size}, extent={self.extent})'

    # def __mul__(self, other):
    #     # Multiply array and return new object
    #     if isinstance(other, Terrain):
    #         multiplied_array = np.multiply(self.array, other.array)
    #     else:
    #         multiplied_array = self.array * other
    # 
    #     return Terrain.from_array(multiplied_array, self.size, self.extent)
    # 
    # def __add__(self, other):
    #     print(f"other:{other}")
    #     print(f"other.size:{other.size}")
    #     # Only add equal sized terrains
    #     # assert self.size == other.size
    #     # Add arrays and return new object
    #     added_array = self.array + other.array
    #     return Terrain.from_array(added_array, self.size, self.extent)

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

    # def __array_ufunc__(self, *args, **kwargs):
    #     print(args)
    #     print(kwargs)

    # def __ge__(self, other):
    #     if isinstance(other, Terrain):
    #         combined_array = self.array >= other.array
    #     elif isinstance(other, (np.ndarray, np.generic)):
    #         combined_array = self.array >= other
    # 
    #     terrain = Terrain.from_array(combined_array, size=self.size, extent=self.extent)
    #     return terrain
    #     # return combined_array
    #     # combined_array = np.maximum(self.array, other.array)
    #     # combined_array = np.greater_equal(self.array, other.array)
    # 
    #     from utils.plots import plot_terrain
    #     terrain = Terrain.from_array(combined_array, size=self.size, extent=self.extent)
    #     fig, ax = plot_terrain(terrain)
    # 
    #     import uuid
    #     h = uuid.uuid4().hex
    #     filename = f'{self.i:05d}.png'
    #     folder = 'runs/data_014/ge'
    #     import os
    #     if not os.path.isdir(folder):
    #         os.makedirs(folder)
    #     filename = os.path.join(folder, filename)
    #     self.i += 1
    #     print(f"filename:{filename}")
    #     fig.savefig(filename)
    # 
    #     return terrain
    # 
    #     if isinstance(other, Terrain):
    #         # return np.all(self.array >= other.array)
    #         return np.maximum(self.array, other.array)
    #     else:
    #         raise TypeError("Comparison between Terrain and non-Terrain")


def load_hf(filename):
    '''
    Load height field from numpy binary file
    Returns an array of heights and an array specifying the size
    '''
    with open(filename, 'rb') as f:
        try:
            read = np.load(f, allow_pickle=True)
        except AttributeError:
            pass
        size = read['size']
        heights = read['heights']

        return heights, size


def load_hf2(filename):
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
