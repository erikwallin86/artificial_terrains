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
