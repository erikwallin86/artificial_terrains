import numpy as np


class Obstacles:
    # array_types = ['position', 'radius', 'height']
    array_types = ['position', 'width', 'height', 'yaw_deg', 'aspect',
                   'pitch_deg']
    limit_name_mapping = {'position': 'pos_lim', 'width': 'width_lim'}

    def __init__(self, Y=None, N=None, **kwargs):
        '''
        Obstacles

        Args:
          Y: terrain classification in [0, 5]
          position_distribution: normalized 2D np.array
        '''
        if Y is not None:
            # Set using Y and position_distribution
            self.set_from_Y(Y=Y, **kwargs)
        elif N is not None:
            # Set using N, pos_lim, rad_lim
            self.set_from_limits(N, **kwargs)

    def __len__(self):
        return self.position.shape[1]

    def __getitem__(self, key):
        cls = type(self)
        # If slice, apply it to the arrays (second axis),
        # else get single index, but make sure too keep dims
        if isinstance(key, (slice, np.ndarray)):
            dict_of_arrays = {
                'position': self.position[:, key],
                'width': self.width[:, key],
                'height': self.height[:, key],
                'yaw_deg': self.yaw_deg[:, key],
                'aspect': self.aspect[:, key],
                'pitch_deg': self.pitch_deg[:, key],
            }
            return cls.from_arrays(dict_of_arrays)
        else:
            dict_of_arrays = {
                'position': self.position[:, key, np.newaxis],
                'width': self.width[:, key, np.newaxis],
                'height': self.height[:, key, np.newaxis],
                'yaw_deg': self.yaw_deg[:, key, np.newaxis],
                'aspect': self.aspect[:, key, np.newaxis],
                'pitch_deg': self.pitch_deg[:, key, np.newaxis],
            }
            return cls.from_arrays(dict_of_arrays)

    def __setitem__(self, position, piece):
        # self.pieces[position] = piece
        raise NotImplementedError

    def __add__(self, other):
        # Construct new combined arrays
        dict_of_new_arrays = {
            'position': np.concatenate((self.position, other.position), axis=1),
            'width': np.concatenate((self.width, other.width), axis=1),
            'height': np.concatenate((self.height, other.height), axis=1),
            'yaw_deg': np.concatenate((self.yaw_deg, other.yaw_deg), axis=1),
            'aspect': np.concatenate((self.aspect, other.aspect), axis=1),
            'pitch_deg': np.concatenate((self.pitch_deg, other.pitch_deg), axis=1),
        }
        cls = type(self)

        return cls.from_arrays(dict_of_new_arrays)

    def generator(self):
        for i in range(len(self)):
            position = self.position[:, i].tolist()
            width = self.width[0, i]
            height = self.height[0, i]
            yaw_deg = self.yaw_deg[0, i]
            aspect = self.aspect[0, i]
            pitch_deg = self.pitch_deg[0, i]

            yield position, width, height, yaw_deg, aspect, pitch_deg

    def __iter__(self):
        return self.generator()

    def __str__(self):
        # Return a string representation of the object
        return self.__repr__()

    def __repr__(self):
        # Return a complete string representation of the object
        repr_str = f"Obstacles(N={len(self)}"

        for name in self.array_types:
            if name in self.limit_name_mapping:
                array = getattr(self, name)
                limit_name = self.limit_name_mapping[name]
                left = [float(f"{a:.5g}") for a in np.min(array, axis=1)]
                right = [float(f"{a:.5g}") for a in np.max(array, axis=1)]
                if len(left) == 1:
                    left = left[0]
                if len(right) == 1:
                    right = right[0]

                repr_str += f", {limit_name}=[{left}, {right}]"

        # Add final parenthesis
        repr_str += ")"

        return repr_str

    def set_from_dict_of_arrays(self, dict_of_arrays):
        for name, array in dict_of_arrays.items():
            setattr(self, name, array)

    def set_from_Y(self, Y=3, size=(50, 50), position_distribution=None):
        '''


        Args:
          size: used for uniform distribution, unless other specified
          position_distribution: optional positional distribution
        '''
        from utils.utils import generate_size_distribution
        from utils.utils import spawn_particles
        # Generate density mapping
        obstacle_density_mapping = generate_size_distribution(Y=Y)
        # Uniform position distribution
        if position_distribution is None:
            position_distribution = np.ones(size)/np.multiply(*size)

        # Spawn particles
        dict_of_arrays = spawn_particles(
            obstacle_density_mapping, position_distribution)
        self.set_from_dict_of_arrays(dict_of_arrays)

    def set_from_limits(self, N=10000, pos_lim=[[0, 0], [100, 100]], rad_lim=[0.1, 0.4]):
        '''
        Setup using uniform/loguniform random numbers
        '''
        # radius = np.random.uniform(rad_lim[0], rad_lim[1], size=(1, N))
        # from scipy.stats import loguniform
        # radius = loguniform.rvs(rad_lim[0], rad_lim[1], size=(1, N))
        exp = np.random.exponential(0.5, size=(1, N))
        exp = np.clip(exp, 0, 3)
        width = np.interp(exp, [0, 6], rad_lim)

        low = np.array(pos_lim)[0].reshape(2, 1)
        high = np.array(pos_lim)[1].reshape(2, 1)
        position = np.random.uniform(low, high, size=(2, N))

        yaw_deg = np.random.uniform(low=0, high=360, size=(1, N))
        pitch_deg = np.random.uniform(low=0, high=30, size=(1, N))
        aspect = np.random.uniform(low=0.5, high=1.5, size=(1, N))

        dict_of_arrays = {
            'position': position,
            'width': width,
            'height': width/2,
            'yaw_deg': yaw_deg,
            'pitch_deg': pitch_deg,
            'aspect': aspect,
        }
        self.set_from_dict_of_arrays(dict_of_arrays)

    @classmethod
    def from_arrays(cls, dict_of_arrays):
        obstacles_obj = cls()
        obstacles_obj.set_from_dict_of_arrays(dict_of_arrays)
        return obstacles_obj

    @classmethod
    def from_Y(cls, *args, **kwargs):
        ''' Return Obstacles given a terrain class '''
        obstacles_obj = cls()
        obstacles_obj.set_from_Y(*args, **kwargs)
        return obstacles_obj

    @classmethod
    def from_numpy(cls, filename):
        with open(filename, 'rb') as f:
            dict_of_arrays = np.load(f, allow_pickle=True)
            return cls.from_arrays(dict_of_arrays)

    @classmethod
    def from_limits(cls, *args, **kwargs):
        ''' Return Obstacles given a terrain class '''
        obstacles_obj = cls()
        obstacles_obj.set_from_limits(*args, **kwargs)

        return obstacles_obj

    def save_numpy(self, filename, append=False):
        dict_of_arrays = {
            'position': self.position,
            'width': self.width,
            'height': self.height,
            'yaw_deg': self.yaw_deg,
            'aspect': self.aspect,
            'pitch_deg': self.pitch_deg,
        }

        np.savez(filename, **dict_of_arrays)

    def save_yaml(self, filename):
        # Convert numpy arrays to lists so they can be serialized to YAML
        dict_of_lists = {
            'position': self.position.T.tolist(),  # transpose back to original shape
            'width': self.width.flatten().tolist(),
            'height': self.height.flatten().tolist(),
            'yaw_deg': self.yaw_deg.flatten().tolist(),
            'aspect': self.aspect.flatten().tolist(),
            'pitch_deg': self.pitch_deg.flatten().tolist(),
        }
        import yaml
        with open(filename, 'w') as f:
            yaml.dump(dict_of_lists, f, sort_keys=False)

    def save_pickle(self, filename, append=False):
        raise NotImplementedError

    @classmethod
    def from_pickle(cls, filename):
        raise NotImplementedError

    def translate(self, dx=0, dy=0, inplace=False):
        if inplace:
            self.position += np.array([[dx, dy]]).reshape(2, 1)
            return self
        else:
            new_obj = self.copy()
            new_obj.position += np.array([[dx, dy]]).reshape(2, 1)
            return new_obj

    def copy(self):
        new_obj = type(self)()
        new_obj.position = self.position.copy()
        new_obj.width = self.width.copy()
        new_obj.height = self.height.copy()
        new_obj.yaw_deg = self.yaw_deg.copy()
        new_obj.aspect = self.aspect.copy()
        new_obj.pitch_deg = self.pitch_deg.copy()

        return new_obj
