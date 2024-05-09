import argparse
import numpy as np
import os


class StoreTuplePair(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreTuplePair, self).__init__(
            option_strings,
            dest,
            nargs=nargs,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Store list of tuple-pairs
        tuple_pair_list = []
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            try:
                tuple_pair_list.append((key, eval(value)))
            except NameError:
                if value.startswith('[') and value.endswith(']'):
                    list_of_values = value[1:-1].split(',')
                    tuple_pair_list.append((key, list_of_values))
                else:
                    tuple_pair_list.append((key, value))
            except SyntaxError:
                if value == "":
                    tuple_pair_list.append((key, None))
                else:
                    if value.startswith('[') and value.endswith(']'):
                        list_of_values = value[1:-1].split(',')
                        tuple_pair_list.append((key, list_of_values))
                    else:
                        tuple_pair_list.append((key, value))
        setattr(namespace, self.dest, tuple_pair_list)


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings,
                                        dest,
                                        nargs=nargs,
                                        **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            arg_dict_local = self.split(arguments)
            arg_dict = {**arg_dict, **arg_dict_local}
        setattr(namespace, self.dest, arg_dict)

    def split(self, arguments):
        arg_dict = {}
        key = arguments.split(":")[0]
        value = ":".join(arguments.split(":")[1:])
        # Evaluate the string as python code
        try:
            if ':' in value:
                arg_dict_lower = self.split(value)
                arg_dict[key] = arg_dict_lower
            else:
                arg_dict[key] = eval(value)
        except NameError:
            arg_dict[key] = value
        except SyntaxError:
            return {key: value}

        return arg_dict


def calc_y(array, ha=1.0):
    shape = array.shape[:-1]
    # Reshape to two dimensions
    array = array.reshape(-1, 4)

    # Rescale classes given area
    # rikligt = np.array([4000, 40000])*ha
    måttligt = np.array([400, 4000])*ha
    sparsamt = np.array([40, 400])*ha
    enstaka = np.array([4, 40])*ha

    # if H[20] < sparsamt[1]:
    a = array[:, 0] < sparsamt[1]
    # if H[40] + H[60] + H[80] < enstaka[1]:
    b = array[:, 1] + array[:, 2] + array[:, 3] < enstaka[1]
    # elif H[40] < sparsamt[1] and H[60] + H[80] < enstaka[1]:
    c1 = array[:, 1] < sparsamt[1]
    c2 = array[:, 2] + array[:, 3] < enstaka[1]
    c = np.logical_and(c1, c2)
    c = np.logical_and(c, np.logical_not(b))
    # elif H[40] < måttligt[1] and H[60] < sparsamt[1]:  # d
    d1 = array[:, 1] < måttligt[1]
    d2 = array[:, 2] < sparsamt[1]
    d = np.logical_and(d1, d2)
    d = np.logical_and(d, np.logical_not(np.logical_or(b, c)))
    # if H[80] < enstaka[1]:
    e = array[:, 3] < enstaka[1]
    # elif H[80] < sparsamt[1]:
    f1 = enstaka[1] < array[:, 3]
    f2 = array[:, 3] < sparsamt[1]
    f = np.logical_and(f1, f2)
    # not e or f
    g = np.logical_not(np.logical_or(e, f))
    # not b, d, or d
    h = np.logical_not(b | c | d)
    # elif sparsamt[1] < H[20] < måttligt[1]: # i
    i1 = sparsamt[1] < array[:, 0]
    i2 = array[:, 0] < måttligt[1]
    i = np.logical_and(i1, i2)
    # if H[40] + H[60] + H[80] < enstaka[0]: # j
    j = array[:, 1] + array[:, 2] + array[:, 3] < enstaka[0]
    # not a or i
    k = np.logical_not(np.logical_or(a, i))

    y1_1 = np.logical_and(a, b)
    y2_1 = np.logical_and(a, c)
    y3_1 = a & d & e
    y4_1 = a & d & f
    y5_1 = a & d & g
    y5_2 = np.logical_and(a, h)

    y1_2 = np.logical_and(i, j)
    y2_2 = np.logical_and(i, c)
    y3_2 = i & d & e
    y4_2 = i & d & f
    y5_3 = i & d & g
    y5_4 = np.logical_and(i, h)

    y2_3 = np.logical_and(k, j)
    y3_3 = k & d & e
    y4_3 = k & d & f
    y5_5 = k & d & g
    y5_6 = np.logical_and(k, h)

    # Combine using logical_or
    y1 = y1_1 | y1_2
    y2 = y2_1 | y2_2 | y2_3
    y3 = y3_1 | y3_2 | y3_3
    y4 = y4_1 | y4_2 | y4_3
    y5 = y5_1 | y5_2 | y5_3 | y5_4 | y5_5 | y5_6

    # Construct y integer
    y = y1*1 + y2*2 + y3*3 + y4*4 + y5*5

    return y.reshape(shape)


def calc_ytstruktur(H_to_number_mapping, ha=1.0):
    # Take input as either list of dict
    is_array = isinstance(H_to_number_mapping, np.ndarray)
    is_list = isinstance(H_to_number_mapping, list)
    if is_list or is_array:
        H = {
            20: H_to_number_mapping[0],
            40: H_to_number_mapping[1],
            60: H_to_number_mapping[2],
            80: H_to_number_mapping[3],
        }
    else:
        H = H_to_number_mapping

    Y = 0
    rikligt = np.array([4000, 40000])*ha
    måttligt = np.array([400, 4000])*ha
    sparsamt = np.array([40, 400])*ha
    enstaka = np.array([4, 40])*ha

    # H20 sparsamt
    if H[20] < sparsamt[1]: # a
        if H[40] + H[60] + H[80] < enstaka[1]: # b
            Y = 1
        elif H[40] < sparsamt[1] and H[60] + H[80] < enstaka[1]: # c
            Y = 2
        elif H[40] < måttligt[1] and H[60] < sparsamt[1]:  # d
            if H[80] < enstaka[1]: # e
                Y = 3
            elif H[80] < sparsamt[1]: # f
                Y = 4
            else: # g
                Y = 5
        else: # h
            Y = 5
    # H20 måttligt
    elif sparsamt[1] < H[20] < måttligt[1]: # i
        # H40-H80 finns inga
        if H[40] + H[60] + H[80] < enstaka[0]: # j
            Y = 1
        # H40 sparsamt, H60-H80 enstaka
        elif H[40] < sparsamt[1] and H[60] + H[80] < enstaka[1]: # c
            Y = 2
        # H40 måttligt
        elif H[40] < måttligt[1] and H[60] < sparsamt[1]: # d
            if H[80] < enstaka[1]: # e
                Y = 3
            elif H[80] < sparsamt[1]: # f
                Y = 4
            else:
                Y = 5 # g
        else: # h
            Y = 5
    # H20 Rikligt
    else: # k
        if H[40] + H[60] + H[80] < enstaka[0]: # j
            Y = 2
        # H40 måttligt
        elif H[40] < måttligt[1] and H[60] < sparsamt[1]: # d
            if H[80] < enstaka[1]: # e
                Y = 3
            elif H[80] < sparsamt[1]: # f
                Y = 4
            else: # g
                Y = 5
        else: # h
            Y = 5

    return Y


def generate_size_distribution(Y=3, random=False):
    '''
    Generate a size distribution
    '''
    rikligt = [4000, 40000]
    måttligt = [400, 4000]
    sparsamt = [40, 400]
    enstaka = [4, 40]

    if Y == 3 and not random:
        H_to_number_mapping = {
            20: 8000,
            40: 2000,
            60: 200,
            80: 20,
        }
    elif Y == 3 and random:
        H_to_number_mapping = {
            20: np.random.randint(*rikligt),
            40: np.random.randint(*måttligt),
            60: np.random.randint(*sparsamt),
            80: np.random.randint(*enstaka),
        }

    return H_to_number_mapping


class Grid():
    def __init__(self, shape=None, size=None, cell_size=None):
        if size is None and cell_size is None and shape is None:
            # Default
            size = (100, 100)
            shape = (32, 32)
        elif size is None and cell_size is None:
            size = (100, 100)
        elif size is None and shape is None:
            shape = (32, 32)
            size = np.multiply(shape, cell_size)
        elif cell_size is None and shape is None:
            shape = (32, 32)
        elif size is None:
            size = np.multiply(shape, cell_size)
        elif shape is None:
            shape = np.divide(size, cell_size).astype(np.uint64).tolist()
        elif cell_size is None:
            pass
        else:
            raise Exception("Grid: Can't specify size, cell_size, and shape")

        # Define coordinates, representing edges of bins
        x_edges = np.linspace(0, size[0], shape[0]+1)
        y_edges = np.linspace(0, size[1], shape[1]+1)

        x_size = (x_edges[1]-x_edges[0])
        y_size = (y_edges[1]-y_edges[0])

        x_centers = x_edges[:-1] + x_size/2
        y_centers = y_edges[:-1] + y_size/2

        self.shape = shape
        self.size = size

        self.x_centers = x_centers
        self.y_centers = y_centers
        self.x_size = x_size
        self.y_size = y_size
        self.x_edges = x_edges
        self.y_edges = y_edges

    def get_points(self):
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                yield np.array((self.x_centers[x], self.y_centers[y]))

    @property
    def area(self):
        return self.x_size * self.y_size

    @property
    def area_ha(self):
        return self.area/1e+4

    @property
    def extent(self):
        x_edges = self.x_edges
        y_edges = self.y_edges
        return [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    def __str__(self):
        return f"Grid: shape={self.shape}, area={self.area_ha:.2f} ha"

    def __repr__(self):
        return f"Grid(size={self.size}, shape={self.shape})"


def count_obstacles(
        grid, obstacles,
        only_center=False,
        return_particle_count=False
):
    '''
    Count number of obstacles, in every grid cell

    Return:
      counts array, of shape (*grid.shape, 4)
        where the last dim is a 4-list e.g. [100, 34, 10, 1]
        (i.e. the number of H20, H40, H60 H80)
    '''
    position = obstacles.position
    radius = obstacles.radius

    list_of_counts = []
    accumulated_particle_count = np.zeros_like(radius)

    for x, y in grid.get_points():
        # Pick obstacles within the grid box
        if only_center:
            x_pick = (np.absolute(x - position[0, :]) < grid.x_size/2)
            y_pick = (np.absolute(y - position[1, :]) < grid.y_size/2)
        else:
            x_pick = (np.absolute(x - position[0, :]) - radius[0, :] < grid.x_size/2)
            y_pick = (np.absolute(y - position[1, :]) - radius[0, :] < grid.y_size/2)
        pick = np.logical_and(x_pick, y_pick).squeeze()
        accumulated_particle_count += pick
        diameter = 2*radius[:, pick].flatten()
        diameter = np.clip(diameter, 0, 0.9)

        # Divide into boxes
        bins = [0.10, 0.30, 0.50, 0.70, 0.90]
        counts, bin_edges = np.histogram(diameter, bins=bins)
        list_of_counts.append(counts)

    count_array = np.array(list_of_counts).reshape(*grid.shape, 4)

    if return_particle_count:
        return count_array, accumulated_particle_count
    else:
        return count_array


def check_double_counts(grid, obstacles):
    '''
    Construct array of how many cells an obstacle is in
    '''
    radius = obstacles.radius[0, :]
    pos_x = obstacles.position[0, :]
    pos_y = obstacles.position[1, :]

    # Get obstacles which are on one or several edges
    x_pick = np.mod(pos_x+radius, grid.x_size) < 2*radius
    y_pick = np.mod(pos_y+radius, grid.y_size) < 2*radius

    pick_corners = np.logical_and(x_pick, y_pick)
    pick_edges = np.logical_xor(x_pick, y_pick)
    pick_corners_and_edges = np.logical_or(x_pick, y_pick)
    pick_neither = np.logical_not(pick_corners_and_edges)

    return (pick_corners * 4 + pick_edges * 2 + pick_neither).reshape(1, -1)


def spawn_particles(obstacle_density_mapping, probability):
    '''
    Generate particles arrays from a 2d probability (for location)
    and a mapping from obstacle size (H) to number of obstacles

    Returns:
      dict of arrays
      particle_pos has size [2, N]
      particle_radius has size [1, N]
      {
    '''
    from utils.noise import generate_points_from_2D_prob
    # Create lists of particles
    particle_pos = []
    particle_width = []
    for H, density in obstacle_density_mapping.items():
        # Generate random integers
        size = probability.shape
        number = round(density * size[0] * size[1]/100**2)
        indices = generate_points_from_2D_prob(probability, N=number)
        # Add uniform noise to turn into float
        rng = np.random.default_rng()
        uniform_noise = rng.uniform(size=indices.shape)
        position = indices + uniform_noise

        particle_pos.append(position)
        particle_width.append(H/100.0 * np.ones((1, indices.shape[1])))

    # Turn into arrays
    particle_pos = np.concatenate(particle_pos, axis=1)
    particle_width = np.concatenate(particle_width, axis=1)

    # Generate other parameters
    N = particle_pos.shape[1]
    yaw_deg = np.random.uniform(low=0, high=360, size=(1, N))
    pitch_deg = np.random.uniform(low=0, high=30, size=(1, N))
    aspect = np.random.uniform(low=0.5, high=1.5, size=(1, N))

    return {
        'position': particle_pos,
        'width': particle_width,
        'height': particle_width/2,
        'yaw_deg': yaw_deg,
        'aspect': aspect,
        'pitch_deg': pitch_deg,
        }


def get_list_of_files(dirs: list, pattern: str):
    '''
    Return a (sorted) list of all files (with directories) in the
    directories dirs
    '''
    out = []
    for d in dirs:
        files = sorted(os.listdir(d))
        j = 0
        for f in files:
            # print(f"f:{f}")
            if pattern in f:
                j += 1
                out.append(f)

    return sorted(out)


def add_id(filename, file_id):
    '''
    Add file_id (e.g. '_00005' or '') to filename
    '''
    root, extension = os.path.splitext(filename)
    filename = root + file_id + extension

    return filename
