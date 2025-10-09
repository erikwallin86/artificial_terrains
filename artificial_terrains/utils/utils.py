import argparse
import numpy as np
import os
from utils.terrains import Terrain
from itertools import cycle, islice


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


def calc_ytstruktur_2(h_20, h_40, h_60, h_80):
    '''
    Def rocks per hectare in each size class
    '''
    def classify(v):
        ''' Classify according to enstaka, sparsamt, måttlit, rikligt '''
        return 0 if v < 4 else 1 if v < 40 else 2 if v < 400 else 3 if v < 4000 else 4

    number_mapping = {
        0: 'Finns inga',
        1: 'Enstaka',
        2: 'Sparsamt',
        3: 'Måttligt',
        4: 'Rikligt',
    }
    none = 0
    occasional = 1
    sparse = 2
    moderate = 3
    abundant = 4

    # Calculate
    if classify(h_20) <= sparse:
        if classify(h_40 + h_60 + h_80) <= occasional:
            Y = 1
        elif classify(h_40) <= sparse and classify(h_60 + h_80) <= occasional:
            Y = 2
        elif classify(h_40) <= moderate:
            if classify(h_60) <= sparse:
                if classify(h_80) <= occasional:
                    Y = 3
                elif classify(h_80) <= sparse:
                    Y = 4
                else:
                    Y = 5
            else:
                Y = 5
        else:
            Y = 5

    elif classify(h_20) == moderate:
        if classify(h_40 + h_60 + h_80) == none:
            Y = 1
        elif classify(h_40) <= sparse and classify(h_60 + h_80) <= occasional:
            Y = 2
        elif classify(h_40) <= moderate:
            if classify(h_60) <= sparse:
                if classify(h_80) <= occasional:
                    Y = 3
                elif classify(h_80) <= sparse:
                    Y = 4
                else:
                    Y = 5
            else:
                Y = 5
        else:
            Y = 5

    else:  # classify(h_20) == abundant
        if classify(h_40 + h_60 + h_80) == none:
            Y = 2
        elif classify(h_40) <= moderate:
            if classify(h_60) <= sparse:
                if classify(h_80) <= occasional:
                    Y = 3
                elif classify(h_80) <= sparse:
                    Y = 4
                else:
                    Y = 5
            else:
                Y = 5
        else:
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


def get_terrains(
        terrain_temp=[], terrain_prim=[],
        last=None, print_fn=print, remove=True):
    ''' Get terrains from temp/heap

    Terrains are removed from the temp/heap in place if `remove` is True.
    '''

    if last is not None and last > 0:
        # When e.g. taking the last two, the slice is [-2:]
        # So here we make sure this is negative
        last = -last

    if len(terrain_temp) > 0:
        # Get terrains from temp (if last=None then all are used)
        terrains = terrain_temp[last:]
        print_fn(f"Use {len(terrains)}/{len(terrain_temp)} terrains from temp")
        # Remove any used terrains from terrain_temp in place
        if remove:
            del terrain_temp[last:]

    elif len(terrain_prim) > 0:
        # Get terrains from heap (if last=None then all are used)
        terrains = terrain_prim[last:]
        print_fn(f"Use {len(terrains)}/{len(terrain_prim)} terrains from primary")

        # Remove any used terrains from terrain_prim in place
        if remove:
            del terrain_prim[last:]
    else:
        raise AttributeError("Both terrain_temp and terrain_prim are empty")

    return terrains


def get_terrain(*args, last=None, **kwargs):
    return get_terrains(*args, last=-1, **kwargs)[0]


def find_rocks(terrain):
    """
    Identify connected high regions in a terrain and return their real-world
    centroids, maximum heights, and pixel counts.

    Returns:
    - positions: Nx2 array of (x, y) centroids in real-world coordinates
    - heights: Maximum elevation per region
    - sizes: Number of pixels in each region
    """
    import numpy as np
    import scipy.ndimage as ndimage

    labeled, num = ndimage.label(terrain.array)
    positions = []
    heights = []
    sizes = []

    for label in range(1, num + 1):
        coords = np.argwhere(labeled == label)
        sizes.append(len(coords))

        center = np.mean(coords, axis=0)
        max_idx = np.argmax(terrain.array[labeled == label])
        max_pos = coords[max_idx]
        max_val = terrain.array[max_pos[0], max_pos[1]]

        # Convert row/col center to (x, y) coordinates
        x_range = [0, terrain.array.shape[0]]
        y_range = [0, terrain.array.shape[1]]

        x = np.interp(center[0], x_range, terrain.extent[:2])
        y = np.interp(center[1], y_range, terrain.extent[2:4])
        positions.append([x, y])
        heights.append(max_val)

    return np.array(positions), np.array(heights), np.array(sizes)


class Distribution:
    def __init__(self, dist_name, *params):
        self.dist_name = dist_name
        self.params = params
        self._sequence = None
        self._sequence_iter = None
        self.dist = self._create_distribution()

    def _create_distribution(self):
        # Random distributions from numpy
        random_distributions = {
            'uniform': np.random.uniform,
            'normal': np.random.normal,
            'exponential': np.random.exponential,
            'beta': np.random.beta,
        }

        regular_distributions = {
            'arange': np.arange,
            'linspace': np.linspace,
            'logspace': np.logspace,
        }

        if self.dist_name in random_distributions:
            return lambda size: random_distributions[self.dist_name](
                *self.params, size=size)
        elif self.dist_name in regular_distributions:
            self._sequence = regular_distributions[self.dist_name](*self.params)
            self._sequence_iter = cycle(self._sequence)
            return self._sample_from_sequence
        else:
            raise ValueError(f"Unsupported distribution: {self.dist_name}")

    def _sample_from_sequence(self, size=1):
        return np.array(list(islice(self._sequence_iter, size)))

    def sample(self, size=1):
        return self.dist(size=size)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def __repr__(self):
        return f"<Distribution: {self.dist_name} params={self.params}>"


def parse_and_assign_distribution(expression):
    import re
    # Replace square parenthesis with soft
    expression = expression.replace('[', '(').replace(']', ')')
    # logger.debug(f"expression:{expression}")
    match = re.match(r"(\w+)=([\w_]+)\((.*)\)", expression)
    # match = re.match(r"(\w+):(\w+)=([\w_]+)\((.*)\)", expression)
    if not match:
        raise ValueError(f"Invalid expression format: {expression}")

    var_name, dist_name, params_str = match.groups()
    params = eval(f"({params_str})")

    # Make sure params is a tuple
    if not isinstance(params, tuple):
        params = (params,)

    dist_obj = Distribution(dist_name, *params)
    return var_name, dist_obj
