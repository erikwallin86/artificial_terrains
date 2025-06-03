from opensimplex import noise2array
import numpy as np
import opensimplex


def get_simplex(Nx=100, Ny=100, scale_x=50, scale_y=50, seed=None):
    '''
    Get noise with shape (Nx, Ny)
    '''
    if seed is None:
        opensimplex.random_seed()
    else:
        opensimplex.seed(seed)

    print(f"opensimplex.get_seed():{opensimplex.get_seed()}")

    f_x = Nx/scale_x
    print(f"f_x:{f_x}")
    f_y = Ny/scale_y

    x = np.linspace(0, f_x, num=Nx)
    y = np.linspace(0, f_y, num=Ny)

    # 'shift' indices due to noise2array definition
    return noise2array(y, x)


def get_simplex2(ppm=None, size=None, N=None, resolution=None,
                 scale_x=50, scale_y=50, seed=None,
                 extent=None,
                 info_dict=None,
                 random_shift=False,
                 logger_fn=None, **_):
    '''
    Get noise with shape (Nx, Ny)

    Can be defined using any two of size, points-per-meter, and total points
    Tuple input overrides individual values

    Args:
    '''
    if seed is None:
        opensimplex.random_seed()
    else:
        opensimplex.seed(seed)

    # Setup sizes etc.
    from utils.artificial_shapes import determine_extent_and_resolution
    extent, (N_x, N_y) = determine_extent_and_resolution(
        ppm, size, resolution, extent)

    size_x = extent[1] - extent[0]
    size_y = extent[3] - extent[2]

    if info_dict is not None:
        # Add to the info-dict if provided
        info_dict['extent'] = extent
        info_dict['opensimplex_seed'] = opensimplex.get_seed()
        info_dict['N'] = (N_x, N_y)
        info_dict['size'] = (size_x, size_y)

    if logger_fn is not None:
        info = f"points:{info_dict['N']}, " +\
            f"extent:{info_dict['extent']}, " +\
            f"seed:{info_dict['opensimplex_seed']}"
        logger_fn(info)

    if random_shift:
        # Add random shift, connected to opensimplex seed to get
        # repeatable results
        import random
        seed = opensimplex.get_seed()
        rng = random.Random(seed)
        # Generate two random numbers in [0, 1)
        rx = rng.random()
        ry = rng.random()
        # Add random shift to the extent
        import copy
        extent = copy.deepcopy(extent)
        extent[:2] = np.add(extent[:2], rx*scale_x)
        extent[2:] = np.add(extent[2:], ry*scale_y)

    x = np.linspace(extent[0]/scale_x, extent[1]/scale_x, num=N_x, endpoint=False)
    y = np.linspace(extent[2]/scale_y, extent[3]/scale_y, num=N_y, endpoint=False)

    # 'shift' indices due to noise2array definition
    return noise2array(y, x)


def generate_points_from_2D_prob(prob_2d, N=100):
    '''
    Sample N random points from a 2D probability distribution

    Return:
      np.array of shape (2, N) of dtype int64
    '''
    # Update the 'flat_probability'
    flat_probability = prob_2d.flatten()

    # Construct generator
    rng = np.random.default_rng()

    # Generate samples
    sample_indices = rng.choice(
        a=flat_probability.size,
        size=N,
        p=flat_probability,
    )

    # Unravel indices to 2D shape again
    adjusted_indices = np.unravel_index(
        sample_indices, prob_2d.shape)

    return np.array(adjusted_indices)
