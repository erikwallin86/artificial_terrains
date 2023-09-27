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


def get_simplex2(ppm_x=2, ppm_y=2, size_x=50, size_y=50, N_x=None, N_y=None,
                 ppm=None, size=None, N=None,
                 scale_x=50, scale_y=50, dx=0, dy=0, seed=None,
                 center=True,
                 info_dict=None,
                 logger_fn=None, **_):
    '''
    Get noise with shape (Nx, Ny)

    Can be defined using any two of size, points-per-meter, and total points
    Tuple input overrides individual values

    Args:
      Nx: (int) grid points in x
      Ny: (int) grid points in y
      scale_x:
      scale_y:
      dx: (float) displacement in x-diretion, in [0, 1]
      dy: (float) displacement in y-diretion, in [0, 1]
    '''
    if seed is None:
        opensimplex.random_seed()
    else:
        opensimplex.seed(seed)

    # Update parameters from any tuple input
    if ppm is not None:
        if isinstance(ppm, list):
            ppm_x, ppm_y = ppm
        else:
            ppm_x, ppm_y = (ppm, ppm)
    if size is not None:
        if isinstance(size, list):
            size_x, size_y = size
        else:
            size_x, size_y = (size, size)
    if N is not None:
        if isinstance(N, list):
            N_x, N_y = N
        else:
            N_x, N_y = (N, N)

    if N_x is None:
        N_x = int(size_x * ppm_x)
    if N_y is None:
        N_y = int(size_y * ppm_y)

    f_x = size_x/scale_x
    f_y = size_y/scale_y

    # calculate
    dx_ = dx*f_x
    dy_ = dy*f_y

    if info_dict is not None:
        # Add to the info-dict if provided
        if center:
            info_dict['simplex_extent'] = [-f_x/2+dx_, f_x/2+dx_, -f_y/2+dy_, f_y/2+dy_]
            info_dict['extent'] = [-size_x/2, size_x/2, -size_y/2, size_y/2]
        else:
            info_dict['simplex_extent'] = [dx_, f_x+dx_, dy_, f_y+dy_]
            info_dict['extent'] = [0, size_x, 0, size_y]

        info_dict['simplex_size'] = [f_x, f_y]
        info_dict['opensimplex_seed'] = opensimplex.get_seed()
        info_dict['N'] = (N_x, N_y)
        info_dict['size'] = (size_x, size_y)
        info_dict['ppm'] = (ppm_x, ppm_y)
        info_dict['center'] = (dx*size_x, dy*size_y)

    if logger_fn is not None:
        info = f"scale:{info_dict['simplex_size']}, " +\
            f"points:{info_dict['N']}, " +\
            f"extent:{info_dict['extent']}, " +\
            f"seed:{info_dict['opensimplex_seed']}"
        logger_fn(info)

    if center:
        x = np.linspace(-f_x/2+dx_, f_x/2+dx_, num=N_x)
        y = np.linspace(-f_y/2+dy_, f_y/2+dy_, num=N_y)
    else:
        # Legacy setting
        x = np.linspace(0+dx_, f_x+dx_, num=N_x)
        y = np.linspace(0+dy_, f_y+dy_, num=N_y)

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
