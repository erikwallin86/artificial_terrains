# Change name of this file!
import numpy as np
import sys
import inspect


def get_meshgrid(extent, N_x, N_y):
    # Generating the x and y coordinates
    x = np.linspace(extent[0], extent[1], N_x, endpoint=False)
    y = np.linspace(extent[2], extent[3], N_y, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    return X, Y


def rotate_meshgrid(X, Y, theta_degree):
    '''

    Args:
      theta_degree: angle, in degrees
    '''
    theta = theta_degree/180.0*np.pi

    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Flatten the grids
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # Apply the rotation to each point
    # Stack x and y for matrix multiplication
    xy_points = np.vstack([x_flat, y_flat])
    # Matrix multiplication
    rotated_points = rotation_matrix @ xy_points

    # Reshape the rotated points back to grid shape
    X_rotated = rotated_points[0, :].reshape(X.shape)
    Y_rotated = rotated_points[1, :].reshape(Y.shape)

    return X_rotated, Y_rotated


def cartesian_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return (rho, phi)


def determine_extent_and_resolution(resolution=None, size=None, grid_size=None, extent=None):
    '''
    Determine extent and resolution.

    Default: use extent (m) and resolution (points/m) to decide -> (extent, grid_size)
    
    If grid_size explicitly given, use that instead.
    '''
    if resolution is not None:
        # Default, use resolution
        if isinstance(resolution, list):
            resolution_x, resolution_y = resolution
        else:
            resolution_x, resolution_y = (resolution, resolution)

        from utils.terrains import extent_to_size
        [size_x, size_y] = extent_to_size(extent)
        grid_size_x = int(size_x * resolution_x)
        grid_size_y = int(size_y * resolution_y)
    elif grid_size is not None:
        # But use grid-size if explicitly given
        if isinstance(grid_size, list):
            grid_size_x, grid_size_y = grid_size
        else:
            grid_size_x, grid_size_y = (grid_size, grid_size)

    return extent, [grid_size_x, grid_size_y]


def gaussian_1d(x, mu=0, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)


def smoothstep(x):
    # 'smoothstep' between x=0 and x=1, using 3x^2 - 2x^3
    return np.where(x <= 0, 0, np.where(x >= 1, 1, 3 * x**2 - 2 * x**3))


class GaussianFunction():
    def __init__(self, position=(0, 0), height=2.5, width=5, yaw_deg=0, aspect=1,
                 **_):
        self.pos = position
        self.r_x = 0.5*width / np.sqrt(aspect)
        self.r_y = 0.5*width * np.sqrt(aspect)
        self.A = height
        self.yaw_deg = yaw_deg

    def __call__(self, x, y):
        # Adjust position
        # NOTE: 'shifted, maybe due to our 'transpose' habit
        x = x - self.pos[0]
        y = y - self.pos[1]

        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        return self.A * gaussian_1d(x, sigma=self.r_x)\
            * gaussian_1d(y, sigma=self.r_y)


class SphereFunction():
    def __init__(self, position=(0, 0), height=2.5, width=5, yaw_deg=0, aspect=1,
                 **_):
        '''
          width: diameter of sphere
          height: maximum height
          aspect:
        '''
        self.pos = position
        self.height = height
        self.yaw_deg = yaw_deg
        self.radius = width/2
        self.aspect = aspect

    def __call__(self, x, y):
        # Adjust position
        # NOTE: 'shifted, maybe due to our 'transpose' habit
        x = x - self.pos[0]
        y = y - self.pos[1]

        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        # Calculate distance from center after rotation
        distance = np.sqrt((x*np.sqrt(self.aspect))**2 + (y/np.sqrt(self.aspect))**2)

        # Calculate height
        sphere_height = np.sqrt(self.radius**2 - distance**2)
        # Adjust sphere position sphere in z
        sphere_height = sphere_height - self.radius + self.height
        # Set to 0 outside
        sphere_height = np.where(sphere_height >= 0, sphere_height, 0)

        return sphere_height


class StepFunction():
    def __init__(self, position=(0, 0), height=2.5, yaw_deg=0, **_):
        self.pos = position
        self.A = height
        self.yaw_deg = yaw_deg

    def __call__(self, x, y):
        # Adjust position
        # NOTE: 'shifted, maybe due to our 'transpose' habit
        x = x - self.pos[0]
        y = y - self.pos[1]

        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        return self.A * np.greater(x, 0)


class SmoothStepFunction():
    def __init__(self, position=(0, 0), height=2.5, yaw_deg=0, **_):
        self.pos = position
        self.A = height
        self.yaw_deg = yaw_deg
        self.step_length = height

    def __call__(self, x, y):
        # Adjust position
        # NOTE: 'shifted, maybe due to our 'transpose' habit
        x = x - self.pos[0]
        y = y - self.pos[1]

        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        # Calculate smoothstep function for each element in x
        smooth_x = smoothstep((x) / self.step_length)

        return self.A * smooth_x


class DonutFunction():
    # Even if this a Gaussian shape
    def __init__(self, position=(0, 0), height=2.5, width=5, yaw_deg=0, aspect=1,
                 **_):
        self.pos = position
        self.A = height
        self.radius = width/2
        self.aspect = aspect
        self.yaw_deg = yaw_deg

    def __call__(self, x, y):
        # Adjust position
        # NOTE: 'shifted, maybe due to our 'transpose' habit
        x = x - self.pos[0]
        y = y - self.pos[1]

        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        # Stretch coordinates
        x = x * np.sqrt(self.aspect)
        y = y / np.sqrt(self.aspect)

        # Turn to polar coordinates
        rho, phi = cartesian_to_polar(x, y)

        return self.A * gaussian_1d(rho, mu=self.radius, sigma=self.A/2)


class PlaneFunction():
    def __init__(self, position=(0, 0), pitch_deg=0, yaw_deg=0, **_):
        self.pitch_deg = pitch_deg
        self.yaw_deg = yaw_deg
        self.pos = position
        # TODO: Fixa så att planet går igenom punkten!

    def __call__(self, x, y):
        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        # Setup angle
        tan = np.tan(self.pitch_deg*np.pi/180.0)

        return y * tan


class SineFunction():
    def __init__(self, yaw_deg=0, height=1, width=2, **_):
        self.yaw_deg = yaw_deg
        self.amplitude = height/2
        self.width = width

    def __call__(self, x, y):
        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        return self.amplitude * np.sin(y*2*np.pi/self.width)


class CubeFunction():
    def __init__(self, position=(0, 0), height=2.5, width=5, yaw_deg=0, aspect=1, **_):
        self.position = position
        self.height = height
        self.width_x = width * np.sqrt(aspect)
        self.width_y = width / np.sqrt(aspect)
        self.yaw_deg = yaw_deg
        self.aspect = aspect

    def __call__(self, x, y):
        # Adjust position
        # NOTE: 'shifted, maybe due to our 'transpose' habit
        x = x - self.position[0]
        y = y - self.position[1]

        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        # Calculate cube
        x_dir = np.logical_and(x > -self.width_x/2, x < self.width_x/2)
        y_dir = np.logical_and(y > -self.width_y/2, y < self.width_y/2)

        return np.logical_and(x_dir, y_dir) * self.height


class SmoothCubeFunction():
    def __init__(self, position=(0, 0), height=2.5, width=10, yaw_deg=0, aspect=1, **_):
        self.position = position
        self.height = height
        self.width_x = width * np.sqrt(aspect)
        self.width_y = width / np.sqrt(aspect)
        self.yaw_deg = yaw_deg
        self.aspect = aspect
        smooth_width = height
        self.smooth_width = smooth_width  # smoothing distance (in same units as x,y)

    def __call__(self, x, y):
        # Translate
        x = x - self.position[0]
        y = y - self.position[1]

        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        # Compute distance to cube edges (positive inside, negative outside)
        dx = self.width_x/2 - np.abs(x)
        dy = self.width_y/2 - np.abs(y)
        dist = np.minimum(dx, dy)

        # Apply smoothing near edges
        t = 0.5 + 0.5 * dist / self.smooth_width
        smooth_val = smoothstep(t)

        return smooth_val * self.height


clsmembers_pairs = inspect.getmembers(sys.modules[__name__], inspect.isclass)
FUNCTIONS = {k.replace('Function', '').lower(): v for (k, v)
             in clsmembers_pairs if 'Function' in k}
