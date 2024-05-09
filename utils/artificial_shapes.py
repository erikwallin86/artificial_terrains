# Change name of this file!
import numpy as np
import sys
import inspect


def get_meshgrid(size_x, size_y, N_x, N_y):
    # Generating the x and y coordinates
    x = np.linspace(-size_x/2, size_x/2, N_x)
    y = np.linspace(-size_y/2, size_y/2, N_y)
    X, Y = np.meshgrid(x, y)

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


def determine_size_and_resolution(ppm=None, size=None, N=None):
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

    # if N_x is None:
    #     N_x = int(size_x * ppm_x)
    # if N_y is None:
    #     N_y = int(size_y * ppm_y)
    if N is None:
        N_x = int(size_x * ppm_x)
        N_y = int(size_y * ppm_y)

    return size_x, size_y, N_x, N_y


def gaussian_1d(x, mu=0, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)


def smoothstep(x):
    # 'smoothstep' between x=0 and x=1, using 3x^2 - 2x^3
    return np.where(x <= 0, 0, np.where(x >= 1, 1, 3 * x**2 - 2 * x**3))


class GaussianFunction():
    def __init__(self, position=(0, 0), height=2.5, width=5, yaw_deg=0, aspect=1,
                 **_):
        self.pos = position
        self.r_x = width / np.sqrt(aspect)
        self.r_y = width * np.sqrt(aspect)
        self.A = height
        self.yaw_deg = yaw_deg

    def __call__(self, x, y):
        # Adjust position
        # NOTE: 'shifted, maybe due to our 'transpose' habit
        x = x - self.pos[1]
        y = y - self.pos[0]

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
        x = x - self.pos[1]
        y = y - self.pos[0]

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
        x = x - self.pos[1]
        y = y - self.pos[0]

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
        x = x - self.pos[1]
        y = y - self.pos[0]

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
        self.A = height/2
        self.width = width
        self.aspect = aspect
        self.yaw_deg = yaw_deg

    def __call__(self, x, y):
        # Adjust position
        # NOTE: 'shifted, maybe due to our 'transpose' habit
        x = x - self.pos[1]
        y = y - self.pos[0]

        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        # Stretch coordinates
        x = x * np.sqrt(self.aspect)
        y = y / np.sqrt(self.aspect)

        # Turn to polar coordinates
        rho, phi = cartesian_to_polar(x, y)

        return self.A * gaussian_1d(rho, mu=self.width, sigma=self.A/2)


class PlaneFunction():
    def __init__(self, pitch_deg=0, yaw_deg=0, **_):
        self.pitch_deg = pitch_deg
        self.yaw_deg = yaw_deg

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
        x = x - self.position[1]
        y = y - self.position[0]

        # Rotate
        x, y = rotate_meshgrid(x, y, self.yaw_deg)

        # Calculate cube
        x_dir = np.logical_and(x > -self.width_x/2, x < self.width_x/2)
        y_dir = np.logical_and(y > -self.width_y/2, y < self.width_y/2)

        return np.logical_and(x_dir, y_dir) * self.height


clsmembers_pairs = inspect.getmembers(sys.modules[__name__], inspect.isclass)
FUNCTIONS = {k.replace('Function', '').lower(): v for (k, v)
             in clsmembers_pairs if 'Function' in k}
