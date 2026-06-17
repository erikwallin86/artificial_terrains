# Change name of this file!
from dataclasses import asdict, dataclass
from typing import Any

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

        from .terrains import extent_to_size
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


def sigmoid(x: np.ndarray, x_0=0.0, y_min=0.0, y_max=1.0, steepness=1.0, asymmetry=1.0):
    """Generic sigmoid function."""
    return y_min + (y_max - y_min) / (1 + np.exp(-steepness * (x - x_0)))**asymmetry


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


class SmallLunarCraterFunction():
    """Small axisymmetric lunar crater.

    The crater is modeled based on Mahanti et al. (2018).
    Craters are classified by age into the classes A, B or C, with class A being
    the youngest. Class A craters have a curved bowl shape and a raised rim
    while class C craters are more degraded, with a straighter bowl shape and
    lower rim. Class B craters sits somewhere in the middle.

    The entire crater profile (bowl + rim) is expressed as a difference of two
    generalized sigmoid functions, where the parameters for each class have been
    selected to trace the corresponding curve in the left plot of figure 21 in
    the paper. The remaining hardcoded parameters have been selected to
    normalize the profile so that diameter, depth and rim height is adjustable.
    """

    max_width_to_diameter_ratio = 2.2
    """Ratio between max width (bowl + rim) and crater diameter (rim-to-rim).

    NOTE: Must not be lower than any of the `self._r_rim_cutoff` definitions.
    """

    @dataclass
    class _SigmoidParameters:
        x_0: float
        y_min: float
        y_max: float
        steepness: float
        asymmetry: float

    def __init__(
            self,
            position: tuple[float, float] = (0.0, 0.0),
            height: float = 0.5,
            width: float = 5.0,
            depth_ratio: float | None = None,
            rim_height: float | None = None,
            rim_height_scale: float = 1.0,
            vertical_scale: float = 1.0,
            **_):
        """Instantiates a CraterFunction.

        :param position: Crater center position.
        :param height: Rim-to-floor depth.
        :param width: Rim-to-rim diameter.
        :param depth_ratio: Optional depth-to-diameter ratio. Overrides height.
        :param rim_height: Overrides default rim-to-terrain height if not None.
        :param rim_height_scale: Scales crater rim height.
        :param vertical_scale: Scales entire crater vertically.
        """
        diameter = width
        depth = diameter * depth_ratio if depth_ratio is not None else height
        self.position = position
        self.diameter = diameter
        self.depth = depth * vertical_scale
        self._set_class()
        if rim_height is not None:
            self.rim_height = rim_height * rim_height_scale * vertical_scale
        else:
            rim_vs_bowl_scale = self._y_scale_bowl / self._y_scale_rim
            self.rim_height = depth * rim_vs_bowl_scale * rim_height_scale * vertical_scale

    def __call__(self, x: np.ndarray, y: np.ndarray):

        # Coordinates relative center.
        x = x - self.position[0]
        y = y - self.position[1]

        # Axisymmetric crater profile: only radial distance from center matters.
        rho, _phi = cartesian_to_polar(x, y)

        bowl = self.get_bowl(rho)
        rim = self.get_rim(rho)
        crater = bowl + rim
        return crater

    def get_bowl(self, r: np.ndarray) -> np.ndarray:
        """Returns a crater bowl hight field.

        The bowl is defined for radii in the range [0, diameter/2). A zero
        elevation is returned for radii outside that range.

        :param r: Crater radii.
        :return y: Height field of same shape as input.
        """
        crater_radius = self.diameter / 2
        y_floor = self.rim_height - self.depth

        r_normalized = r / crater_radius
        bowl_mask = (r_normalized >= 0) & (r_normalized < 1)

        y = np.zeros_like(r)
        y[bowl_mask] = self._get_bowl_normalized(r_normalized[bowl_mask]) * self.depth + y_floor
        return y

    def get_rim(self, r: np.ndarray) -> np.ndarray:
        """Returns a crater rim hight field.

        The rim is defined for radii in the range [diameter/2, cutoff), where
        cutoff is defined internal to the class so that the rim is strictly
        decreasing and relatively flat at the cutoff. The elevation drops to
        zero at the cutoff so that the crater fades into the terrain. A zero
        elevation is returned for radii outside that range.

        :param r: Crater radii.
        :return y: Height field of same shape as input.
        """
        crater_radius = self.diameter / 2

        r_normalized = r / crater_radius
        rim_mask = r_normalized >= 1

        y = np.zeros_like(r)
        y[rim_mask] = self._get_rim_normalized(r_normalized[rim_mask]) * self.rim_height
        return y

    def _get_bowl_normalized(self, r_normalized: np.ndarray):
        """Returns a normalized crater bowl hight field.

        The bowl is defined for normalized radii in the range [0, 1) such that
        the crater rim is at normalized radius 1. A zero elevation is returned
        for radii outside that range.

        :param r_normalized: Normalized crater radii with rim at 1.
        :return y: Height field of same shape as input.
        """
        r_bowl = r_normalized[r_normalized >= 0]

        y = np.zeros_like(r_normalized)
        y_bowl = self._double_sigmoid(r_bowl / self._r_scale, self._p1, self._p2)
        y_bowl -= self._y_min_bowl
        y_bowl *= self._y_scale_bowl
        y[:len(y_bowl)] = y_bowl
        return y

    def _get_rim_normalized(self, r_normalized: np.ndarray):
        """Returns a normalized crater rim hight field.

        The rim is defined for normalized radii in the range [1, cutoff) such
        that the crater rim is at normalized radius 1, where the cutoff is
        defined internal to the class so that the rim is strictly decreasing and
        relatively flat at the cutoff. The elevation drops to zero at the cutoff
        so that the crater fades into the terrain. A zero elevation is returned
        for radii outside that range.

        :param r_normalized: Normalized crater radii with rim at 1.
        :return y: Height field of same shape as input.
        """
        # Cutoff rim and ensure end is at zero elevation.
        r_cutoff = self._r_rim_cutoff * self._r_scale
        r_mask = r_normalized < r_cutoff
        r_rim = r_normalized[r_mask]

        y = np.zeros_like(r_normalized)
        y_rim = self._double_sigmoid(r_rim / self._r_scale, self._p1, self._p2)
        # Set end of rim to zero elevation and normalize height.
        y_rim -= self._y_min_rim
        y_rim *= self._y_scale_rim
        y[r_mask] = y_rim
        return y

    def _double_sigmoid(self, x: np.ndarray, p1: _SigmoidParameters, p2: _SigmoidParameters):
        """Returns difference of two generic sigmoid functions."""
        y1 = sigmoid(x, **asdict(p1))
        y2 = sigmoid(x, **asdict(p2))
        return y1 - y2

    def _set_class(self):
        """Sets curve fitting parameters based on depth-to-diameter ratio (d/D).

        The curve fitting parameters are used to fit the double sigmoid curve to
        figure 21 of Mahanti et al. (2018).

        From section 3.2.2 of Mahanti et al. (2018):
        - class A: d/D > 0.12
        - class B: 0.10 < d/D <= 0.12
        - class C: d/D <= 0.10
        """
        class_c_max_ratio = 0.10
        class_b_max_ratio = 0.12
        depth_ratio = self.depth / self.diameter
        if depth_ratio <= class_c_max_ratio:
            self._set_class_c()
        elif depth_ratio <= class_b_max_ratio:
            self._set_class_b()
        else:
            self._set_class_a()

    def _set_class_a(self):
        """Sets curve fitting parameters for a class A crater."""
        self._p1 = self._SigmoidParameters(0.5331, -0.0713, 0.1862, 3.6643, 1.0000)
        self._p2 = self._SigmoidParameters(1.0607, -0.0384, 0.0493, 5.4761, 1.0000)
        self._r_scale = 0.9872
        self._r_rim_cutoff = 1.72
        self._y_min_bowl = -0.001187
        self._y_scale_bowl = 6.676220
        self._y_min_rim = 0.135927
        self._y_scale_rim = 78.918183

    def _set_class_b(self):
        """Sets curve fitting parameters for a class B crater."""
        self._p1 = self._SigmoidParameters(0.4171, 1.6479, 1.7993, 4.2651, 1.0000)
        self._p2 = self._SigmoidParameters(1.0191, 1.7090, 1.6701, -4.9087, 1.0000)
        self._r_scale = 1.0111
        self._r_rim_cutoff = 2.2
        self._y_min_bowl = -0.000593
        self._y_scale_bowl = 10.037093
        self._y_min_rim = 0.090349
        self._y_scale_rim = 115.095467

    def _set_class_c(self):
        """Sets curve fitting parameters for a class C crater."""
        self._p1 = self._SigmoidParameters(0.3503, 5.0821, 5.1586, 4.9416, 1.0000)
        self._p2 = self._SigmoidParameters(1.1626, 5.0932, 5.1028, 5.9999, 1.0000)
        self._r_scale = 0.9718
        self._r_rim_cutoff = 2.2
        self._y_min_bowl = 0.000401
        self._y_scale_bowl = 16.822928
        self._y_min_rim = 0.055811
        self._y_scale_rim = 247.973697


class CraterFunction():
    def __init__(
            self,
            position: tuple[float, float] = (0, 0),
            depth_ratio: float = 0.13,
            width: float = 5.0,
            rim_height_ratio: float = 0.024,
            rim_width_ratio: float = 0.10,
            outer_radius_factor: float = 1.7,
            cavity_exponent: float = 2.0,
            **_: Any):
        """Idealized axisymmetric crater with width-scaled shape parameters.

        The cavity depth and rim height are expressed as ratios of the crater
        width, which makes the same shape model scale naturally across crater
        sizes. The cavity is a smooth bowl inside the crater radius, and the
        rim is a Gaussian ring that tapers back to zero before the outer radius.
        """
        self.position = position
        self.radius = width / 2

        self.depth = depth_ratio * width
        self.cavity_exponent = cavity_exponent
        self.rim_height = rim_height_ratio * width
        self.rim_sigma = max(rim_width_ratio * width, 1e-9)
        self.outer_radius = max(self.radius, outer_radius_factor * self.radius)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = x - self.position[0]
        y = y - self.position[1]

        # Axisymmetric crater profile: only radial distance from center matters.
        rho, _phi = cartesian_to_polar(x, y)

        normalized_radius = rho / self.radius
        # Bowl-shaped cavity that reaches maximum depth at the center and
        # returns to zero elevation at the rim crest.
        bowl = np.where(
            rho <= self.radius,
            -self.depth * (1 - normalized_radius**self.cavity_exponent),
            0,
        )

        # Raised rim represented as a Gaussian ring centered on the crater radius.
        rim = self.rim_height * np.exp(
            -0.5 * ((rho - self.radius) / self.rim_sigma)**2
        )

        if self.outer_radius > self.radius:
            # Smoothly fade the rim uplift back to the surrounding terrain.
            taper = np.where(
                rho <= self.radius,
                1,
                1 - smoothstep(
                    (rho - self.radius) / (self.outer_radius - self.radius)
                ),
            )
            rim = rim * taper

        return bowl + rim


clsmembers_pairs = inspect.getmembers(sys.modules[__name__], inspect.isclass)
FUNCTIONS = {k.replace('Function', ''): v for (k, v)
             in clsmembers_pairs if 'Function' in k}
