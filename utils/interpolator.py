import numpy as np
from scipy.interpolate import RegularGridInterpolator


class Interpolator():
    """
    Sample from heightfield
    """
    def __init__(self, hf_array, extent):
        # Setup the interpolator using the hf_array and extent
        self.setup_interpolator(hf_array, extent)

    def setup_interpolator(
            self, hf_array,
            extent,
            **_,
    ):
        left, right, bottom, top = extent

        x = np.linspace(left, right, hf_array.shape[0])
        y = np.linspace(bottom, top, hf_array.shape[1])

        self.interpolator = RegularGridInterpolator(
            (x, y),
            hf_array,
            method='linear',
            bounds_error=False,
            fill_value=None)

    def __call__(self, x, y):
        """
        Computes the height at a given position.

        Args:
            x (float): The x-coordinate of the position.
            y (float): The y-coordinate of the position.

        Returns:
            float: The height at the given position.
        """
        result = self.interpolator([x, y])
        return result

    def setup_points(self, size=(2, 2), resolution=(128, 128), offset_x=0):
        '''
        Setup points for sampling heightfield
        '''
        half_x = size[0] / 2
        half_y = size[1] / 2
        nx, ny = resolution
        self.resolution = resolution
        x = offset_x + np.linspace(-half_x, half_x, nx)
        y = np.linspace(-half_y, half_y, ny)

        return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    def get_height_map(self, pos: tuple, theta: float, points):
        '''
        Return a heightmap, from the 'points', positioned at 'pos'
        and angled in 'theta'.
        '''
        def rotmat(theta):
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        # Transform grid to front body frame in world coordinates
        trans_grid_points = rotmat(theta).dot(points.T).T
        trans_grid_points += np.array(pos)

        heights = self.interpolator(trans_grid_points)

        # Reshape to grid size
        heights = heights.reshape(self.resolution)
        self.heights = heights

        return heights, trans_grid_points
