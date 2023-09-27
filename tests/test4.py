from utils.noise import get_simplex
import numpy as np
from utils.plots import plot_diverging
from utils.utils import calc_y
from utils.plots import (plot_obstacles, plot_probability, add_4m_grid, add_grid)
from utils.obstacles import Obstacles
from utils.utils import count_obstacles
from utils.utils import Grid
from utils.plots import plot_y_array
import os

save_dir = os.path.join('tests', 'test4')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Generate 2d noise, and plot
simplex_noise = get_simplex(Nx=200, Ny=200)
filename = os.path.join(save_dir, 'simplex_noise.png')
plot_diverging(simplex_noise, filename)

# Use noise to generate probability
rescaled = np.interp(simplex_noise, [-1, 1], [0.0, 1])
probability = rescaled/np.sum(rescaled)

obstacles = Obstacles.from_Y(3, position_distribution=probability)

# Plot particles and probability
fig, ax = plot_probability(probability)
plot_obstacles(obstacles, fig_ax=(fig, ax))
add_4m_grid(ax, size=1)
fig.savefig(os.path.join(save_dir, 'prob.png'), dpi=400)

for size in [200, 100, 50, 25, 4]:
    grid = Grid(size=(200, 200), cell_size=(size, size))

    # Count obstacles, and calculate y
    counts_array = count_obstacles(grid, obstacles, only_center=False)
    y_array = calc_y(counts_array, ha=grid.area_ha)

    # Plot 2d array together with particles
    fig, ax = plot_y_array(y_array, grid)
    add_grid(ax, size=grid.x_size)
    plot_obstacles(obstacles, fig_ax=(fig, ax))
    filename = os.path.join(save_dir, f'count_{size:05d}.png')
    fig.savefig(filename, dpi=400)

    # Plot hisograms of y-values
    from utils.plots import plot_histogram
    fig, ax = plot_histogram(y_array, 5, [1, 5])
    filename = os.path.join(save_dir, f'y_hist_{size:05d}.png')
    fig.savefig(filename)
