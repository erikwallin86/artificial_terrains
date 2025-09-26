import numpy as np
from utils.plots import plot_obstacles, add_4m_grid
from utils.obstacles import Obstacles
from utils.utils import count_obstacles
from utils.utils import Grid
from utils.utils import check_double_counts
import os

# Prepare folder to save to
save_dir = os.path.join('tests', 'test5')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Load obstacles from file, or generate
obstacle_file = os.path.join(save_dir, 'obstacles.npz')
if os.path.isfile(obstacle_file):
    obstacles = Obstacles.from_numpy(obstacle_file)
else:
    obstacles = Obstacles.from_limits()
    obstacles.save_numpy(obstacle_file)

# Plot obstacles and 4x4 m grid
fig, ax = plot_obstacles(obstacles)
add_4m_grid(ax)
filename = os.path.join(save_dir, 'obstacles.png')
fig.savefig(filename, dpi=400)

# Define grid with 4x4 m2 cells
grid = Grid(cell_size=(4, 4))

# Compare accumulating counts, with or without double-counting
counts_array = count_obstacles(grid, obstacles, only_center=False)
counts_array2 = count_obstacles(grid, obstacles, only_center=True)

# Sum over grids
count = np.sum(counts_array.reshape(-1, 4), axis=0)
no_double_count = np.sum(counts_array2.reshape(-1, 4), axis=0)
# Calculate ratio for each obstacle size class
ratio = np.divide(count, no_double_count)
print(f"ration count / no_double_count:{ratio}")

# Plot number of cells each obstacle is part of
particle_counts = check_double_counts(grid, obstacles).squeeze()
filename = os.path.join(save_dir, 'obstacles_count.png')
fig, ax = plot_obstacles(obstacles, color=particle_counts)
add_4m_grid(ax)
fig.savefig(filename, dpi=400)

# And plot only double counts
obstacles_double = obstacles[particle_counts > 1]
filename = os.path.join(save_dir, 'obstacles_double.png')
fig, ax = plot_obstacles(obstacles_double)
add_4m_grid(ax)
fig.savefig(filename, dpi=400)
