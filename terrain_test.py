from utils.obstacles import Obstacles
from utils.plots import plot_obstacles
from utils.noise import get_simplex
from utils.plots import plot_image
import numpy as np


# obstacles = Obstacles(Y=3)
# plot_obstacles(obstacles, 'test.png')
# obstacles.save_numpy('obstacles.npz')

N = 100

# Generate random noise with large features
test = []

heights = []
xs = [1/4096, 1/2048, 1/1024, 1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

xs = np.logspace(-10, 3, 12, base=2)


simplex_noise_list = []
for i, x in enumerate(xs):
    print(f"(i, x):{(i, x)}")
    t = 1.75/(1+x*0.75)
    test.append(t)

    simplex_noise = get_simplex(Nx=N, Ny=N, scale_x=N*x, scale_y=N*x)/t
    plot_image(simplex_noise, filename=f'noise_{i:05d}.png', vmin=-1, vmax=1)
    heights.append(np.max(simplex_noise)-np.min(simplex_noise))
    simplex_noise_list.append(simplex_noise)

print(f"xs:{xs}")
print(f"heights:{heights}")

from utils.plots import new_fig
fig, ax = new_fig()
ax.plot(xs, heights)
ax.plot(xs, test)
ax.set_xscale('log')
fig.savefig('scaling.png')

simplex_noise_array = np.array(simplex_noise_list)
print(f"simplex_noise_array.shape:{simplex_noise_array.shape}")

sum1 = np.sum(simplex_noise_array, axis=0)
plot_image(sum1, filename='sum.png', vmin=-1, vmax=1)

simplex_noise_array = simplex_noise_array * (2*xs+0.1).reshape(-1, 1, 1)
for i, x in enumerate(xs):
    test = simplex_noise_array[-i:]
    print(f"{(x, test.shape)}")
    sum1 = np.sum(test, axis=0)
    plot_image(sum1, filename=f'sum_{i:05d}.png')
    from utils.terrains import Terrain
    terrain = Terrain.from_array(sum1)
    terrain.save(f'test_terrain_{i:05d}')

exit(0)

# Generate fine noise

simplex_fine = get_simplex(Nx=N, Ny=N, scale_x=5, scale_y=5)

# Apply small noise using some coarse mask
combination_1_2 = simplex_noise + simplex_fine*0.1

simplex_mask = get_simplex(Nx=N, Ny=N)
simplex_mask = np.clip(simplex_mask*2, 0, 1)

combination_1_2_mask1 = simplex_noise*3 + simplex_fine*0.3*simplex_mask

plot_image(simplex_mask, filename='mask_1.png')

plot_image(simplex_noise, filename='noise_1.png')
plot_image(simplex_fine, filename='noise_2.png')
plot_image(combination_1_2, filename='combination_1_2.png')
plot_image(combination_1_2_mask1, filename='combination_1_2_mask1.png')

from utils.terrains import Terrain
terrain = Terrain.from_array(combination_1_2_mask1)
terrain.save('test_terrain')
