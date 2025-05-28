from modules.data import Module, debug_decorator
import os
import numpy as np
import yaml

class Save(Module):
    create_folder = False

    ''' Save numpy terrain '''
    @debug_decorator
    def __call__(self, folder='Save',
                 default=None,
                 terrain_temp=[], terrain_heap=[],
                 overwrite=False,
                 **_):
        # Possibly set folder from 'default'.
        folder = default if default is not None else folder
        # Use folder
        basename = os.path.basename(self.save_dir)
        if basename != folder:
            dirname = os.path.dirname(self.save_dir)
            self.save_dir = os.path.join(dirname, folder)

        list_of_filenames = []

        # Create folder if needed
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # Save terrain
        for i, terrain in enumerate(terrain_heap):
            filename = f'terrain{self.file_id}_{i:05d}.npz'
            filename = os.path.join(self.save_dir, filename)
            list_of_filenames.append(filename)
            if not os.path.exists(filename) or overwrite:
                terrain.save(filename)

        # Save from terrain
        for i, terrain in enumerate(terrain_temp):
            filename = f'terrain_temp{self.file_id}_{i:05d}.npz'
            filename = os.path.join(self.save_dir, filename)
            list_of_filenames.append(filename)
            if not os.path.exists(filename) or overwrite:
                terrain.save(filename)

        return {
            'save_filenames': list_of_filenames
            }


class Load(Module):
    ''' Load terrains '''
    create_folder = False

    @debug_decorator
    def __call__(self, file_list=None, default=None, overwrite=False,
                 call_number=None, call_total=None, **pipe):
        """
        Load terrain files into structured lists.

        Args:
            file_list (list): List of filenames or patterns to load.
            default (list): Default list of filenames if file_list is None.
            call_number (int): Current call number (optional).
            call_total (int): Total number of calls (optional).
            **pipe (dict): Additional arguments.

        Returns:
            list: A list of pipes containing loaded terrain data.
        """
        # Get file-list
        file_list = default if default is not None else file_list
        # Parse any wildcards, and make sure it is a list
        file_list = self.parse_wildcards(file_list)
        # Check that files exist, and possibly prepend 'save-dir'
        file_list = self.check_exist_and_possibly_prepend_savedir(file_list)
        # Expand folders to get npz files
        file_list = self.expand_folders_to_get_npz(file_list)

        import re
        from utils.terrains import Terrain
        # Regular expressions to match filenames
        terrain_regex = re.compile(r'terrain(_\d{5})?_(\d{5}).npz')
        terrain_temp_regex = re.compile(r'terrain_temp(_\d{5})?_(\d{5}).npz')

        terrains = self.parse_regex(file_list, terrain_regex)
        terrain_temps = self.parse_regex(file_list, terrain_temp_regex)

        # Convert dictionaries to lists of lists
        terrain_list = [sorted(terrains[key].items()) for key in sorted(terrains)]
        terrain_temp_list = [sorted(terrain_temps[key].items()) for key in sorted(terrain_temps)]

        # Populate new pipes with copy of current
        pipes = [pipe.copy() for _ in range(
            max(len(terrain_temp_list), len(terrain_list)))]

        # Load to create terrain_heap
        for terrains, pipe in zip(terrain_list, pipes):
            terrain_heap = pipe.get('terrain_heap', [])
            for _, filename in terrains:
                terrain = Terrain.from_numpy(filename)
                terrain_heap.append(terrain)
            pipe['terrain_heap'] = terrain_heap

        # Load to create terrain_temp
        for terrains, pipe in zip(terrain_temp_list, pipes):
            terrain_temp = pipe.get('terrain_temp', [])
            for i, filename in terrains:
                terrain = Terrain.from_numpy(filename)
                terrain_temp.append(terrain)
            pipe['terrain_temp'] = terrain_temp

        if not pipes:
            raise ValueError("Empty pipe")
        return pipes[0] if len(pipes) == 1 else pipes

    def parse_regex(self, file_list, regex):
        data = {}
        for filename in file_list:
            match = regex.search(filename)
            if match:
                file_id = match.group(1) if match.group(1) else ''
                index = int(match.group(2))
                data.setdefault(file_id, {})[index] = filename

        return data

    def parse_wildcards(self, file_list):
        # Draft of handling wildcards
        # Not sure how to best handle this systematically
        if '*' in file_list:
            import glob
            file_list = glob.glob(file_list)
            file_list.sort()

        if '[' in file_list:
            import glob
            file_list = glob.glob(file_list)
            file_list.sort()

        if not isinstance(file_list, list):
            file_list = [file_list]

        return file_list

    def check_exist_and_possibly_prepend_savedir(self, file_list):
        new_file_list = []
        for filename in file_list:
            # Check if filename is a file or folder
            if os.path.isfile(filename):
                new_file_list.append(filename)
            elif os.path.isdir(filename):
                new_file_list.append(filename)
            else:
                # Prepend save_dir if *that* file exists
                test = os.path.join(self.save_dir, filename)
                if os.path.exists(test):
                    new_file_list.append(os.path.join(self.save_dir, filename))
                else:
                    # Or raise error
                    raise FileNotFoundError

        return new_file_list

    def expand_folders_to_get_npz(self, file_list):
        from utils.utils import get_list_of_files
        # If folder, get all npz files in it
        new_file_list = []
        for filename in file_list:
            if os.path.isdir(filename):
                folder = filename
                files = get_list_of_files([folder], '.npz')
                for f in files:
                    new_file_list.append(os.path.join(folder, f))
            else:
                new_file_list.append(filename)

        return new_file_list


class ClearTerrain(Module):
    @debug_decorator
    def __call__(self,
                 hf_array=None, hf_info_dict=None,
                 call_number=None, call_total=None, **pipe):

        return {
            'terrain_temp': [],
            'terrain_heap': [],
        }


class CombinePipes(Module):
    ''' Combine terrains from different pipes '''
    create_folder = False

    @debug_decorator
    def __call__(self, operation='Add', default=None,
                 hf_array=None, hf_info_dict=None,
                 call_number=None, call_total=None, **pipe):

        operation = default if default is not None else operation

        # If only one call, do nothing
        if call_total == 1:
            return

        if call_number == 0:
            # first call
            self.arrays = [hf_array]
            return 'remove'
        elif call_number + 1 != call_total:
            # not last call
            self.arrays.append(hf_array)
            return 'remove'

        # on last call
        self.arrays.append(hf_array)

        if operation == 'Add':
            hf_array = np.sum(self.arrays, axis=0)
        elif operation == 'Max':
            hf_array = np.maximum.reduce(self.arrays)
        elif operation == 'Min':
            hf_array = np.minimum.reduce(self.arrays)
        elif operation == 'Prod':
            hf_array = np.prod(self.arrays, axis=0)
        else:
            raise AttributeError

        return {
            'hf_array': hf_array,
            'hf_info_dict': hf_info_dict,
        }


class FindRocks(Module):
    ''' Plot terrain '''
    @debug_decorator
    def __call__(self, default=None, overwrite=False,
                 terrain=None, terrain_temp=None,
                 exportmode=True, dpi=400,
                 call_number=None, call_total=None, plot=True, **kwargs):
        from utils.plots import plot_terrain

        positions_list = []
        heights_list = []
        sizes_list = []

        if len(terrain_temp) > 0:
            terrains = terrain_temp
        else:
            terrains = [terrain]

        for i, terrain in enumerate(terrains):
            filename = f'rocks_{i}{self.file_id}.png'
            filename = os.path.join(
                self.save_dir, filename)
            if os.path.exists(filename) and not overwrite:
                # Skip if file already exists
                return

            # test 4
            positions, heights, sizes = self.localMax4(terrain.array)
            positions_list.append(positions)
            heights_list.append(heights)
            sizes_list.append(sizes)

            if plot:
                fig4, ax4 = plot_terrain(terrain)
                x4 = np.interp(positions[:, 0], [0, terrain.array.shape[0]], terrain.extent[:2])
                y4 = np.interp(positions[:, 1], [0, terrain.array.shape[1]], terrain.extent[2:4])
                x4, y4, heights, sizes = self.filter_points(x4, y4, heights=heights, sizes=sizes)
                ax4.scatter(x4, y4, facecolor='none', color='red', s=sizes)
                fig4.savefig(filename, dpi=dpi)

        filename = f'rocks_info{self.file_id}.npz'
        filename = os.path.join(self.save_dir, filename)
        np.savez(filename, position=np.concatenate(positions_list),
                 height=np.concatenate(heights_list), size=np.concatenate(sizes_list))

    def filter_points(self, x, y, heights=None, sizes=None, threshold=0.1):
        # Create a mask to identify points greater or equal to the threshold
        mask = (heights >= threshold)

        # Apply the mask to filter out the points below the threshold
        x = x[mask]
        y = y[mask]
        sizes = sizes[mask]
        heights = heights[mask]

        return x, y, heights, sizes

    def localMax4(self, array_2d):
        import scipy.ndimage as ndimage
        # Find islands using label connected components
        labelled_array, num_islands = ndimage.label(array_2d)

        # Initialize arrays to store results
        positions_central = []
        positions_highest = []
        heights_max = []
        sizes = []

        # Iterate through each island
        for island_label in range(1, num_islands+1):
            # Get the coordinates of all points in the current island
            island_coords = np.argwhere(labelled_array == island_label)
            sizes.append(len(island_coords))

            # Calculate the central point of the island
            central_point = np.mean(island_coords, axis=0)

            # Find the highest point in the island
            highest_point = island_coords[np.argmax(array_2d[labelled_array == island_label])]
            max_height = array_2d[highest_point[0], highest_point[1]]

            # Append the positions to the respective arrays
            positions_central.append(central_point)
            positions_highest.append(highest_point)
            heights_max.append(max_height)

        return np.array(positions_highest), np.array(heights_max), np.array(sizes)


class FindRocks2(Module):
    """
    Detect rock-like features from terrain data and output them as 'obstacles'.

    The method identifies local maxima (interpreted as rocks), computes their
    positions, heights, and estimated widths, and returns them in a structured
    format suitable for obstacle processing.
    """

    @debug_decorator
    def __call__(self, terrain_heap=None, terrain_temp=None,
                 default=None, last=None, **kwargs):
        """
        Parameters:
        - terrain_heap, terrain_temp: Terrain sources.
        - last: Fallback terrain if others are not provided.

        Returns:
        - dict with fields:
            - 'position': Nx2 array of (x, y) positions.
            - 'height': Estimated height of each feature.
            - 'width': Estimated width based on region size.
            - 'yaw_deg', 'pitch_deg': Zeros (placeholders).
            - 'aspect': Ones (placeholder for shape).
        """
        from utils.utils import get_terrains

        positions_all = []
        heights_all = []
        sizes_all = []

        terrains = get_terrains(
            terrain_temp, terrain_heap, last=last, remove=False,
            print_fn=self.info)

        for i, terrain in enumerate(terrains):
            # Detect local maxima and get stats
            positions, heights, sizes = self.localMax4(terrain.array)
            positions_all.append(positions)
            heights_all.append(heights)
            sizes_all.append(sizes)

        # Assume same resolution for all terrains
        resolution = terrain.resolution
        pixel_diag = np.linalg.norm(resolution)

        # Merge results
        positions = np.concatenate(positions_all)
        heights = np.concatenate(heights_all)
        sizes = np.concatenate(sizes_all)
        widths = np.sqrt(sizes) * pixel_diag

        # Convert pixel indices to physical coordinates
        x = np.interp(
            positions[:, 0], [0, terrain.array.shape[0]], terrain.extent[:2])
        y = np.interp(
            positions[:, 1], [0, terrain.array.shape[1]], terrain.extent[2:4])
        positions[:, 0] = x
        positions[:, 1] = y

        return {
            'position': positions,
            'height': heights,
            'width': widths,
            'yaw_deg': np.zeros_like(heights),
            'pitch_deg': np.zeros_like(heights),
            'aspect': np.ones_like(heights),
        }

    def localMax4(self, array_2d):
        """
        Identify connected regions (islands) and extract central and max points.

        Parameters:
        - array_2d: 2D numpy array representing elevation.

        Returns:
        - positions: Nx2 array of region centroids.
        - heights: Maximum value per region.
        - sizes: Number of pixels in each region.
        """
        import scipy.ndimage as ndimage

        labeled, num = ndimage.label(array_2d)
        positions = []
        heights = []
        sizes = []

        for label in range(1, num + 1):
            coords = np.argwhere(labeled == label)
            sizes.append(len(coords))

            center = np.mean(coords, axis=0)
            max_idx = np.argmax(array_2d[labeled == label])
            max_pos = coords[max_idx]
            max_val = array_2d[max_pos[0], max_pos[1]]

            positions.append(center)
            heights.append(max_val)

        return np.array(positions), np.array(heights), np.array(sizes)


class PlotRocks(Module):
    ''' Plot rocks
    '''
    @debug_decorator
    def __call__(self, terrain_heap=None, terrain_temp=None,
                 position=None, width=None,
                 default=None, last=None,
                 dpi=200,
                 **kwargs):
        from utils.utils import get_terrain
        from utils.plots import plot_terrain, new_fig

        # Get terrain
        try:
            terrain = get_terrain(
                terrain_temp, terrain_heap, last=last, remove=False,
                print_fn=self.info)
        except AttributeError:
            terrain = None
        except TypeError:
            terrain = None

        filename = 'rocks.png'
        filename = os.path.join(self.save_dir, filename)

        if terrain is not None:
            fig, ax = plot_terrain(terrain)
        else:
            fig, ax = new_fig()

        ax.scatter(
            # position[:, 0], position[:, 1],
            *position.T,
            facecolor='none', color='red', s=40*width**2)
        fig.savefig(filename, dpi=dpi)


class Plot(Module):
    create_folder = False

    ''' Plot terrain '''
    @debug_decorator
    def __call__(self, default=None, overwrite=False,
                 terrain_temp=[], terrain_heap=[], folder='Plot',
                 exportmode=False, dpi=400,
                 vmin=None, vmax=None,
                 call_number=None, call_total=None, **kwargs):
        from utils.plots import plot_terrain, save_all_axes

        # Possibly set folder from 'default'.
        folder = default if default is not None else folder

        list_of_filenames = []

        # Use folder
        basename = os.path.basename(self.save_dir)
        if basename != folder:
            dirname = os.path.dirname(self.save_dir)
            self.save_dir = os.path.join(dirname, folder)

        # Create folder if needed
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # Plot terrain
        for i, terrain in enumerate(terrain_heap):
            filename = f'terrain{self.file_id}_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)
            list_of_filenames.append(filename)
            if not os.path.exists(filename) or overwrite:
                fig, ax = plot_terrain(terrain, vmin=vmin, vmax=vmax)
                fig.savefig(filename, dpi=dpi)
                if exportmode:
                    save_all_axes(fig, filename, delta=0.0, dpi=dpi)

        # Plot terrain_temp
        for i, terrain in enumerate(terrain_temp):
            filename = f'terrain_temp{self.file_id}_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)
            list_of_filenames.append(filename)
            if not os.path.exists(filename) or overwrite:
                fig, ax = plot_terrain(terrain, vmin=vmin, vmax=vmax)
                fig.savefig(filename, dpi=dpi)
                if exportmode:
                    save_all_axes(fig, filename, delta=0.0, dpi=dpi)

        return {
            'plot_filenames': list_of_filenames
            }


class SaveYaml(Module):
    ''' Plot terrain '''
    @debug_decorator
    def __call__(self, default=None, overwrite=False,
                 call_number=None, call_total=None,
                 terrain_temp=[], terrain_heap=[],
                 **kwargs):

        extent = kwargs['extent']
        print(f"type(extent):{type(extent)}")

        # Initialize
        if call_number == 0:
            self.list_of_kwargs = []

        # Collect data
        self.list_of_kwargs.append(
            {**kwargs,
             'call_total': call_total,
             'call_number': call_number,
             })

        # Return, except on last call
        if call_number+1 != call_total:
            return

        # Save
        filename = 'data.yml'
        filename = os.path.join(self.save_dir, filename)
        with open(filename, 'w') as outfile:
            yaml.dump(
                self.list_of_kwargs, outfile,
                default_flow_style=None,
                allow_unicode=True,)
