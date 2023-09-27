from datahandlers.data import DataHandler, debug_decorator
import os
import numpy as np


class LoadTerrain(DataHandler):
    ''' Load terrains '''
    create_folder = False

    @debug_decorator
    def __call__(self, file_list=None, default=None,
                 call_number=None, call_total=None, **pipe):
        from utils.utils import get_list_of_files

        # Create empty pipes list
        pipes = []

        # If hf already in pipe, make sure to add this first
        if 'hf_array' in pipe:
            pipes.append(pipe)

        # Make sure file-list is a list
        file_list = default if default is not None else file_list

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

        # Load all terrain-files
        for filename in new_file_list:
            # Load data
            loaded_data = np.load(filename, allow_pickle=True)
            # Split in 'array' and 'info_dict'
            hf_info_dict = {}
            for key in loaded_data.files:
                if key == 'heights':
                    hf_array = loaded_data['heights']
                else:
                    hf_info_dict[key] = loaded_data[key]

            new_pipe = pipe.copy()
            new_pipe['hf_array'] = hf_array
            new_pipe['hf_info_dict'] = hf_info_dict
            pipes.append(new_pipe)

        if len(pipes) == 0:
            raise ValueError("Empty pipe")
        elif len(pipes) == 1:
            # Just return pipe
            return pipes[0]
        else:
            return pipes


class Save(DataHandler):
    create_folder = False

    ''' Save numpy terrain '''
    @debug_decorator
    def __call__(self, filename="terrain.npz", folder='Save',
                 default=None,
                 terrain=None, terrain_dict={},
                 overwrite=False,
                 call_number=None, call_total=None, **kwargs):
        # Possibly set folder from 'default'.
        folder = default if default is not None else folder
        # Use folder
        basename = os.path.basename(self.save_dir)
        if basename != folder:
            dirname = os.path.dirname(self.save_dir)
            self.save_dir = os.path.join(dirname, folder)

        # Create folder if needed
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # Save terrain
        filename = os.path.join(self.save_dir, f'terrain{self.file_id}.npz')
        if terrain is not None:
            if not os.path.exists(filename) or overwrite:
                terrain.save(filename)

        # Save from terrain-dict
        for name, terrain in terrain_dict.items():
            filename = f'terrain_dict_{name}{self.file_id}.npz'
            filename = os.path.join(self.save_dir, filename)
            if not os.path.exists(filename) or overwrite:
                terrain.save(filename)


class Load(DataHandler):
    ''' Load terrains '''
    create_folder = False

    @debug_decorator
    def __call__(self, file_list=None, default=None,
                 call_number=None, call_total=None, **pipe):
        from utils.utils import get_list_of_files

        # Create empty pipes list
        pipes = []

        # If hf already in pipe, make sure to add this first
        if 'hf_array' in pipe:
            pipes.append(pipe)

        # Make sure file-list is a list
        file_list = default if default is not None else file_list

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

        # Possibly prepend 'save_dir' if input is neither file nor dir
        new_file_list = []
        for filename in file_list:
            if os.path.isdir(filename):
                new_file_list.append(filename)
            elif os.path.isdir(filename):
                new_file_list.append(filename)
            else:
                test = os.path.join(self.save_dir, filename)
                print(f"test:{test}")
                print(f"os.path.exists(test):{os.path.exists(test)}")
                if os.path.exists(test):
                    new_file_list.append(os.path.join(self.save_dir, filename))
        file_list = new_file_list
        print(f"file_list:{file_list}")

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

        # tests
        parsed_files = []
        dict_files = {}
        for filename in new_file_list:
            basename = os.path.basename(filename)
            if 'terrain_dict' in basename:
                split = basename.split('_')
                # Get digits, or make up
                if len(split) == 4:
                    # terrain_dict_0.5_00001.npz
                    digits = split[3].replace('.npz', '')
                else:
                    digits = 'no_digits'
                # Add to dict
                if digits in dict_files:
                    dict_files[digits].append(filename)
                else:
                    dict_files[digits] = [filename]
            else:
                parsed_files.append(filename)
                import re
                if '_' in basename:
                    digits_match = re.search(r'\d{5}', filename)
                    digits = digits_match.group() if digits_match is not None else None

        print(f"dict_files:{dict_files}")
        print(f"parsed_files:{parsed_files}")
        import itertools

        # Load all terrain-files
        for filename, (digits, filenames) in itertools.zip_longest(
                parsed_files, dict_files.items()):
            from utils.terrains import Terrain
            new_pipe = pipe.copy()
            if filename is not None:
                terrain = Terrain.from_numpy(filename)
                new_pipe['terrain'] = terrain

            terrain_dict = {}
            for filename in filenames:
                name = os.path.basename(filename).split('_')[2]
                terrain_dict[name] = Terrain.from_numpy(filename)

            new_pipe['terrain_dict'] = terrain_dict

            pipes.append(new_pipe)

        if len(pipes) == 0:
            raise ValueError("Empty pipe")
        elif len(pipes) == 1:
            # Just return pipe
            return pipes[0]
        else:
            return pipes


class ClearTerrain(DataHandler):
    @debug_decorator
    def __call__(self,
                 hf_array=None, hf_info_dict=None,
                 call_number=None, call_total=None, **pipe):
        return pipe


class CombinePipes(DataHandler):
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


class FindRocks(DataHandler):
    ''' Plot terrain '''
    @debug_decorator
    def __call__(self, default=None, overwrite=False,
                 terrain=None, terrain_dict=None,
                 exportmode=True, dpi=400,
                 call_number=None, call_total=None, plot=True, **kwargs):
        from utils.plots import plot_terrain

        positions_list = []
        heights_list = []
        sizes_list = []

        if len(terrain_dict) > 0:
            terrains = terrain_dict.values()
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
            max_height = array_2d[*highest_point]

            # Append the positions to the respective arrays
            positions_central.append(central_point)
            positions_highest.append(highest_point)
            heights_max.append(max_height)

        return np.array(positions_highest), np.array(heights_max), np.array(sizes)


class Plot(DataHandler):
    create_folder = False

    ''' Plot terrain '''
    @debug_decorator
    def __call__(self, filename="terrain.png", default=None, overwrite=False,
                 terrain=None, terrain_dict={}, folder='Plot',
                 exportmode=False, dpi=400,
                 call_number=None, call_total=None, **kwargs):
        from utils.plots import plot_terrain, save_all_axes
        from utils.utils import add_id

        # Possibly set folder from 'default'.
        folder = default if default is not None else folder
        # Use folder
        basename = os.path.basename(self.save_dir)
        if basename != folder:
            dirname = os.path.dirname(self.save_dir)
            self.save_dir = os.path.join(dirname, folder)

        # Create folder if needed
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # Plot terrain
        if terrain is not None:
            filename = os.path.join(self.save_dir, filename)
            filename = add_id(filename, self.file_id)
            if not os.path.exists(filename) or overwrite:
                fig, ax = plot_terrain(terrain)
                fig.savefig(filename, dpi=dpi)
                if exportmode:
                    save_all_axes(fig, filename, delta=0.0, dpi=dpi)

        # Plot terrain_dict
        for name, terrain in terrain_dict.items():
            filename = f'terrain_dict_{name}{self.file_id}.png'
            filename = os.path.join(self.save_dir, filename)
            if not os.path.exists(filename) or overwrite:
                fig, ax = plot_terrain(terrain)
                fig.savefig(filename, dpi=dpi)
                if exportmode:
                    save_all_axes(fig, filename, delta=0.0, dpi=dpi)
