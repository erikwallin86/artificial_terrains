from modules.data import Module, debug_decorator
import os
import numpy as np
import yaml
from utils.plots import save_all_axes


class Save(Module):
    create_folder = False

    ''' Save numpy terrain '''
    @debug_decorator
    def __call__(self, folder='Save',
                 default=None,
                 terrain_temp=[], terrain_prim=[],
                 overwrite=False,
                 loop_id="",
                 **_):
        # Possibly set folder from 'default'.
        folder = default if default is not None else folder

        # Use folder
        self.save_dir = os.path.join(self.save_dir_original, folder)

        list_of_filenames = []

        # Create folder if needed
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # Save terrain
        for i, terrain in enumerate(terrain_prim):
            filename = f'terrain{loop_id}_{i:05d}.npz'
            filename = os.path.join(self.save_dir, filename)
            list_of_filenames.append(filename)
            if not os.path.exists(filename) or overwrite:
                terrain.save(filename)

        # Save from terrain
        for i, terrain in enumerate(terrain_temp):
            filename = f'terrain_temp{loop_id}_{i:05d}.npz'
            filename = os.path.join(self.save_dir, filename)
            list_of_filenames.append(filename)
            if not os.path.exists(filename) or overwrite:
                terrain.save(filename)

        return {
            'save_filenames': list_of_filenames
            }


class LoadTIF(Module):
    ''' Test, this should be included in Load '''
    create_folder = False

    @debug_decorator
    def __call__(self, filename=None, default=None,
                 terrain_temp=None, terrain_prim=None,
                 loop_id="", filter_on_loop_id=True,
                 **_):
        filename = default if default is not None else filename

        # Populate primary and temporary lists
        terrain_temp = [] if terrain_temp is None else terrain_temp
        terrain_prim = [] if terrain_prim is None else terrain_prim

        from utils.terrains import Terrain
        terrain = Terrain.from_geotiff(filename)
        terrain_temp.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            'terrain_prim': terrain_prim,
        }


class Load(Module):
    ''' Load terrains '''
    create_folder = False

    @debug_decorator
    def __call__(self, file_list=None, default=None,
                 terrain_temp=None, terrain_prim=None,
                 loop_id="", filter_on_loop_id=True,
                 **_):
        """
        Load terrain files into structured lists.

        Args:
            file_list (list): List of filenames or patterns to load.
            default (list): Default list of filenames if file_list is None.

        # One specific terrain --> use that
        Load:runs/data_051_new_load_save/Save/terrain_temp_00_00000.npz

        # List of specific terrains --> use them
        Load:"['runs/data_051_new_load_save/Save/terrain_temp_00_00000.npz']"

        # One specific folder --> find all npz in it
        Load:runs/data_051_new_load_save/Save

        # List folders --> find npz in all of them
        Load:"['runs/data_051_new_load_save/Save']"

        # One folder, in a loop --> Load
        Loop:2 Load:runs/data_051_new_load_save/Save

        # Wildcards -->

        Returns:
            list: A list of pipes containing loaded terrain data.
        """
        # Get file-list
        file_list = default if default is not None else file_list

        # Populate primary and temporary lists
        terrain_temp = [] if terrain_temp is None else terrain_temp
        terrain_prim = [] if terrain_prim is None else terrain_prim

        # Parse any wildcards, and make sure it is a list
        file_list = self.parse_wildcards(file_list)
        # Check that files exist, and possibly prepend 'save-dir'
        file_list = self.check_exist_and_possibly_prepend_savedir(file_list)
        # Expand folders to get npz files
        file_list = self.expand_folders_to_get_npz(file_list)

        import re
        from utils.terrains import Terrain
        # Regular expressions to match filenames

        terrain_regex = re.compile(r"terrain(?:(_.*?))?_(\d{5})\.npz$")
        terrain_temp_regex = re.compile(r"terrain_temp(?:(_.*?))?_(\d{5})\.npz$")

        terrain_list = self.filter_files_by_regex(
            file_list, terrain_regex, loop_id, filter_on_loop_id)
        terrain_temp_list = self.filter_files_by_regex(
            file_list, terrain_temp_regex, loop_id, filter_on_loop_id)

        # Load to create terrain
        for filename in terrain_list:
            terrain = Terrain.from_numpy(filename)
            terrain_prim.append(terrain)

        # Load to create terrain_temp
        for filename in terrain_temp_list:
            terrain = Terrain.from_numpy(filename)
            terrain_temp.append(terrain)

        return {
            'terrain_temp': terrain_temp,
            'terrain_prim': terrain_prim,
        }

    def filter_files_by_regex(self, file_list, regex, loop_id, filter_on_loop_id=True):
        """Return subset of files matching regex and (optionally) loop_id."""
        new_file_list = []

        for filename in file_list:
            base = os.path.basename(filename)
            m = regex.match(base)
            if not m:
                continue

            file_loop_id = m.group(1) or ""  # extracted loop id
            if (not filter_on_loop_id or file_loop_id.startswith(loop_id)) and 'temp' not in file_loop_id:
                new_file_list.append(filename)

        return new_file_list

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
            'terrain_prim': [],
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
    """
    Detect rock-like features from terrain data and return them as 'obstacles'.

    Identifies local maxima in elevation, estimates their real-world position,
    height, and width.
    """
    @debug_decorator
    def __call__(self, terrain_prim=None, terrain_temp=None,
                 default=None, last=None, **kwargs):
        from utils.utils import get_terrains, find_rocks

        terrains = get_terrains(
            terrain_temp, terrain_prim, last=last, remove=False,
            print_fn=self.info)

        # position, height, size = zip(*(find_rocks(t) for t in terrains))
        position_list = []
        height_list = []
        size_list = []
        terrain_number_list = []

        for i, terrain in enumerate(terrains):
            position, height, size = find_rocks(terrain)
            position_list.append(position)
            height_list.append(height)
            size_list.append(size)
            terrain_number_list.append(np.full(len(height), i))

        position = np.concatenate(position_list)
        height = np.concatenate(height_list)
        size = np.concatenate(size_list)
        terrain_number = np.concatenate(terrain_number_list)

        pixel_diag = np.linalg.norm(terrains[0].resolution)
        width = np.sqrt(size) * pixel_diag

        result = {
            'position': position,
            'height': height,
            'width': width,
            'terrain_number': terrain_number,
            'yaw_deg': np.zeros_like(height),
            'pitch_deg': np.zeros_like(height),
            'aspect': np.ones_like(height),
        }

        # Filter to include only rocks of some height
        pick = (height > 0.1)
        result = {k: v[pick] for k, v in result.items()}

        return result


class PlotRocks(Module):
    ''' Plot rocks
    '''
    @debug_decorator
    def __call__(self, terrain_prim=None, terrain_temp=None,
                 position=None, width=None,
                 default=None, last=None,
                 dpi=200, exportmode=False,
                 **kwargs):
        from utils.utils import get_terrain
        from utils.plots import plot_terrain, new_fig

        # Get terrain
        try:
            terrain = get_terrain(
                terrain_temp, terrain_prim, last=last, remove=False,
                print_fn=self.info)
        except AttributeError:
            terrain = None
        except TypeError:
            terrain = None

        filename = f'rocks{self.file_id}.png'
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
        if exportmode:
            save_all_axes(fig, filename, delta=0.0, dpi=dpi)
            filename = filename.replace('.png', f'_{0:01d}.png')
            # Possibly use as image texture
            return {'texture': filename}


class Plot(Module):
    create_folder = False

    ''' Plot terrain '''
    @debug_decorator
    def __call__(self, default=None, overwrite=False,
                 terrain_temp=[], terrain_prim=[], folder='Plot',
                 filename_base='terrain',
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
        for i, terrain in enumerate(terrain_prim):
            filename = f'{filename_base}{self.file_id}_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)
            list_of_filenames.append(filename)
            if not os.path.exists(filename) or overwrite:
                fig, ax = plot_terrain(terrain, vmin=vmin, vmax=vmax)
                fig.savefig(filename, dpi=dpi)
                if exportmode:
                    save_all_axes(fig, filename, delta=0.0, dpi=dpi)

        # Plot terrain_temp
        for i, terrain in enumerate(terrain_temp):
            filename = f'{filename_base}_temp{self.file_id}_{i:05d}.png'
            filename = os.path.join(self.save_dir, filename)
            list_of_filenames.append(filename)
            if not os.path.exists(filename) or overwrite:
                fig, ax = plot_terrain(terrain, vmin=vmin, vmax=vmax)
                fig.savefig(filename, dpi=dpi)
                if exportmode:
                    save_all_axes(fig, filename, delta=0.0, dpi=dpi)

        result = {'plot_filenames': list_of_filenames}

        if exportmode:
            filename = filename.replace('.png', f'_{0:01d}.png')
            # Possibly use last result as image texture
            result['texture'] = filename

        return result


class SaveYaml(Module):
    ''' Plot terrain '''
    @debug_decorator
    def __call__(self, default=None, overwrite=False,
                 call_number=None, call_total=None,
                 terrain_temp=[], terrain_prim=[],
                 **kwargs):
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
