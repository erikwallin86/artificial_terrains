from datahandlers.data import DataHandler, debug_decorator
import os
import sys
import numpy as np


def fix_blender_path():
    try:
        import bpy
        # Make sure current dir is in path. A blender py workaround
        dir = os.path.dirname(bpy.data.filepath)
        if dir not in sys.path:
            sys.path.append(dir)
    except Exception:
        pass


def fix_blender_argv():
    # Fix to read command line arguments (after --)
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv)
    argv = argv[index:]
    sys.argv = ['blender.py'] + argv


class Clean(DataHandler):
    ''' Remove Cube and Camera '''
    create_folder = False

    @debug_decorator
    def __call__(self, data=None, default=None, **_):
        from utils.Blender import remove_object
        remove_object(name='Cube')


class Ground(DataHandler):
    ''' Print array info '''
    create_folder = False

    @debug_decorator
    def __call__(self, filename=None, default=None,
                 hf_array=None, hf_info_dict=None,
                 ground_material=None, **_):
        filename = default if default is not None else filename
        if filename is not None:
            # Load data
            loaded_data = np.load(filename)
            # Split in 'array' and 'info_dict'
            hf_info_dict = {}
            for key in loaded_data.files:
                if key == 'heights':
                    hf_array = loaded_data['heights']
                else:
                    hf_info_dict[key] = loaded_data[key]
        else:
            assert hf_info_dict is not None, 'No hf_array'

        # Make grid from array
        from utils.Blender import grid_from_array

        name = 'Ground' + self.file_id
        grid_obj = grid_from_array(hf_array, name=name, **hf_info_dict)

        if ground_material is not None:
            grid_obj.data.materials.append(ground_material)

        return {
            'hf_array': hf_array,
            'hf_info_dict': hf_info_dict,
            'grid_obj': grid_obj,
            }


class Setup(DataHandler):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self, hf_info_dict=None, default=None, grid_obj=None, **_):
        from utils.Blender import (
            get_material, setup_z_coord_shader, add_grid, setup_lights)
        # Create Groundmaterial
        ground_material = get_material(name='GroundMaterial')
        ground_material.use_nodes = True

        # Setup groundmaterial
        setup_z_coord_shader(ground_material)

        # Add grid image
        grid = True
        grid_kwargs = {}
        if grid:
            add_grid(ground_material, **grid_kwargs)

        # Setup lights
        if hf_info_dict is not None:
            size = hf_info_dict['size']
            sun_position = (0, size[0], 70)
        else:
            sun_position = (0, 70, 70)

        use_hdri = True
        light_kwargs = {'level': 10}
        setup_lights(use_hdri, sun_position=sun_position, **light_kwargs)

        if grid_obj is not None:
            grid_obj.data.materials.append(ground_material)

        return {
            'ground_material': ground_material,
            }


class ImageTexture(DataHandler):
    ''' Add image texture to ground'''
    create_folder = False

    @debug_decorator
    def __call__(self, ground_material=None, filename=None, default=None, **_):
        from utils.Blender import (use_image_texture)

        filename = default if default is not None else filename
        image_texture_kwargs = {}
        use_image_texture(ground_material, filename, **image_texture_kwargs)


class ColorMap(DataHandler):
    ''' Add image colormap'''
    create_folder = False

    @debug_decorator
    def __call__(self, ground_material=None, filename=None, default=None, **_):
        from utils.Blender import (colormap_to_colorramp)
        import matplotlib.cm as cm
        cmap = cm.viridis
        cr = ground_material.node_tree.nodes[
            'Color Ramp'].color_ramp
        colormap_to_colorramp(cr, cmap)


class GenericCamera(DataHandler):
    ''' Setup camera'''
    create_folder = False

    @debug_decorator
    def __call__(self, camera='top', default=None, **_):
        from utils.Blender import (
            setup_camera, set_view_to_camera)

        # Setup generic camera and return
        setup_camera(resolution_x=500, resolution_y=500)
        set_view_to_camera()
        hf_info_dict = {
            'size': [50, 50],
            }

        return {
            'hf_info_dict': hf_info_dict,
        }


class Camera(DataHandler):
    ''' Setup camera'''
    create_folder = False

    @debug_decorator
    def __call__(self, camera='top', default=None,
                 hf_array=None, hf_info_dict=None,
                 **_):
        from utils.Blender import (
            setup_top_camera, set_view_to_camera, setup_angled_camera)

        camera = default if default is not None else camera
        print(f"camera:{camera}")
        assert camera in ['top', 'angled']
        assert hf_info_dict is not None, 'Run LoadTerrain before'

        size = hf_info_dict['size']
        camera_kwargs = {}
        # Setup camera
        if camera == 'top':
            camera = setup_top_camera(hf_array, size)
        elif camera == 'angled':
            middle_z = hf_array[int(hf_array.shape[0]/2), int(hf_array.shape[1]/2)]
            camera = setup_angled_camera(
                center=[0, 0, middle_z], distance=size[0]*2, **camera_kwargs)
        set_view_to_camera()


class AnimateCube(DataHandler):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self, filename=None, default=None,
                 hf_array=None, hf_info_dict=None,
                 **_):
        from utils.Blender import animate_cube, camera_to_follow_cube
        filename = default if default is not None else filename
        filename = '/home/erik/csgit/trafficability_experiments/runs/data_019_film_fram/03_test/PathToBlender/camera.npz'
        import bpy
        import os
        # animate cube
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        cube = bpy.context.object
        camera_to_follow_cube(cube)
        animate_cube(cube, filename)
        # Set the frame rate
        bpy.context.scene.render.fps = 5
        bpy.context.scene.render.fps_base = 1.029
        bpy.context.scene.frame_end = 574
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        # bpy.context.scene.render.resolution_x = 1920  # Width
        # bpy.context.scene.render.resolution_y = 1080  # Height
        bpy.context.scene.render.resolution_x = 1280  # Width
        bpy.context.scene.render.resolution_y = 720  # Height
        filename = os.path.join(self.save_dir, 'test.mkv')
        bpy.context.scene.render.filepath = filename


class Tube(DataHandler):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self, filename=None, default=None,
                 hf_array=None, hf_info_dict=None,
                 **_):
        path = '/home/erik/csgit/trafficability_experiments/runs/data_023_ta_ut_terränger/003_terräng_plant_full/PathToBlender/path.npy'
        kwargs = {
            'image_texture_file': '/home/erik/csgit/trafficability_experiments/runs/data_023_ta_ut_terränger/003_terräng_plant_full/PathToBlender/FuelConsumptionCurrent.png',
        }
        from utils.Blender import setup_tube
        tube_obj = setup_tube(path, **kwargs)
        print(f"tube_obj:{tube_obj}")
        tube_obj.location.z += -4


class Depth(DataHandler):
    ''' '''
    @debug_decorator
    def __call__(self, filename="terrain.npz", default=None,
                 hf_array=None, hf_info_dict=None,
                 overwrite=False,
                 **_):
        from utils.Blender import get_depth, render_eevee
        from utils.Blender import setup_render_z

        setup_render_z()

        render_eevee(save_dir=self.save_dir, filename='test.png')
        dmap = -get_depth()

        # Set lowest point at 0
        dmap = dmap - np.min(dmap)

        # Remove file ending
        filename = filename.replace('.npz', '')
        filename = os.path.join(
            self.save_dir, filename + self.file_id + '.npz')
        if os.path.exists(filename) and not overwrite:
            # Skip if file already exists
            return

        with open(filename, 'wb') as f:
            np.savez(f, heights=dmap, **hf_info_dict)


class Render(DataHandler):
    ''' '''
    @debug_decorator
    def __call__(self, filename='render.png', default=None,
                 overwrite=False,
                 **_):
        from utils.Blender import render_eevee

        filename = default if default is not None else filename
        # Render
        render_eevee(save_dir=self.save_dir, filename=filename)


class Exit(DataHandler):
    ''' '''
    @debug_decorator
    def __call__(self, **_):
        exit(0)


class MakeInterpolator(DataHandler):
    ''' '''
    @debug_decorator
    def __call__(self,
                 hf_array=None, hf_info_dict=None,
                 **_):
        from utils.interpolator import Interpolator

        size = hf_info_dict['size']

        if 'extent' in hf_info_dict:
            extent = hf_info_dict['extent']
        else:
            extent = [-size[0]/2, size[0]/2, -size[1]/2, size[0]/2]

        resolution = hf_array.shape

        # Setup interpolator
        interpolator = Interpolator(hf_array, extent)
        points = interpolator.setup_points(
            size=size, resolution=resolution)

        # Debug
        # # Get heights and grid_points
        # x = 0
        # y = 0
        # yaw = 0
        # heights, grid_points = interpolator.get_height_map((x, y), yaw, points)
        #
        # from utils.plots import plot_image
        # filename = os.path.join(self.save_dir, 'test.png')
        # fig, ax = plot_image(heights)
        # fig.savefig(filename)

        return {
            'interpolator': interpolator,
            }


class AddRocks(DataHandler):
    ''' '''
    @debug_decorator
    def __call__(self,
                 filename='obstacles.npz',
                 hf_array=None, hf_info_dict=None,
                 interpolator=None,
                 default=None,
                 **_):
        # assert interpolator is not None, 'Run MakeInterpolator before'
        import bpy

        if hf_info_dict is not None:
            # TODO: should be tuple!!
            size = hf_info_dict['size']
        else:
            # Workaround
            size = 50

        filename = default if default is not None else filename

        from utils.obstacles import Obstacles
        obstacles = Obstacles.from_numpy(filename)
        height = obstacles.height
        print(f"np.max(height):{np.max(height)}")

        for position, radius, height in obstacles:
            if height > 0.29:
                print(f"position:{position}")
                obj_path = '/home/erik/csgit/artificial_terrains/models/Rocks/agx_data_models/convex_rock2.obj'
                # Import the OBJ file
                bpy.ops.import_scene.obj(filepath=obj_path)
                obj = bpy.context.selected_objects[0]

                # Set the origin to the center of mass
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

                scale_factor = 0.001 * 6 * height
                bpy.ops.transform.resize(value=(
                    scale_factor, scale_factor, scale_factor))

                # TODO: size should be a tuple!!
                x, y = np.subtract(position, size/2)
                if interpolator is not None:
                    z = interpolator(x, y)
                else:
                    z = 0
                obj.location = (x, y, z)
                obj.rotation_euler = np.random.random(3)*np.pi*2
