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


class Ground(DataHandler):
    ''' Print array info '''
    create_folder = False

    @debug_decorator
    def __call__(self, filename=None, default=None,
                 call_number=None, call_total=None,
                 ground_material=None, **pipe):
        filename = default if default is not None else filename
        if filename is not None:
            from utils.terrains import Terrain
            # Load data
            terrain = Terrain.from_numpy(filename)
            # And add to pipe
            pipe['terrain'] = terrain
        elif 'terrain' in pipe:
            terrain = pipe['terrain']
        elif 'terrain_dict' in pipe and len(pipe['terrain_dict']) > 0:
            # Pick last from terrain-dict
            terrain = list(pipe['terrain_dict'].values())[-1]
        else:
            print('No terrain')
            exit(0)

        # Make grid from array
        from utils.Blender import grid_from_array

        name = 'Ground' + self.file_id
        grid_obj = grid_from_array(terrain.array, name=name, size=terrain.size)

        if ground_material is not None:
            grid_obj.data.materials.append(ground_material)

        pipe['grid_obj'] = grid_obj,

        return pipe


class BasicSetup(DataHandler):
    '''
    Basic setup of blender scene. Remove cube and add hrdi lights
    '''
    create_folder = False

    @debug_decorator
    def __call__(self, default=None, **_):
        from utils.Blender import (
            get_material, setup_z_coord_shader, add_grid, setup_lights)
        # Remove Cube
        from utils.Blender import remove_object
        remove_object(name='Cube')

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
        sun_position = (0, 70, 70)

        use_hdri = True
        light_kwargs = {'level': 10}
        setup_lights(use_hdri, sun_position=sun_position, **light_kwargs)

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
                 terrain=None,
                 size=None,
                 ppm=None,
                 resolution=None,
                 **_):
        from utils.artificial_shapes import determine_size_and_resolution
        size_x, size_y, N_x, N_y = determine_size_and_resolution(
            ppm, size, resolution)

        from utils.Blender import (
            setup_top_camera, set_view_to_camera, setup_angled_camera)

        camera = default if default is not None else camera
        assert camera in ['top', 'angled']
        assert terrain is not None, 'Run LoadTerrain before'

        size = terrain.size
        camera_kwargs = {}
        # Setup camera
        if camera == 'top':
            camera = setup_top_camera(terrain.array, (N_x, N_y), size)
        elif camera == 'angled':
            middle_z = terrain.array[int(terrain.array.shape[0]/2), int(terrain.array.shape[1]/2)]
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
                 terrain=None,
                 overwrite=False,
                 **_):
        from utils.Blender import get_depth, render_eevee
        from utils.Blender import setup_render_z

        setup_render_z()

        # (The render must not be done in order to retrieve the array)
        render_eevee(save_dir=self.save_dir, filename='test.png')
        dmap = -get_depth()

        # Set lowest point at 0
        dmap = dmap - np.min(dmap)

        from utils.terrains import Terrain
        terrain = Terrain.from_array(
            dmap,
            size=terrain.size,  # TODO: take from camera instead
            extent=terrain.extent,  # TODO: take from camera instead
        )

        return {
            'terrain': terrain,
        }


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


class RenderSegmentation(DataHandler):
    '''
    Uniform color for 'Rock's and 'Ground'. Render and get array
    '''
    @debug_decorator
    def __call__(self, filename='render.png', default=None,
                 overwrite=False,
                 **_):
        from utils.Blender import render_eevee, get_depth
        from utils.Blender import setup_segmentation_render
        setup_segmentation_render()

        # Render. This can be skipped
        render_eevee(save_dir=self.save_dir, filename=filename)

        # We can use the 'get_depth' function to get the
        # 'segmentation-image' from the 'Viewer Node'
        array = get_depth()

        return {'segmentation_array': array}


class Exit(DataHandler):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self, **_):
        exit(0)


class Holdout(DataHandler):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self,
                 terrain=None,
                 default=None,
                 **_):
        from utils.Blender import add_holdout_plane
        add_holdout_plane(terrain.array)


class AddRocks(DataHandler):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self,
                 filename='obstacles.npz',
                 position=[[-10, 10], [10, 10], [10, -10], [-10, -10]],
                 width=[10, 5, 3, 1],
                 yaw_deg=[0, 90, 180, 270],
                 pitch_deg=[0, 10, 20, 30],
                 terrain=None,
                 size=None,
                 default=None,
                 **_):
        # Construct 'interpolator'
        if terrain is not None:
            from utils.interpolator import Interpolator
            interpolator = Interpolator(terrain.array, terrain.extent)
        else:
            interpolator = None

        import bpy
        self.position = position
        self.width = width
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg

        for position, width, yaw_deg, pitch_deg in zip(
                self.position, self.width, self.yaw_deg, self.pitch_deg):
            if width > 0.29:
                print(f"position:{position}")
                rocks = [
                    'models/Rocks/agx_data_models/convex_rock2.obj',
                    # 'models/Rocks/agx_data_models/convex_rock3.obj',
                    # 'models/Rocks/agx_data_models/convex_rock4.obj',
                    # 'models/Rocks/agx_data_models/convex_rock5.obj',
                    # 'models/Rocks/agx_data_models/convex_rock6.obj',
                ]

                import random
                obj_path = random.choice(rocks)
                # Import the OBJ file
                bpy.ops.wm.obj_import(filepath=obj_path)

                obj = bpy.context.selected_objects[0]

                # Set the origin to the center of mass
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

                scale_factor = 0.001 * width
                bpy.ops.transform.resize(value=(
                    scale_factor, scale_factor, scale_factor))

                # TODO: size should be a tuple!!
                # x, y = np.subtract(position, size/2)
                x, y = position
                if interpolator is not None:
                    z = interpolator(x, y)
                else:
                    z = 0
                obj.location = (x, y, z)
                # obj.rotation_euler = np.random.random(3)*np.pi*2
                obj.rotation_euler = [np.deg2rad(pitch_deg), 0, np.deg2rad(yaw_deg)]
                obj.name = 'Rock'
