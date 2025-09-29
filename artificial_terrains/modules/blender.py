from modules.data import Module, debug_decorator
import os
import sys
import numpy as np
from utils.utils import get_terrains, get_terrain

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


class Ground(Module):
    ''' Print array info '''
    create_folder = False

    @debug_decorator
    def __call__(self, filename=None, default=None,
                 terrain_temp=[], terrain_prim=[],
                 ground_material=None, grid=True,
                 last=None,
                 **_):
        from utils.Blender import grid_from_array
        import hashlib

        filename = default if default is not None else filename
        if filename is not None:
            from utils.terrains import Terrain
            # Load data
            terrain = Terrain.from_numpy(filename)
            terrains = [terrain]
        else:
            # Get latest from dict/heap
            # terrain = get_terrain(terrain_temp, terrain_prim, remove=False)
            terrains = get_terrains(
                terrain_temp, terrain_prim, last, remove=False)

        # Make grid from array
        for i, terrain in enumerate(terrains):
            # Generate a short hash from the terrain array
            array_bytes = terrain.array.tobytes()
            hash_digest = hashlib.sha1(array_bytes).hexdigest()[:6]  # Short hash
            name = f"Ground{i:05d}{self.file_id}_{hash_digest}"

            grid_obj = grid_from_array(
                terrain.array, name=name, size=terrain.size,
                center=terrain.position)

            if ground_material is not None:
                # Add grid image
                if grid:
                    from utils.Blender import add_grid

                    min_resolution = 2048
                    # Resolution proportional to terrain size (10 px per unit)
                    res_from_terrain = int(terrain.size[0] * 10), int(terrain.size[1] * 10)

                    # Final resolution = take maximum in each dimension
                    resolution = (max(min_resolution, res_from_terrain[0]),
                                  max(min_resolution, res_from_terrain[1]))

                    grid_kwargs = {
                        'extent': terrain.extent,
                        'size': resolution,
                        # 'filename': 'test.png',
                    }
                    add_grid(ground_material, grid_kwargs=grid_kwargs)
                grid_obj.data.materials.append(ground_material)

        # This line has not been adopted for possibly multiple objs.
        return {'grid_obj': grid_obj}


class BasicSetup(Module):
    '''
    Basic setup of blender scene. Remove cube and add hrdi lights
    '''
    create_folder = False

    @debug_decorator
    def __call__(self, default=None, **_):
        import bpy
        from utils.Blender import (
            get_material, setup_z_coord_shader, setup_lights)
        # Remove Cube
        from utils.Blender import remove_object
        remove_object(name='Cube')

        # Create Groundmaterial
        ground_material = get_material(name='GroundMaterial')
        ground_material.use_nodes = True

        # Setup groundmaterial
        setup_z_coord_shader(ground_material)

        # Setup lights
        sun_position = (0, 70, 70)

        use_hdri = True
        light_kwargs = {'level': 0}
        setup_lights(use_hdri, sun_position=sun_position, **light_kwargs)

        # Set standard view transform to not distort colors
        bpy.context.scene.view_settings.view_transform = 'Standard'

        return {
            'ground_material': ground_material,
            }


class ClearScene(Module):
    """
    Removes all objects from the current Blender scene.
    """
    create_folder = False

    @debug_decorator
    def __call__(self, **_):
        import bpy

        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # Find and delete objects with names starting with "Ground"
        for obj in bpy.data.objects:
            if obj.name.startswith("Ground"):
                obj.select_set(True)

        # Delete selected objects
        bpy.ops.object.delete()

        # Optionally clear unused data blocks (meshes, materials, etc.)
        # This helps free memory and avoid clutter
        # for data_collection in [bpy.data.meshes, bpy.data.materials, bpy.data.images,
        #                         bpy.data.cameras, bpy.data.lights, bpy.data.curves]:
        for data_collection in [bpy.data.meshes]:
            for datablock in data_collection:
                if datablock.users == 0:
                    data_collection.remove(datablock)

        return {'cleared': True}


class ImageTexture(Module):
    ''' Add image texture to ground'''
    create_folder = False

    @debug_decorator
    def __call__(self, ground_material=None, texture=None, default=None, **_):
        from utils.Blender import (use_image_texture)

        texture = default if default is not None else texture
        image_texture_kwargs = {}
        use_image_texture(ground_material, texture, **image_texture_kwargs)


class ColorMap(Module):
    ''' Add image colormap'''
    create_folder = False

    @debug_decorator
    def __call__(self, ground_material=None, cmap_name='viridis', default=None, **_):
        from utils.Blender import (colormap_to_colorramp)
        import colorcet as cc
        import matplotlib.cm as cm
        cmap_name = default if default is not None else cmap_name

        if default is not None:
            # Use colorcet as default
            try:
                cmap = cc.cm[cmap_name]
            except KeyError:
                cmap = cm.get_cmap(cmap_name)

        try:
            # blender 4.x
            cr = ground_material.node_tree.nodes['Color Ramp'].color_ramp
        except KeyError:
            # blender 3.x
            cr = ground_material.node_tree.nodes['ColorRamp'].color_ramp

        colormap_to_colorramp(cr, cmap)


class GenericCamera(Module):
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


class Camera(Module):
    ''' Setup camera'''
    create_folder = False

    @debug_decorator
    def __call__(self, camera='top', default=None,
                 terrain_temp=[], terrain_prim=[],
                 size=None,
                 resolution=None,
                 extent=None,
                 grid_size=None,
                 angle=45,
                 center=None,
                 distance=None,
                 **_):
        from utils.artificial_shapes import determine_extent_and_resolution

        extent, (N_x, N_y) = determine_extent_and_resolution(
            resolution, size, grid_size, extent)

        from utils.Blender import (
            setup_top_camera, set_view_to_camera, setup_angled_camera)

        if isinstance(default, str):
            camera = default if default is not None else camera
            assert camera in ['top', 'angled']
        elif isinstance(default, (int, float)):
            camera = 'angled'
            angle = default

        # Get latest terrain from dict/heap
        terrain = get_terrain(terrain_temp, terrain_prim, remove=False)

        size = terrain.size
        camera_kwargs = {}
        # Setup camera
        if camera == 'top':
            camera = setup_top_camera(terrain.array, (N_x, N_y), size)
        elif camera == 'angled':
            if center is None:
                middle_z = terrain.array[int(terrain.array.shape[0]/2), int(terrain.array.shape[1]/2)]
                center = [0, 0, middle_z]
            if distance is None:
                distance = size[0]*2
            camera = setup_angled_camera(
                angle=angle,
                center=center, distance=distance, **camera_kwargs)
        set_view_to_camera()


class AnimateCube(Module):
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


class Tube(Module):
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


class Depth(Module):
    ''' '''
    @debug_decorator
    def __call__(self, filename="terrain.npz", default=None,
                 folder='Depth',
                 terrain_temp=[], terrain_prim=[],
                 overwrite=False,
                 **_):
        from utils.Blender import get_depth, render_eevee
        from utils.Blender import setup_render_z

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

        # Get latest terrain from dict/heap
        terrain = get_terrain(terrain_temp, terrain_prim)

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

        # TODO: Fix this. Unsure how to handle this when I remove
        # the 'terrain' attribute
        terrain_prim = [terrain]

        return {
            'terrain_prim': terrain_prim,
        }


class Render(Module):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self, filename=None, default=None,
                 folder='Render',
                 overwrite=False,
                 loop_id=None,
                 **_):
        from utils.Blender import render_eevee

        if filename is None:
            filename = f'render{loop_id}.png'

        # Possibly set folder from 'default'.
        if default is not None and '.' in default:
            filename = default
        elif default is not None:
            folder = default

        # Use folder
        basename = os.path.basename(self.save_dir)
        if basename != folder:
            dirname = os.path.dirname(self.save_dir)
            self.save_dir = os.path.join(dirname, folder)

        # Create folder if needed
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # Render
        render_eevee(save_dir=self.save_dir, filename=filename)


class RenderSegmentation(Module):
    '''
    Uniform color for 'Rock's and 'Ground'. Render and get array
    '''
    @debug_decorator
    def __call__(self, filename='render.png', default=None,
                 overwrite=False, folder='RenderSegmentation',
                 **_):
        from utils.Blender import render_eevee, get_depth
        from utils.Blender import setup_segmentation_render

        # Possibly set folder from 'default'.
        if default is not None and '.' in default:
            filename = default
        elif default is not None:
            folder = default

        # Use folder
        basename = os.path.basename(self.save_dir)
        if basename != folder:
            dirname = os.path.dirname(self.save_dir)
            self.save_dir = os.path.join(dirname, folder)

        # Create folder if needed
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        setup_segmentation_render()

        # Render. This can be skipped
        render_eevee(save_dir=self.save_dir, filename=filename)

        # We can use the 'get_depth' function to get the
        # 'segmentation-image' from the 'Viewer Node'
        array = get_depth()

        return {'segmentation_array': array}


class Exit(Module):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self, **_):
        exit(0)


class Holdout(Module):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self,
                 terrain_temp=[], terrain_prim=[],
                 default=None,
                 **_):
        import bpy
        bpy.context.scene.render.film_transparent = True


class AddMeshObjects(Module):
    ''' '''
    create_folder = False

    @debug_decorator
    def __call__(self,
                 filename='obstacles.npz',
                 position=[[-10, 10], [10, 10], [10, -10], [-10, -10]],
                 width=[10, 5, 3, 1],
                 yaw_deg=[0, 90, 180, 270],
                 pitch_deg=[0, 10, 20, 30],
                 terrain_temp=[], terrain_prim=[],
                 size=None,
                 default=None,
                 **_):
        # Get latest terrain from dict/heap
        terrain = get_terrain(terrain_temp, terrain_prim, remove=False)

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
                    'assets/Rocks/agx_data_models/convex_rock2.obj',
                    # 'assets/Rocks/agx_data_models/convex_rock3.obj',
                    # 'assets/Rocks/agx_data_models/convex_rock4.obj',
                    # 'assets/Rocks/agx_data_models/convex_rock5.obj',
                    # 'assets/Rocks/agx_data_models/convex_rock6.obj',
                ]

                import random
                obj_path = random.choice(rocks)
                # Import the OBJ file
                try:
                    # blender 4.x
                    bpy.ops.wm.obj_import(filepath=obj_path)
                except AttributeError:
                    #  blender 3.0
                    bpy.ops.import_scene.obj(filepath=obj_path)

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
