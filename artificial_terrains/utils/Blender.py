import bpy
import numpy as np
import os


def setup_near_far():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.clip_start = 0.1
                    space.clip_end = 5000   # <-- set far clip here


def remove_object(name='Cube'):
    if name in bpy.data.objects:
        grid_to_remove = bpy.data.objects[name]
        bpy.data.objects.remove(grid_to_remove, do_unlink=True)

        return True
    return False


def get_material(name="GroundMaterial"):
    ''' Create or get a material '''
    if name not in bpy.data.materials:
        material = bpy.data.materials.new(name=name)
    else:
        material = bpy.data.materials.get(name)

    return material


def setup_z_coord_shader(material):
    ''' Setup shader with colormap for z value '''
    # Setup to use nodes
    material.use_nodes = True

    # Get material node tree and nodes
    node_tree = material.node_tree
    nodes = node_tree.nodes

    # Remove all previous nodes
    for node in nodes:
        nodes.remove(node)

    # Dict with shaders to create
    nodes_to_add = {
        'texture': 'ShaderNodeTexCoord',
        'separate': 'ShaderNodeSeparateXYZ',
        # 'multiply': 'ShaderNodeMath',
        # 'maprange': 'ShaderNodeMapRange',
        'colorramp': 'ShaderNodeValToRGB',
        'bsdf': 'ShaderNodeBsdfPrincipled',
        'output': 'ShaderNodeOutputMaterial',
    }

    # Create shaders
    added_nodes = {}
    x_location = -600
    for name, shader in nodes_to_add.items():
        node = nodes.new(shader)
        # Set position
        node.location.x = x_location
        node.location.y = 200
        x_location += 250
        # Add to dict
        added_nodes[name] = node

    # Make shorter names for convenience
    texture_node = added_nodes['texture']
    separate_node = added_nodes['separate']
    # multiply_node = added_nodes['multiply']
    # maprange_node = added_nodes['maprange']
    colorramp_node = added_nodes['colorramp']
    bsdf_node = added_nodes['bsdf']
    output_node = added_nodes['output']

    # Setup some nodes
    # multiply_node.operation = 'MULTIPLY'
    try:
        bsdf_node.inputs['Specular IOR Level'].default_value = 0
    except KeyError:
        bsdf_node.inputs['Specular'].default_value = 0

    # Connect nodes
    node_tree.links.new(texture_node.outputs[0], separate_node.inputs[0])
    node_tree.links.new(separate_node.outputs[2], colorramp_node.inputs[0])
    # node_tree.links.new(multiply_node.outputs[0], maprange_node.inputs[0])
    # node_tree.links.new(maprange_node.outputs[0], colorramp_node.inputs[0])
    node_tree.links.new(colorramp_node.outputs[0], bsdf_node.inputs[0])
    node_tree.links.new(bsdf_node.outputs[0], output_node.inputs[0])


def add_grid(
        material,
        grid_kwargs={},
):
    ''' Insert a mix shader connected to a image node with a grid'''
    # Get material node tree and nodes
    node_tree = material.node_tree
    nodes = node_tree.nodes

    image_node = nodes.new('ShaderNodeTexImage')
    image_node.location = (250, 400)

    mix_shader_node = nodes.new('ShaderNodeMixShader')
    mix_shader_node.location = (500, 400)

    # get output node
    output_node = nodes["Material Output"]

    # Get node which is connected to output.
    for link in node_tree.links:
        if link.to_node == output_node:
            second_to_last_node = link.from_node
            node_tree.links.remove(link)

    # Generate grid
    from utils.plots import generate_line_grid
    arr = generate_line_grid(**grid_kwargs)
    height, width, _ = arr.shape

    # Create empty Blender image
    img = bpy.data.images.new('image_name', width=width, height=height, alpha=True)

    # Flatten RGBA values into 1D list
    flat = arr.flatten()
    img.pixels = flat.tolist()  # assign pixels

    # image_node.image = bpy.data.images.load(grid_file)
    image_node.image = img

    # Make new links
    node_tree.links.new(image_node.outputs[0], mix_shader_node.inputs[0])
    node_tree.links.new(second_to_last_node.outputs[0], mix_shader_node.inputs[2])
    node_tree.links.new(mix_shader_node.outputs[0], output_node.inputs[0])


def use_image_texture(
        material, image_texture_file,
        multiply=False,
):
    ''' Insert a image texture and connect to BSDF

    Args:
      multiply: Insert as multiplyRGB with current setup

    '''
    # Get material node tree and nodes
    node_tree = material.node_tree
    nodes = node_tree.nodes

    image_node = nodes.new('ShaderNodeTexImage')
    image_node.location = (0, 400)

    # get output node
    bsdf_node = nodes["Principled BSDF"]

    # Setup nodes
    image_node.image = bpy.data.images.load(image_texture_file)
    image_node.location = (200, 400)

    if multiply:
        # Get node connected to bsdf
        for link in node_tree.links:
            if link.to_node == bsdf_node:
                second_to_last_node = link.from_node
                node_tree.links.remove(link)

        mix_node = nodes.new('ShaderNodeMixRGB')
        # Setup nodes
        mix_node.inputs[0].default_value = 1.0
        mix_node.blend_type = 'MULTIPLY'
        mix_node.location = (600, 400)

        # Make new links
        node_tree.links.new(image_node.outputs[0], mix_node.inputs[1])
        node_tree.links.new(second_to_last_node.outputs[0], mix_node.inputs[2])

        node_tree.links.new(mix_node.outputs[0], bsdf_node.inputs[0])

    else:
        # Simply connect image node and bsdf node
        node_tree.links.new(image_node.outputs[0], bsdf_node.inputs[0])


def setup_world(hdri='forest.exr', hdri_level=None):
    ''' Setup world node with hdri light '''

    scene = bpy.context.scene
    node_tree = scene.world.node_tree
    nodes = node_tree.nodes

    hdri_list = [
        sl.path for sl in bpy.context.preferences.studio_lights
        if sl.type == 'WORLD']
    hdri_dict = {os.path.basename(hdri): hdri for hdri in hdri_list}

    # Clear all nodes
    nodes.clear()

    # Add Background node
    node_background = nodes.new(type='ShaderNodeBackground')
    # And possibly set the strength level
    if hdri_level is not None:
        node_background.inputs['Strength'].default_value = hdri_level

    # Add Environment Texture node
    node_environment = nodes.new('ShaderNodeTexEnvironment')
    # Load and assign the image to the node property
    hdri_file = hdri_dict[hdri]
    node_environment.image = bpy.data.images.load(hdri_file)
    node_environment.location = (-300, 0)

    # Add Output node
    node_output = nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = (200, 0)

    # Add Texture coordinates node
    node_texture = nodes.new(type='ShaderNodeTexCoord')
    node_texture.location = (-700, 0)

    # Add Mapping node
    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_mapping.location = (-500, 0)

    # Link all nodes
    links = node_tree.links
    links.new(node_texture.outputs[0], node_mapping.inputs[0])
    links.new(node_mapping.outputs[0], node_environment.inputs[0])
    links.new(
        node_environment.outputs["Color"], node_background.inputs["Color"])
    links.new(
        node_background.outputs["Background"], node_output.inputs["Surface"])


def setup_camera(
        resolution_x=1920,
        resolution_y=1080,
        x=0, y=0, z=10,
        rx=0, ry=0, rz=0,
        size=(50, 50),
        camera_name='Camera', camera_type='ORTHO',
):
    '''
    Setup camera to produce array
    '''
    camera = bpy.data.objects[camera_name]

    # Set resolution from array shape
    scene = bpy.data.scenes["Scene"]
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y

    # Set location
    camera.location.x = x
    camera.location.y = y
    camera.location.z = z
    # Set camera rotation in euler angles
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler[0] = rx*(np.pi/180.0)
    camera.rotation_euler[1] = ry*(np.pi/180.0)
    camera.rotation_euler[2] = rz*(np.pi/180.0)
    # Setup camera type
    camera.data.type = camera_type
    camera.data.ortho_scale = max(size)

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.compression = 20

    return camera


def setup_top_camera(
        array, grid_size, size,
        camera_name='Camera', camera_type='ORTHO',
):
    '''
    Setup camera to produce array

    Args:
      size: physical size of camera

    '''
    camera = bpy.data.objects[camera_name]
    z = np.max(array) + 1.0  # due to camera extension?
    z += 10  # Add 10 m margin, since it doesn't matter for 'Depth'

    rx, ry, rz = 0, 0, 0
    x, y = 0, 0

    # Set resolution from array shape
    scene = bpy.data.scenes["Scene"]
    scene.render.resolution_x = grid_size[0]
    scene.render.resolution_y = grid_size[1]

    # Set location
    camera.location.x = x
    camera.location.y = y
    camera.location.z = z
    # Set camera rotation in euler angles
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler[0] = rx*(np.pi/180.0)
    camera.rotation_euler[1] = ry*(np.pi/180.0)
    camera.rotation_euler[2] = rz*(np.pi/180.0)
    # Setup camera type
    camera.data.type = camera_type
    camera.data.ortho_scale = max(size)

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.compression = 20

    return camera


def setup_angled_camera(
        angle=45,
        center=[0, 0, 0],
        distance=None,
        view_angle=50,
        camera_name='Camera',
):
    if distance is None:
        distance = 150  # replace by auto calc

    x = center[0] + 0
    y = center[1] - np.sin(np.deg2rad(angle)) * distance
    z = center[2] + np.cos(np.deg2rad(angle)) * distance

    rx = angle
    ry, rz = 0, 0

    camera = bpy.data.objects[camera_name]
    # Set location
    camera.location.x = x
    camera.location.y = y
    camera.location.z = z

    # Set camera rotation in euler angles
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler[0] = rx*(np.pi/180.0)
    camera.rotation_euler[1] = ry*(np.pi/180.0)
    camera.rotation_euler[2] = rz*(np.pi/180.0)

    camera.data.lens_unit = 'FOV'
    camera.data.angle = np.deg2rad(view_angle)
    camera.data.clip_end = 500

    return camera


def set_view_to_camera():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            break


def add_holdout_plane(array, name='Plane'):
    ''' Add holdout plane '''

    # Remove any object called <name> to avoid duplicate complications
    if name in bpy.data.objects:
        obj_to_remove = bpy.data.objects[name]
        bpy.data.objects.remove(obj_to_remove, do_unlink=True)

    # Add plane
    bpy.ops.mesh.primitive_plane_add(
        size=2000,
        align='WORLD',
        location=(0, 0, -100),
        rotation=(0, 0, 0),
    )
    # Create material
    mat = bpy.data.materials.get("HoldoutMaterial")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="HoldoutMaterial")

    # Assign it to object
    ob = bpy.data.objects.get("Plane")
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = mat
    else:
        # no slots
        ob.data.materials.append(mat)

    setup_holdout_material(mat)


def set_object_material(obj, material):
    ''' Set material on object'''
    # Get object by name if obj is string
    if type(obj) == str:
        pass
    else:
        print("type(obj):{}".format(type(obj)))

    if obj.data.materials:
        # assign to 1st material slot
        obj.data.materials[0] = material
    else:
        # no slots
        obj.data.materials.append(material)


def setup_holdout_material(material):
    # Setup to use nodes
    material.use_nodes = True

    # Get material node tree and nodes
    node_tree = material.node_tree
    nodes = node_tree.nodes

    # Remove all previous nodes
    for node in nodes:
        nodes.remove(node)

    # Create nodes
    holdout_node = nodes.new('ShaderNodeHoldout')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    holdout_node.location = (-300, 0)
    output_node.location = (0, 0)

    # Link nodes
    links = node_tree.links
    links.new(holdout_node.outputs[0], output_node.inputs[0])


def setup_bsdf(material,
               specular=0.0,
               roughness=1.0,
               ):
    node_tree = material.node_tree
    nodes = node_tree.nodes

    bsdf_node = nodes["Principled BSDF"]
    bsdf_node.inputs['Specular'].default_value = specular
    bsdf_node.inputs['Roughness'].default_value = roughness


def hide_light():
    ob = bpy.data.objects.get("Light")
    ob.hide_render = True
    ob.hide_viewport = True


def setup_lights(
        use_hdri=False,
        hdri_level=1.0,
        level=None,
        sun_position=(0, 0, 70),
        sun_angle=55,
):
    setup_sun(*sun_position, sun_angle)

    if use_hdri:
        setup_world(hdri='forest.exr', hdri_level=hdri_level)
        hide_light()

    if level is not None:
        # Explicitly set sun level
        ob = bpy.data.objects.get("Light")
        ob.hide_render = False
        ob.hide_viewport = False
        ob.data.energy = level


def setup_sun(x=0, y=0, z=70, angle=55):
    # Place sun
    light = bpy.data.objects['Light']
    light.location.x = x/2
    light.location.y = y
    light.location.z = 50
    light.data.type = 'SUN'
    light.data.energy = 10.0
    # Rotate sun
    (rx, ry, rz) = (angle, 0.0, 180.0)
    light.rotation_mode = 'XYZ'
    light.rotation_euler[0] = rx*(np.pi/180.0)
    light.rotation_euler[1] = ry*(np.pi/180.0)
    light.rotation_euler[2] = rz*(np.pi/180.0)


def colormap_to_colorramp(colorramp, colormap, gamma=2.2):
    '''
    Apply a matplotlib colormap to a blender colorramp

    bpy.types.ColorRamp
    matplotlib.colors.LinearSegmentedColormap
    '''
    # import matplotlib.cm as cm
    # from matplotlib.colors import rgb2hex

    cr = colorramp

    N = 32
    # Add up to 32 elements in colorramp
    for i in range(N-len(cr.elements)):
        cr.elements.new(i/N)

    a = np.arange(N)/(N-1)
    # colormap = cm.bone
    colors = colormap(a)
    # Make list of hex
    # hexs = [rgb2hex(colors[i, :]) for i in range(colors.shape[0])]

    # Test to 'invert gamma'
    if gamma is not None:
        colors = np.power(colors, gamma)

    # Get and setup colorramp
    # Setup colorramp
    for i, e in enumerate(cr.elements):
        e.position = a[i]
        new_color = (colors[i, 0], colors[i, 1], colors[i, 2], 0)
        e.color = new_color


def grid_from_array(array, size, center=(0, 0), name='Grid', **kwargs):
    '''
    np array of 2 dim
    size: [x, y] size in m
    '''
    import bmesh

    # Remove any object called <name> to avoid duplicate complications
    if name in bpy.data.objects:
        grid_to_remove = bpy.data.objects[name]
        bpy.data.objects.remove(grid_to_remove, do_unlink=True)

    # Create blender mesh
    rows, cols = array.shape
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=rows-1, y_subdivisions=cols-1)

    # obj = bpy.data.objects[name]
    obj = bpy.context.object
    obj.dimensions = (size[0], size[1], 0)
    # mesh = obj.data
    obj.location.x = center[0]
    obj.location.y = center[1]
    obj.name = name

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.app.debug = True

    context = bpy.context
    bm = bmesh.from_edit_mesh(context.edit_object.data)
    # deselect all
    for v in bm.verts:
        v.select = False

    bm.verts.ensure_lookup_table()
    for x in range(rows):
        for y in range(cols):
            bm.verts[rows*y + x].co.z = array[x, y]

    bpy.ops.object.mode_set(mode="OBJECT")

    return obj


def get_depth():
    """Obtains depth map from Blender render.
    :return: The depth map of the rendered camera view as a numpy array of size (H,W).
    """
    for image in bpy.data.images:
        print(f"image:{image}")
    z = bpy.data.images['Viewer Node']
    # z = bpy.data.images['Render Result']
    w, h = z.size
    dmap = np.array(z.pixels[:], dtype=np.float32)  # convert to numpy array
    dmap = np.reshape(dmap, (h, w, 4))[:, :, 0]

    # Transpose first two axes
    axes = np.arange(dmap.ndim)
    axes[:2] = [1, 0]
    dmap = np.transpose(dmap, axes=axes)

    return dmap


def render(
        render_engine='CYCLES',
        cycles_samples=128,
        save_blender=False,
        filename=None,
        save_dir=None,
):
    bpy.context.scene.render.engine = render_engine
    bpy.context.scene.cycles.samples = cycles_samples

    if filename is not None:
        filename = os.path.join(save_dir, filename)

    bpy.context.scene.render.filepath = filename
    bpy.ops.render.render(write_still=True)


def render_eevee(*args, **kwargs):
    return render(render_engine='BLENDER_EEVEE', *args, **kwargs)


def setup_render_z():
    # Setup scene to use nodes
    bpy.context.scene.use_nodes = True
    # Enable render z-pass
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    # Get node tree for the compositor
    tree = bpy.context.scene.node_tree

    # Prepare nodes
    node_link_list = (
        ('CompositorNodeRLayers', None, 'Depth'),
        # ('CompositorNodeNormalize', 'Value', 'Value'),
        # ('CompositorNodeComposite', 'Image', None)
        ('CompositorNodeViewer', 'Image', None)
        )

    setup_nodes(tree, node_link_list)

    # Add another branch
    node_link_list = (
        ('CompositorNodeRLayers', None, 'Depth'),
        ('CompositorNodeNormalize', 'Value', 'Value'),
        ('CompositorNodeComposite', 'Image', None)
        )
    add_nodes_branch(tree, node_link_list, y=100)


def setup_segmentation_render():
    # Connect Render Layers to 'Viewer'
    # We can not get image from 'Render Result', only 'Viewer Layer'

    # Setup scene to use nodes
    bpy.context.scene.use_nodes = True
    # Get node tree for the compositor
    tree = bpy.context.scene.node_tree

    # Prepare nodes
    node_link_list = (
        ('CompositorNodeRLayers', None, 'Image'),
        ('CompositorNodeViewer', 'Image', None)
        )
    setup_nodes(tree, node_link_list)

    # Add another branch
    node_link_list = (
        ('CompositorNodeRLayers', None, 'Image'),
        ('CompositorNodeComposite', 'Image', None)
        )
    add_nodes_branch(tree, node_link_list, y=100)

    # Set color management to use raw linear colors
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.view_settings.view_transform = 'Raw'

    # Loop through all objects in the current scene, and color
    for obj in bpy.data.objects:
        # Check if 'Ground' or 'Rock' is in the object's name
        if "Ground" in obj.name:
            set_emission_material(obj, [0, 0, 0])
        if "Rock" in obj.name:
            set_emission_material(obj, [1, 1, 1])


def set_emission_material(obj, color):
    # Ensure the object is of type 'MESH'
    if obj.type == 'MESH':
        # Remove all existing materials from the object
        obj.data.materials.clear()

        # Create a new material
        mat = bpy.data.materials.new(name="EmissionMaterial")

        # Use nodes for the material
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        # Clear default nodes
        nodes.clear()

        # Create Emission node
        emission_node = nodes.new(type='ShaderNodeEmission')
        emission_node.location = (0, 0)
        emission_node.inputs[0].default_value = (color[0], color[1], color[2], 1)  # RGB and Alpha

        # Create Material Output node
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (200, 0)

        # Link Emission node to Material Output
        links = mat.node_tree.links
        links.new(emission_node.outputs[0], output_node.inputs[0])

        # Assign material to object
        obj.data.materials.append(mat)
    else:
        print("The function is only applicable to mesh objects.")


def setup_nodes(
        node_tree,
        node_link_list,
        separation_x=50,
        remove_old=True,
):
    '''
    Create nodes and links from a list of node-link info

    node_link_list = (
        ('CompositorNodeRLayers', None, 'Depth'),
        ('CompositorNodeNormalize', 'Value', 'Value'),
        ('CompositorNodeComposite', 'Image', None)
        )
    '''
    if remove_old:
        # Remove old nodes
        for node in node_tree.nodes:
            node_tree.nodes.remove(node)
            print(f"Remove:{node}")

    previous_output = None
    accumulated_width = 0

    # Loop and create nodes and links
    for node_name, input_name, output_name in node_link_list:
        new_node = node_tree.nodes.new(node_name)
        print(f"Add:{new_node}")

        # Place node
        new_node.location = (accumulated_width, 0)
        accumulated_width += new_node.width + separation_x

        # Link to previous output, if it exists
        if previous_output is not None:
            node_tree.links.new(previous_output, new_node.inputs[input_name])
        # Store output, if it exist
        if output_name is not None:
            previous_output = new_node.outputs[output_name]


def add_nodes_branch(
        node_tree,
        node_link_list,
        separation_x=50,
        y=0
):
    # Find first node-type in node_link_list in the node tree
    node_name, input_name, output_name = node_link_list[0]
    for i, node in enumerate(node_tree.nodes):
        if node_name == type(node).__name__:
            break

    # Setup location and the previous output
    accumulated_width = node.location[0] + node.width + separation_x
    previous_output = node.outputs[output_name]

    # Loop and create nodes and links
    # (skipping the first entry, as this already exists)
    for node_name, input_name, output_name in node_link_list[1:]:
        new_node = node_tree.nodes.new(node_name)
        print(f"Add:{new_node}")

        # Place node
        new_node.location = (accumulated_width, y)
        accumulated_width += new_node.width + separation_x

        # Link to previous output, if it exists
        if previous_output is not None:
            node_tree.links.new(previous_output, new_node.inputs[input_name])
        # Store output, if it exist
        if output_name is not None:
            previous_output = new_node.outputs[output_name]


def camera_to_follow_cube(cube):
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    # Rename the camera object if desired
    camera.name = "Camera_cube"

    camera.data.lens = 12
    camera.location = cube.location
    camera.rotation_euler = (np.deg2rad(55), 0, 0)
    # camera.location.z += 2

    # Hide cube
    cube.hide_viewport = True
    cube.hide_render = True

    # Parent camera to cube
    camera.parent = cube
    camera.parent_type = 'OBJECT'

    bpy.context.scene.camera = camera


def camera_to_follow_path():
    # Add a camera object to the scene
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    # Rename the camera object if desired
    camera.name = "MyCamera"

    # Set the camera's properties
    camera.data.type = 'PERSP'  # Perspective camera
    camera.data.angle = 0.8  # Field of view angle (in radians)
    camera.data.lens = 50  # Focal length of the lens

    # Set up the Follow Path constraint for the camera
    camera.constraints.new(type='FOLLOW_PATH')
    follow_path_constraint = camera.constraints['Follow Path']
    path = bpy.data.objects['Path']
    follow_path_constraint.target = path
    follow_path_constraint.up_axis = 'UP_Z'
    # follow_path_constraint.use_fixed_location = True
    # follow_path_constraint.use_fixed_rotation = True
    bpy.ops.constraint.followpath_path_animate(
        constraint="Follow Path", owner='OBJECT')

    # Create a cube object
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    cube = bpy.context.object
    cube.constraints.new(type='FOLLOW_PATH')
    cube_path_constraint = cube.constraints['Follow Path']
    cube_path_constraint.offset = -20
    cube_path_constraint.target = path

    # bpy.ops.object.constraint_add(type='TRACK_TO')
    # bpy.context.object.constraints["Track To"].target = bpy.data.objects["Cube"]
    camera.constraints.new(type='TRACK_TO')
    track_to = camera.constraints['Track To']
    track_to.target = cube

    path.data.twist_mode = 'Z_UP'
    path.data.path_duration = 1000


def image_from_terrain(terrain, name="TerrainHeight"):
    """
    Create a Blender IMAGE datablock from terrain.array, as a float buffer.

    terrain.array is assumed to be 2D (nx, ny).
    We use a grayscale image: R=G=B=normalized height, A=1.
    """
    arr = np.asarray(terrain.array, dtype=np.float32)

    # Optional: match your imshow behavior (you used terrain.array.T there)
    # If your orientation looks wrong in Blender, try arr = arr.T or np.flipud/np.fliplr.
    # arr = arr.T  # to be consistent with your plot_terrain
    arr = np.flip(arr, axis=0)

    h, w = arr.shape  # rows, cols

    # Normalize to 0..1 (or keep raw; for now we normalize for convenience)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax > vmin:
        norm = (arr - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(arr, dtype=np.float32)

    # Remove old image with same name if it exists
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name], do_unlink=True)

    img = bpy.data.images.new(
        name=name,
        width=w,
        height=h,
        alpha=True,
        float_buffer=True,   # 32-bit float per channel
    )

    # Blender expects flat [R,G,B,A, R,G,B,A, ...] in row-major order
    pixels = np.zeros((h * w, 4), dtype=np.float32)
    g = norm.ravel()
    pixels[:, 0] = g  # R
    pixels[:, 1] = g  # G
    pixels[:, 2] = g  # B
    pixels[:, 3] = 1.0  # A

    img.pixels = pixels.ravel().tolist()
    img.update()

    # Return image and also the original min/max so you can map “strength” if needed
    return img, vmin, vmax


def create_displaced_ground_from_terrain(
    terrain,
    base_cuts=128,        # resolution of manual grid
    simple_levels=2,      # extra Simple subdiv levels (0..6)
    name="Terrain",
    **kwargs,
):
    """
    Create a Blender terrain object from a terrain object
    (with .array and .extent), using a float image + Displace.

    - base_cuts: number of quads per side in the base grid
    - simple_levels: extra Simple subdivision (0–6)
    - strength: displacement strength in Blender units
    """

    strength = np.max(terrain.array) - np.min(terrain.array)

    # Build float image directly from terrain.array
    img_name = f"{name}_height"
    img, vmin, vmax = image_from_terrain(terrain, name=img_name)

    # Remove existing object with this name
    if name in bpy.data.objects:
        obj_old = bpy.data.objects[name]
        bpy.data.objects.remove(obj_old, do_unlink=True)

    # --- Create a grid base mesh ---
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=base_cuts,
        y_subdivisions=base_cuts,
        size=1.0,
    )
    obj = bpy.context.object
    obj.name = name

    # Scale to world size
    obj.scale.x = terrain.size[0]
    obj.scale.y = terrain.size[1]

    bpy.context.view_layer.objects.active = obj

    # Basic UVs (grid usually already has UVs, but we ensure something exists)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')

    # --- Optional Simple subdiv modifier ---
    if simple_levels > 0:
        subsurf = obj.modifiers.new(name="SubdivSimple", type='SUBSURF')
        subsurf.subdivision_type = 'SIMPLE'
        subsurf.levels = min(simple_levels, 6)
        subsurf.render_levels = min(simple_levels, 6)

    # --- Create texture from image for Displace ---
    tex = bpy.data.textures.new(name=f"{name}_height_tex", type='IMAGE')
    tex.image = img

    disp = obj.modifiers.new(name="Displace", type='DISPLACE')
    disp.texture = tex
    disp.texture_coords = 'UV'
    disp.strength = strength
    disp.mid_level = 0.0  # since we normalized to 0..1; tweak as needed

    # Apply object scale so displacement is in world units cleanly
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    return obj
