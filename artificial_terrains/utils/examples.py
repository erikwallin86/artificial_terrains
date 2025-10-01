# Flat ground
FLAT = [
    ('Plane', {}),
]


# Flat ground
ANGLED_PLANE = [
    ('Plane', {'pitch_deg': 10}),
]


# Gaussian hill, fully parameterized
GAUSSIAN_HILL = [
    ('Gaussian', {
        'position': [5, 5],
        'height': 2.5,
        'width': 5,
        'aspect': 1.5,
        'yaw_deg': 45,
        'pitch_deg': None,
    }),
]


# One way to get two Gaussian hills
GAUSSIAN_HILL_2 = [
    ('Gaussian', {
        'position': [[5, 5], [-5, 5]],
        'height': [2.5, 1.5],
        'width': 5,
        'aspect': 1.5,
        'yaw_deg': 45,
        'pitch_deg': None,
    }),
    ('Combine', {})
]

# Three random Gaussian hills
GAUSSIAN_HILL_RANDOM_3 = [
    ('Random', 3),
    ('Gaussian', {}),
    ('Combine', {})
]


# Large donut
DONUT_HILL = [
    ('Donut', {'width': 20}),
]


# Sample gaussian hills, along a circle
GAUSSIAN_HILLS_WITH_SAMPLED_LOCATIONS = [
    ('Donut', {'width': 20}),
    ('AsProbability', {}),  # Use donut as distribution,
    ('Random', 5),  # when sampling positions
    ('Gaussian', {'width': [2, 2, 2, 2, 2],  # We override the widths and heights
                  'height': [1, 1, 1, 1, 1]}),
    ('Combine', 'Max')  # combine with 'max', such that hills don't add to each other
]


# Mountain in horizon
MOUNTANS_IN_HORIZON = [
    ('Basic', 10),
    ('Add', 1.5),  # Get noise in [1.5, 2.5]
    ('Scale', 0.5),
    ('Donut', {'width': 30}),
    ('Combine', 'Prod')
]


# Explicit function of x
SINUS_X = [
    ('Function', 'np.sin(2*np.pi*x/20.0)'),
]


# Explicit function of r=np.sqrt(x^2+y^2)
SINUS_R = [
    ('Function', 'np.sin(2*np.pi*r/10.0)'),
]

# Explicit function of x and y
FUNCTION_XY = [
    ('Function', '5*(x/np.max(x))**2+np.sin(y/2)'),
]

# Rocks of different size
ROCKS = [
    ('Rocks', {}),
    ('Combine', {}),
]


# Rocks, unevenly scattered
SCATTERED_ROCKS = [
    ('Basic', 10),
    ('Scale', 10),
    ('Clip', {}),
    ('AsFactor', {}),
    ('Rocks', {}),
    ('Combine', {}),
    ('Scale', {}),
]


# Toy lunar surface example
LUNAR = [
    ('SetDistribution', 'height=uniform(0,0.5)'),
    ('SetDistribution', 'width=uniform(5,10)'),
    ('Random', 10),
    ('Donut', {}),
    ('Combine', 'Max'),
    ('Sphere', {'width': 500}),
    ('ToPrimary', {}),
    ('Combine', {}),
]


# Save to folder
SAVE = [
    ('Basic', 15),
    ('Save', 'Terrains/basic'),
]

# Load from folder
LOAD = [
    ('Load', 'Terrains/basic'),
]


# Set position/extent
EXTENT = [
    ('Extent', [-10, 20, -10, 20]),
    ('Plane', {}),
]


# Use Octaves
GROUND_FROM_NOISE = [
    ('Octaves', {}),
    ('WeightedSum', {}),
]


# Use Octaves, and save
GROUND_FROM_NOISE_SAVE = [
    ('Octaves', {}),
    ('Save', 'Terrains/octaves'),
    ('WeightedSum', {}),
]

# Load from single octaves, and construct different results
GROUND_FROM_NOISE_LOAD = [
    ('Load', 'Terrains/octaves'),
    ('Random', 'weights'),  # Special command to generate new weights
    ('WeightedSum', {}),
]


# Load from single octaves, and construct different results with set (approximate) slope and roughness
GROUND_FROM_NOISE_LOAD_SET_SLOPE_AND_ROUGHNESS = [
    ('Load', 'Terrains/octaves'),
    ('Random', 'weights'),  # Special command to generate new weights
    ('Slope', {}),
    ('SetSlope', 5),
    ('Roughness', {}),
    ('SetRoughness', 1.03),
    ('WeightedSum', {}),
]


# Sample slope from some distribution
GROUND_FROM_NOISE_LOAD_SAMPLE_SLOPE = [
    ('Load', 'Terrains/octaves'),
    ('Random', 'weights'),  # Special command to generate new weights
    ('Slope', {}),
    ('SetDistribution', 'target_slope_deg=uniform(0,10)'),
    ('Sample', 'target_slope_deg'),
    ('SetSlope', {}),
    ('Roughness', {}),
    ('SetRoughness', 1.03),
    ('WeightedSum', {}),
]

# Use Loop:2x2 and Stack.
# Could be used for parallelization, but is currently not
SPLIT_AND_STACK = [
    ('GridSize', 150),  # Hack, must half this
    ('Loop', '2x2'),
    ('Seed', 'persistent_random'),  # Get same seed each loop, but different each run
    ('Basic', {}),
    ('WeightedSum', {}),
    ('EndLoop', {}),
    ('Stack', {}),
]


# Use Over to make flat at machine
FLAT_IN_CENTER = [
    ('Basic', {}),
    ('WeightedSum', {}),
    ('Cube', {'width': 5}),
    ('AsFactor', {}),
    ('Plane', {}),
    ('ToPrimary', {}),
    ('Compose', {}),
]


if __name__ == "__main__":
    """
    Script entry point for plot/renders of example configurations

    Usage:
    Run the script directly with Python:
            python -m artificial_terrains.examples

    # Note: for the reason of my probably not handling all imports
    # correctly, the following does not work:
    python artificial_terrains_cfg.py

    Or run it inside Blender in background mode:
        blender --python artificial_terrains_cfg.py --background

    Notes:
    - Rendering requires Blender and artificial_terrains package
    """
    current_module = globals()
    configs = {
        name: value for name, value in current_module.items()
        if name.isupper() and isinstance(value, list)
    }

    for name, config in configs.items():
        print(f"\n=== Running config: {name} ===")
        import artificial_terrains as at

        base_cfg = [
            ('Size', [30, 30]),
            ('GridSize', [300, 300]),
        ]

        plot_cfg = [
            ('Plot', {'filename_base': f'{name}',
                      'folder': 'example_plots'}),
        ]

        render_cfg = [
            ('BasicSetup', None),
            ('Holdout', None),
            ('Ground', {}),
            ('Camera', 45),
            ('Render', {'folder': 'example_renders',
                        'filename': f'{name}.png'}),
            ('ClearScene', None),
        ]

        # Run script for plots
        at.run(base_cfg + config + plot_cfg, verbose=True)

        # Try to run as blender script
        try:
            at.run(base_cfg + config + render_cfg, verbose=True)
        except ModuleNotFoundError:
            pass
