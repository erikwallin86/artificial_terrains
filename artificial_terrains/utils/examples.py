from dataclasses import dataclass, field
import inspect
import os
import numpy as np


# ## Examples which are simply lists ###

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


#
BASIC_AND_MAX = [
    ('Basic', [10, 10, 10, 10, 10]),
    ('Combine', 'Max'),
    ('Scale', 3),
]


BASIC_AND_MIN = [
    ('Basic', [10, 10, 10, 10, 10]),
    ('Combine', 'Min'),
    ('Scale', 3),
]


BASIC_AND_MAX2 = [
    ('Basic', [50, 20, 5]),
    ('Combine', 'Max'),
    ('Scale', 3),
]


BASIC_AND_MIN2 = [
    ('Basic', [50, 20, 5]),
    ('Combine', 'Min'),
    ('Scale', 3),
]


RANDOM_AND_SMOOTHSTEP = [
    ('SetDistribution', "height=uniform[0,2]"),
    ('Random', 5),
    ('SmoothStep', {}),
    ('Combine', 'Add'),
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


# Classes to make 'configs' (Cfg) with default, changeable parameters
@dataclass
class ArtificialTerrainCfg:
    @property
    def modules(self) -> list:
        """Each config must expose a list of modules."""
        raise NotImplementedError(f"{type(self).__name__} must define a `modules` property")
        # return []

    # Needs to be iterable to behave as a list for artifical_terrains
    def __iter__(self):
        return iter(self.modules)

    # Make sure cfg:s can be added, with the result being a 'Combined cfg'
    def __add__(self, other: "ArtificialTerrainCfg") -> "CombinedArtificialTerrainCfg":
        if not isinstance(other, (ArtificialTerrainCfg, list)):
            return NotImplemented
        return CombinedArtificialTerrainCfg(children=[self, other])

    def __radd__(self, other):
        if not isinstance(other, (ArtificialTerrainCfg, list)):
            return NotImplemented
        return CombinedArtificialTerrainCfg(children=[other, self])


@dataclass
class CombinedArtificialTerrainCfg(ArtificialTerrainCfg):
    """Composite config, holds children configs."""
    children: list = field(default_factory=list)

    @property
    def modules(self):
        # flatten children’s modules
        combined = []
        for child in self.children:
            if isinstance(child, list):
                combined.extend(child)
            else:
                combined.extend(child.modules)
        return combined

    def __setattr__(self, key, value):
        """Try to propagate sets into children if they have the attribute."""
        if key in {"children", "modules"}:
            # Don't redirect for these two special attributes.
            # They are "structural" parts of the composite.
            object.__setattr__(self, key, value)
            return
        else:
            # propagate to children only
            for child in self.children:
                if hasattr(child, key):
                    setattr(child, key, value)


@dataclass
class FlatCfg(ArtificialTerrainCfg):
    @property
    def modules(self):
        return [
            ('Plane', {}),
        ]


@dataclass
class AngledPlaneCfg(ArtificialTerrainCfg):
    pitch_deg: float = 10

    @property
    def modules(self):
        return [
            ('Plane', {'pitch_deg': self.pitch_deg}),
        ]


# Gaussian hill, fully specified
@dataclass
class GaussianHillCfg(ArtificialTerrainCfg):
    @property
    def modules(self):
        return [
            ('Gaussian', {
                'position': [5, 5],
                'height': 2.5,
                'width': 5,
                'aspect': 1.5,
                'yaw_deg': 45,
                'pitch_deg': None,
            }),
        ]


# N number of random Gaussian hills
@dataclass
class GaussianHillRandomCfg(ArtificialTerrainCfg):
    number: int = 3

    @property
    def modules(self):
        return [
            ('Random', self.number),
            ('Gaussian', {}),
            ('Combine', {}),
            # Allowing adding to other:
            # (even if that terrain is in 'temporary')
            ('ToPrimary', {}),
            ('Combine', {}),
        ]


# Mountain in horizon
@dataclass
class MountansInHorizonCfg(ArtificialTerrainCfg):
    width: float = 30

    @property
    def modules(self):
        return [
            ('Basic', 10),
            ('Add', 1.5),  # Get noise in [1.5, 2.5]
            ('Scale', 0.5),
            ('Donut', {'width': self.width}),
            ('Combine', 'Prod'),
            # Allowing adding to other:
            ('ToPrimary', {}),
            ('Combine', {}),
        ]


# Explicit function
@dataclass
class ExplicitFunctionCfg(ArtificialTerrainCfg):
    function: str = 'np.sin(2*np.pi*x/20.0)'

    @property
    def modules(self):
        return [
            ('Function', self.function),
        ]


@dataclass
class LoadOrGenerateAndSetSlopeCfg(ArtificialTerrainCfg):
    ''' Example of some extended functionality'''
    folder: str = 'Terrains/octaves'
    slope_deg: float = 8
    roughness: float = 1.03

    @property
    def modules(self):
        modules = []
        if os.path.exists(self.folder):
            modules.extend(
                [('Load', self.folder),
                 ('Random', 'weights'),]
            )
        else:
            modules.extend(
                [('Octaves', {}),
                 ('Save', self.folder),]
            )
        modules.extend(
            [
                ('Slope', {}),
                ('SetSlope', self.slope_deg),
                ('Roughness', {}),
                ('SetRoughness', self.roughness),
                ('WeightedSum', {}),
            ]
        )
        return modules


@dataclass
class LoadFromSeveralAndSetSlopeAndRoughnessCfg(ArtificialTerrainCfg):
    ''' Example of some extended functionality'''
    folder: str = 'Terrains/multiple_octaves'
    number_of_sets: int = 5
    slope_deg: float = 8
    roughness: float = 1.05

    @property
    def modules(self):
        import random
        digits = int(2 + np.log10(self.number_of_sets))

        modules = []
        # Define filename to check that files exist
        a_filename = f'terrain_temp_{self.number_of_sets - 1:0{digits}d}_{0:05d}.npz'
        if not os.path.exists(os.path.join(self.folder, a_filename)):
            modules.extend(
                [('Loop', self.number_of_sets),
                 ('Octaves', {}),
                 ('Save', self.folder),
                 ('ClearTerrain', None),
                 ('EndLoop', None),
                 ]
            )

        files = [
            f'{self.folder}/terrain_temp_{random.randint(0, self.number_of_sets - 1):0{digits}d}_{i:05d}.npz'
            for i in range(10)
        ]
        modules.extend(
            [
                ('Load', files),
                ('Random', 'weights'),
                ('Slope', {}),
                ('SetSlope', self.slope_deg),
                ('Roughness', {}),
                ('SetRoughness', self.roughness),
                ('WeightedSum', {}),
            ]
        )
        return modules


@dataclass
class LoadFromSeveralAndSetSlopeCfg(ArtificialTerrainCfg):
    ''' Example of some extended functionality'''

    num_octaves: int = 10
    start: float = 128

    folder: str = 'Terrains/multiple_octaves2'
    number_of_sets: int = 5
    slope_deg: float = 8

    @property
    def modules(self):
        import random
        digits = int(2 + np.log10(self.number_of_sets))

        modules = []
        # Define filename to check that files exist
        a_filename = f'terrain_temp_{self.number_of_sets - 1:0{digits}d}_{0:05d}.npz'
        if not os.path.exists(os.path.join(self.folder, a_filename)):
            modules.extend(
                [('Loop', self.number_of_sets),
                 ('Octaves', {'num_octaves': self.num_octaves, 'start': self.start}),
                 ('Save', self.folder),
                 ('ClearTerrain', None),
                 ('EndLoop', None),
                 ]
            )

        files = [
            f'{self.folder}/terrain_temp_{random.randint(0, self.number_of_sets - 1):0{digits}d}_{i:05d}.npz'
            for i in range(self.num_octaves)
        ]
        modules.extend(
            [
                ('Load', files),
                ('Random', 'weights'),
                ('Slope', {}),
                ('SetSlope', self.slope_deg),
                ('WeightedSum', {}),
            ]
        )
        return modules


@dataclass
class AndFlatInCenterCfg(ArtificialTerrainCfg):
    width: float = 5

    @property
    def modules(self):
        return [
            # Assumes there is already a terrain in primary
            ('Cube', {'width': self.width}),
            ('AsFactor', {}),
            ('Plane', {}),
            ('ToPrimary', {}),
            ('Compose', {}),
        ]


@dataclass
class AndSetDifficulty(ArtificialTerrainCfg):
    ''' Curriculum thing, interpolating with Plane to set difficulty '''
    difficulty: float = 0

    @property
    def modules(self):
        return [
            # Assumes there is already a terrain in primary
            ('Set', f'factor={self.difficulty}'),
            ('Plane', {}),
            ('ToPrimary', {}),
            ('Compose', 'Under')
        ]


@dataclass
class StoneCircleCfg(ArtificialTerrainCfg):
    width: float = 25
    number_of_stones: int = 30

    @property
    def modules(self):
        return [
            ('Donut', {'width': self.width}),
            ('AsProbability', {}),
            ('SetDistribution', "width=uniform(2,2)"),
            ('Random', self.number_of_stones),
            ('Cube', {}),
            ('Combine', "Max"),
            # Allowing adding to other:
            ('ToPrimary', {}),
            ('Combine', {}),
        ]


@dataclass
class PerimiterWallsCfg(ArtificialTerrainCfg):
    width: float = 28.0
    height: float = 2.0

    @property
    def modules(self):
        return [
            ('Cube', {'width': self.width, 'height': self.height}),
            ('Negate', {}),
            ('Add', self.height),
            # Allowing adding to other:
            ('ToPrimary', {}),
            ('Combine', {}),
        ]


@dataclass
class TerrainAndHillsAndPerimiterCfg(ArtificialTerrainCfg):
    ''' Combine several things into something semi-complicated'''

    # Size and grid-size
    size_x: float = 50
    size_y: float = 50
    grid_size_x: int = 500
    grid_size_y: int = 500
    # Parameters for general terrain
    slope_deg: float = 5

    folder: str = 'Terrains/combine_several'

    perimiter_walls: bool = True
    gaussian_hills: int = 10

    def __post_init__(self):
        # Add hash to self.folder, to get unique folders for different
        # size and grid-size

        import hashlib
        # Compute hash from instance values
        data = (self.size_x, self.size_y, self.grid_size_x, self.grid_size_y)
        data_bytes = repr(data).encode("utf-8")
        hash_digest = hashlib.sha1(data_bytes).hexdigest()[:6]

        self.folder = f"{self.folder}_{hash_digest}"

    @property
    def modules(self):

        # Add hash to folder
        print(f"self.folder:{self.folder}")
        # exit(0)

        # Calculate number of octaves and start
        # We want 'start' a factor 2~4 more than the max size,
        # and then set num_octaves to get the smallest features
        # in the order of 1 meter.
        max_size = np.maximum(self.size_x, self.size_y)
        start_log = int(np.log2(max_size)+2)
        end_log = 1
        num_octaves = start_log - end_log + 1
        start = np.power(2, start_log)

        size = [
            # Set Size and GridSize
            ('Size', [self.size_x, self.size_y]),
            ('GridSize', [self.grid_size_x, self.grid_size_y]),
        ]

        # Setup general terrain
        terrain = LoadFromSeveralAndSetSlopeCfg(
            folder=self.folder,
            slope_deg=self.slope_deg,
            num_octaves=num_octaves,
            start=start,
        )

        # Setup walls
        if self.perimiter_walls:
            perimiter = PerimiterWallsCfg(width=self.size_x-2)
        else:
            perimiter = []

        if self.gaussian_hills:
            hills = GaussianHillRandomCfg(number=self.gaussian_hills)
        else:
            hills = []

        return size + terrain + perimiter + hills


COMBINED_SET_SLOPE_AND_FLAT_IN_CENTER = (
    LoadOrGenerateAndSetSlopeCfg()
    + AndFlatInCenterCfg()
)


COMBINED_SET_SLOPE_AND_PERIMITER_WALLS = (
    LoadOrGenerateAndSetSlopeCfg()
    + PerimiterWallsCfg()
)


COMBINED_SET_SLOPE_AND_STONE_CIRCLE = (
    LoadOrGenerateAndSetSlopeCfg()
    + StoneCircleCfg()
)


COMBINED_SET_SLOPE_AND_MOUNTAINS_IN_HORIZON = (
    LoadOrGenerateAndSetSlopeCfg()
    + MountansInHorizonCfg()
)


COMBINED_SET_SLOPE_AND_GAUSSIAN_HILLS = (
    LoadOrGenerateAndSetSlopeCfg()
    + GaussianHillRandomCfg()
)


INSTANTIATED_CFG_GAUSSIAN_HILLS = GaussianHillRandomCfg(number=10)

INSTANTIATED_CFG_TERRAIN_HILL_PERIMITER_SLOPE_5 = TerrainAndHillsAndPerimiterCfg()
INSTANTIATED_CFG_TERRAIN_HILL_PERIMITER_SLOPE_10 = TerrainAndHillsAndPerimiterCfg(
    slope_deg=10)
INSTANTIATED_CFG_TERRAIN_HILL = TerrainAndHillsAndPerimiterCfg(
    perimiter_walls=False
)
INSTANTIATED_CFG_TERRAIN_PERIMITER = TerrainAndHillsAndPerimiterCfg(
    gaussian_hills=0
)
INSTANTIATED_CFG_TERRAIN = TerrainAndHillsAndPerimiterCfg(
    gaussian_hills=0, perimiter_walls=False
)

INSTANTIATED_CFG_TERRAIN_STRIP_Y = TerrainAndHillsAndPerimiterCfg(
    gaussian_hills=0, perimiter_walls=False, size_x=10, grid_size_x=100,
)
INSTANTIATED_CFG_TERRAIN_STRIP_X = TerrainAndHillsAndPerimiterCfg(
    gaussian_hills=0, perimiter_walls=False, size_y=10, grid_size_y=100,
)


# 10 different terrains from the same set of octaves


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
        and not name.startswith("AND")
    }

    # other style: dataclass config with a .modules property
    cfg_configs = {
        name: value() for name, value in current_module.items()
        if inspect.isclass(value) and issubclass(value, ArtificialTerrainCfg)
        and not name.startswith("And")
    }
    del cfg_configs['ArtificialTerrainCfg']
    del cfg_configs['CombinedArtificialTerrainCfg']

    combined_configs = {
        name: value for name, value in current_module.items()
        if name.isupper() and isinstance(value, CombinedArtificialTerrainCfg)
    }

    instantiated_configs = {
        name: value for name, value in current_module.items()
        if name.isupper() and isinstance(value, ArtificialTerrainCfg)
    }

    # for name, _class in cfg_configs.items():
    configs = {
        **configs,
        **cfg_configs,
        **combined_configs,
        **instantiated_configs,
    }

    # Override with test
    # configs = {'test': LoadFromSeveralAndSetSlopeCfg()}

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

        folder = 'example_renders'
        filename = f'{name}.png'

        render_cfg = [
            ('ClearScene', None),
            ('BasicSetup', None),
            ('Holdout', None),
            ('Ground', {}),
            ('Camera', 45),
            ('Render', {'folder': folder,
                        'filename': filename}),
        ]

        # Possibly skip already done
        if os.path.isfile(os.path.join(folder, filename)):
            continue

        # Try to run as blender script
        try:
            at.run(base_cfg + config + render_cfg + plot_cfg, verbose=True)
        except ModuleNotFoundError:
            # Otherwise just run as a plot script
            at.run(base_cfg + config + plot_cfg, verbose=True)
