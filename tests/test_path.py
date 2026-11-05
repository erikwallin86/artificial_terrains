import logging
import os
import tempfile
import unittest

import numpy as np

import artificial_terrains as at
from artificial_terrains.modules.generating_function import Path
from artificial_terrains.modules.modules import MODULES
from artificial_terrains.utils.artificial_shapes import PathFunction
from artificial_terrains.utils.examples import (
    PathCfg,
    RandomPathCfg,
    SinePathCfg,
    WheelTracksCfg,
)
from artificial_terrains.utils.parse_utils import create_parser


class PathFunctionTest(unittest.TestCase):
    def test_interpolates_signed_cross_section_and_zeros_outside(self):
        sweep = PathFunction(
            points=[(-2, 0), (2, 0)],
            cross_section=[(-2, 0), (-1, -0.2), (0, 0.1), (1, -0.4), (2, 0)],
        )

        y = np.array([[-3.0, -1.0, 0.5, 1.0, 3.0]])

        np.testing.assert_allclose(
            sweep(np.zeros_like(y), y),
            [[0.0, -0.2, -0.15, -0.4, 0.0]],
        )

    def test_rejects_invalid_sweep_geometry(self):
        invalid_kwargs = [
            {"points": [(0, 0)]},
            {"points": [(0, 0), (0, 0)]},
            {"points": [(0, 0), (np.nan, 0)]},
            {"points": [(0, 0), (1, 0)], "cross_section": [(0, 0)]},
            {
                "points": [(0, 0), (1, 0)],
                "cross_section": [(-1, 0), (-1, 1), (1, 0)],
            },
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PathFunction(**kwargs)


class PathModuleTest(unittest.TestCase):
    def setUp(self):
        self.module = Path("", logging.getLogger("test_path_module"))
        self.extent = [-4, 6, -3, 5]

    def test_vertical_and_horizontal_paths_use_extent_center(self):
        vertical = self.module(default="vertical", extent=self.extent)
        horizontal = self.module(default="horizontal", extent=self.extent)

        self.assertEqual(vertical["path_points"], [[1.0, -3.0], [1.0, 5.0]])
        self.assertEqual(horizontal["path_points"], [[-4.0, 1.0], [6.0, 1.0]])

    def test_explicit_points_pass_through(self):
        points = [(-2, -1), (0, 2), (3, 1)]

        result = self.module(points=points, extent=self.extent)

        self.assertEqual(
            result["path_points"],
            [[-2.0, -1.0], [0.0, 2.0], [3.0, 1.0]],
        )

    def test_sine_path_is_centered_and_bounded_by_amplitude(self):
        result = self.module(
            default="sine",
            extent=[-5, 5, -4, 4],
            amplitude=2,
            wavelength=5,
            heading=0,
            phase=0,
            samples=101,
        )
        points = np.asarray(result["path_points"])

        self.assertEqual(points.shape, (101, 2))
        np.testing.assert_allclose(points[[0, -1], 0], [-5, 5])
        self.assertLessEqual(np.max(np.abs(points[:, 1])), 2.0)
        np.testing.assert_allclose(points[50], [0, 0], atol=1e-12)

    def test_sine_angle_alias_rotates_baseline(self):
        result = self.module(
            default="sine",
            extent=[-5, 5, -4, 4],
            amplitude=0,
            heading=0,
            angle=90,
            samples=3,
        )

        np.testing.assert_allclose(
            result["path_points"],
            [[0, -4], [0, 0], [0, 4]],
            atol=1e-12,
        )

    def test_random_path_is_deterministic_for_integer_seed(self):
        kwargs = {
            "default": "random",
            "extent": self.extent,
            "samples": 20,
            "curvature": 0.1,
        }

        first = np.asarray(self.module(seed=42, **kwargs)["path_points"])
        repeated = np.asarray(self.module(seed=42, **kwargs)["path_points"])
        other = np.asarray(self.module(seed=43, **kwargs)["path_points"])

        np.testing.assert_allclose(first, repeated)
        self.assertFalse(np.allclose(first, other))
        self.assertTrue(-4 <= first[0, 0] <= 6)
        self.assertTrue(-3 <= first[0, 1] <= 5)
        segment_angles = np.unwrap(np.arctan2(
            np.diff(first[:, 1]), np.diff(first[:, 0])))
        step = np.hypot(10, 8) / 19
        self.assertLessEqual(
            np.max(np.abs(np.diff(segment_angles))),
            0.1 * step,
        )


class PathPipelineTest(unittest.TestCase):
    def parse(self, modules):
        parser = create_parser(specific_args=["modules"])
        return parser.parse_args([
            "--save-dir", "/tmp/path-test", "--modules", *modules])

    def run_modules(self, modules, save_dir):
        at.run(
            modules,
            save_dir=save_dir,
            logger=logging.getLogger("test_path_pipeline"),
            pipe={
                "size": [4, 4],
                "resolution": 2,
                "extent": [-2, 2, -2, 2],
                "call_number": 0,
                "call_total": 1,
                "terrain_prim": [],
                "terrain_temp": [],
            },
        )
        if not (
                at._last_pipe.get("terrain_prim")
                or at._last_pipe.get("terrain_temp")):
            return None
        return at.get_terrain()

    def test_parser_and_runtime_match_explicit_pipeline(self):
        command_modules = [
            "Plane", "Path:vertical", "WheelTracks", "Combine:Add", "Plot"]
        args = self.parse(command_modules)

        self.assertEqual(
            args.modules,
            [
                ("Plane", None),
                ("Path", "vertical"),
                ("WheelTracks", None),
                ("Combine", "Add"),
                ("Plot", None),
            ],
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as save_dir:
            terrain = self.run_modules(args.modules, save_dir)

            self.assertLess(np.min(terrain.array), 0.0)
            self.assertAlmostEqual(np.max(terrain.array), 0.0)
            self.assertTrue(os.path.exists(
                os.path.join(save_dir, "Plot", "terrain_00000.png")))

    def test_custom_path_and_wheel_parameters(self):
        args = self.parse([
            "Plane",
            "Path:dict(points=[(-2,0),(2,0)])",
            "WheelTracks:dict(gauge=2.0,track_width=1.0,depth=0.2)",
            "Combine:Add",
        ])

        with tempfile.TemporaryDirectory(dir="/tmp") as save_dir:
            terrain = self.run_modules(args.modules, save_dir)

        self.assertAlmostEqual(np.min(terrain.array), -0.2)
        self.assertAlmostEqual(np.max(terrain.array), 0.0)

    def test_sine_cli_style_pipeline(self):
        args = self.parse([
            "Plane",
            "Path:sine",
            "WheelTracks",
            "Combine:Add",
        ])

        self.assertEqual(args.modules[1], ("Path", "sine"))
        with tempfile.TemporaryDirectory(dir="/tmp") as save_dir:
            terrain = self.run_modules(args.modules, save_dir)

        self.assertLess(np.min(terrain.array), 0.0)
        self.assertAlmostEqual(np.max(terrain.array), 0.0)

    def test_seed_module_controls_random_path(self):
        modules = [("Seed", 17), ("Path", "random")]
        with tempfile.TemporaryDirectory(dir="/tmp") as save_dir:
            self.run_modules(modules, save_dir)
            first = np.asarray(at._last_pipe["path_points"])
            self.run_modules(modules, save_dir)
            repeated = np.asarray(at._last_pipe["path_points"])

        np.testing.assert_allclose(first, repeated)

    def test_modules_are_registered(self):
        self.assertIn("Path", MODULES)
        self.assertIn("WheelTracks", MODULES)

    def test_cfgs_do_not_hide_combination(self):
        self.assertEqual(PathCfg().modules, [("Path", "vertical")])
        self.assertEqual(
            WheelTracksCfg().modules,
            [(
                "WheelTracks",
                {"gauge": 1.6, "track_width": 0.6, "depth": 0.08},
            )],
        )

    def test_sine_and_random_path_cfgs(self):
        self.assertEqual(
            SinePathCfg(
                amplitude=1,
                wavelength=4,
                heading=30,
                phase=0.5,
                samples=10,
            ).modules,
            [(
                "Path",
                {
                    "direction": "sine",
                    "amplitude": 1,
                    "wavelength": 4,
                    "heading": 30,
                    "phase": 0.5,
                    "samples": 10,
                },
            )],
        )
        self.assertEqual(
            RandomPathCfg(seed=7, curvature=0.2, samples=10).modules,
            [
                ("Seed", 7),
                ("Path", {
                    "direction": "random",
                    "curvature": 0.2,
                    "samples": 10,
                }),
            ],
        )


if __name__ == "__main__":
    unittest.main()
