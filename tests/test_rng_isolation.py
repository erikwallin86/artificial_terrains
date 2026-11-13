"""The first ``at.run()`` in a fresh process must not consume global NumPy RNG.

Registering presets used to evaluate each preset's ``modules`` property, and
some presets (e.g. ``AndFlatInCenterCfg``) draw random numbers there. That made
results depend on whether artificial_terrains had already been initialized.
"""
import subprocess
import sys
import textwrap
import unittest

SCRIPT = textwrap.dedent("""
    import numpy as np
    import artificial_terrains as at

    SEED = 123

    def generate_flat():
        at.run([
            ("Seed", SEED),
            ("Size", [1.0, 1.0]),
            ("GridSize", [3, 3]),
            ("Plane", {}),
        ])

    np.random.seed(SEED)
    expected = np.random.random(5)

    np.random.seed(SEED)
    generate_flat()  # first use initializes the module/preset registry
    after_first = np.random.random(5)

    np.random.seed(SEED)
    generate_flat()
    after_second = np.random.random(5)

    print(np.array_equal(expected, after_first),
          np.array_equal(expected, after_second))
""")


class RngIsolationTest(unittest.TestCase):
    def test_first_run_in_fresh_process_preserves_global_rng(self):
        # Fresh interpreter, so the registry is not already initialized
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        last_line = result.stdout.strip().splitlines()[-1]
        self.assertEqual(last_line, "True True", result.stdout)


if __name__ == "__main__":
    unittest.main()
