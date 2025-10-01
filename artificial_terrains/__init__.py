# Expose a friendly API to use as a library

"""
Artificial Terrains
===================

Modular, extensible terrain generation library.

"""
import sys
import os
import logging

# make artificial_terrains.examples as a valid import:
from .utils import examples
__all__ = ["examples"]

# Hack: extend sys.path so imports still work after moving all code into
# artificial_terrains/artificial_terrains for packaging. This avoids having
# to rewrite all existing `import modules...` calls right now.
sys.path.append(os.path.dirname(__file__))


def run(modules, save_dir="", logger=None, general_kwargs=None, pipe=None,
        verbose=False):
    """
    Run a list of (name, options) modules just like the CLI script.
    Example:
        run([("Basic", {}), ("WeightedSum", {})])
    """

    global _last_pipe

    if logger is None:
        logger = logging.getLogger("artificial_terrains")

        if verbose:
            handler = logging.StreamHandler(sys.stdout)   # send logs to terminal
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)  # or DEBUG if you want more detail
        else:
            logger.addHandler(logging.NullHandler())

    if general_kwargs is None:
        general_kwargs = {}

    if pipe is None:
        pipe = {
            'size': 50,
            'resolution': 2,
            'extent': [-25, 25, -25, 25],
            'call_number': 0,
            'call_total': 1,
        }

    from modules.modules import MODULES
    # Build module objects
    list_of_modules_kwargs_tuples = []
    for module, options in modules:
        if module is None:
            continue
        module_class = MODULES[module]
        module_obj = module_class(save_dir, logger)
        if isinstance(options, dict):
            kwargs = options
        elif options is None:
            kwargs = {}
        else:
            kwargs = {'default': options}
        kwargs = {**general_kwargs, **kwargs}
        list_of_modules_kwargs_tuples.append((module_obj, kwargs))

    # Run recursively
    from .generate_terrain import recursive_module_call
    last_pipe = recursive_module_call(
        list_of_modules_kwargs_tuples, pipe=pipe, logger=logger)

    # Store last_pipe globally
    _last_pipe = last_pipe


def get_terrain():
    """Return the last terrain (single)."""

    if 'terrain_prim' in _last_pipe:
        return _last_pipe['terrain_prim'][-1]
    elif 'terrain_temp' in _last_pipe:
        return _last_pipe['terrain_temp'][-1]
    else:
        return None


def get_terrains():
    """Return all terrains from the last run."""

    if 'terrain_prim' in _last_pipe:
        return _last_pipe['terrain_prim']
    elif 'terrain_temp' in _last_pipe:
        return _last_pipe['terrain_temp']
    else:
        return None
