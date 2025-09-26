# Expose a friendly API to use as a library

"""
Artificial Terrains
===================

Modular, extensible terrain generation library.

"""
import sys
import os
import logging

sys.path.append(os.path.dirname(__file__))

from modules.modules import MODULES


def run(modules, save_dir="", logger=None, general_kwargs=None, pipe=None,
        verbose=False):
    """
    Run a list of (name, options) modules just like the CLI script.
    Example:
        run([("Basic", {}), ("WeightedSum", {})])
    """

    global _last_pipe, _last_primary

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
    _last_primary = last_pipe.get('terrain_prim', [])


def get_terrain():
    """Return the last terrain (single)."""
    if _last_primary:
        return _last_primary[-1]
    return None


def get_terrains():
    """Return all terrains from the last run."""
    return _last_primary
