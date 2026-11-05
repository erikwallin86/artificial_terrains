# Expose a friendly API to use as a library

"""
Artificial Terrains
===================

Modular, extensible terrain generation library.

"""
import sys
import logging

from typing import TYPE_CHECKING, Any, Iterable

from . import configs

if TYPE_CHECKING:
    from .constructors import *  # noqa: F401,F403
    from .utils.terrains import Terrain

__all__ = ["configs"]


def __getattr__(name):
    """Load module constructors (at.Gaussian, ...) on first use.

    Importing them loads all module files, which is slow (plotting libs).
    """
    import importlib
    constructors = importlib.import_module(f"{__name__}.constructors")
    if name in constructors.__all__:
        return getattr(constructors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_last_pipe = {}


def run(modules: Iterable[tuple[str | None, Any]], save_dir: str = "",
        logger: logging.Logger | None = None,
        general_kwargs: dict[str, Any] | None = None,
        pipe: dict[str, Any] | None = None, verbose: bool = False) -> None:
    """
    Run a list of (name, options) modules just like the CLI script.
    Example:
        run([("Basic", {}), ("WeightedSum", {})])
    """

    global _last_pipe

    if logger is None:
        # Always use the same base logger name
        logger = logging.getLogger("artificial_terrains")
        # Essentially turn of logging by default
        logger.setLevel(logging.CRITICAL)

    if verbose:
        # Clear any logger to avoid doubles
        logger.handlers.clear()

        # Add a logger printing to stdout
        handler = logging.StreamHandler(sys.stdout)
        # formatter = logging.Formatter("%(levelname)s: %(message)s")
        # handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    if general_kwargs is None:
        general_kwargs = {}

    if pipe is None:
        pipe = {
            'size': 50,
            'resolution': 2,
            'extent': [-25, 25, -25, 25],
            'call_number': 0,
            'call_total': 1,
            # Fresh lists each run; modules otherwise fall back to shared
            # mutable defaults, and terrains leak between runs
            'terrain_prim': [],
            'terrain_temp': [],
        }

    # from modules.modules import MODULES
    from artificial_terrains.modules.modules import MODULES
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
        module_obj.module_options = kwargs.copy()
        kwargs = {**general_kwargs, **kwargs}
        module_obj.module_kwargs = kwargs.copy()
        list_of_modules_kwargs_tuples.append((module_obj, kwargs))

    # Run recursively
    from .generate_terrain import recursive_module_call
    last_pipe = recursive_module_call(
        list_of_modules_kwargs_tuples, pipe=pipe, logger=logger)

    # Store last_pipe globally
    _last_pipe = last_pipe


def get_terrain() -> "Terrain | None":
    """Return the last terrain (single)."""

    if _last_pipe.get('terrain_prim'):
        return _last_pipe['terrain_prim'][-1]
    elif _last_pipe.get('terrain_temp'):
        return _last_pipe['terrain_temp'][-1]
    else:
        return None


def get_terrains() -> "list[Terrain] | None":
    """Return all terrains from the last run."""

    if _last_pipe.get('terrain_prim'):
        return _last_pipe['terrain_prim']
    elif _last_pipe.get('terrain_temp'):
        return _last_pipe['terrain_temp']
    else:
        return None
