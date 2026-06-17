"""Public configuration presets for terrain generation."""

from .utils.examples import *  # noqa: F401,F403


def _is_cfg_target(target):
    from dataclasses import is_dataclass
    from .utils.examples import ArtificialTerrainCfg, CombinedArtificialTerrainCfg

    if isinstance(target, type):
        return (
            issubclass(target, ArtificialTerrainCfg)
            and target not in (ArtificialTerrainCfg, CombinedArtificialTerrainCfg)
        )

    if isinstance(target, list):
        return True

    if is_dataclass(target):
        return hasattr(target, "modules")

    return hasattr(target, "modules")


def get_cfg_names():
    """Return the names of public config targets exposed by examples."""
    from .utils import examples

    names = []
    for name in dir(examples):
        if (
            name.startswith("_")
            or name.startswith("INSTANTIATED")
            or name in examples.EXCLUDED_CFG_NAMES
        ):
            continue
        target = getattr(examples, name)
        if not _is_cfg_target(target):
            continue
        names.append(name)
    return names


def is_cfg_name(name):
    """Return True when ``name`` resolves to a known config target."""
    from .utils import examples

    if not hasattr(examples, name):
        return False

    return _is_cfg_target(getattr(examples, name))


def get_cfg_target(name, options=None):
    """Resolve a cfg name to a list or instantiated cfg object."""
    from dataclasses import is_dataclass
    from .utils import examples

    target = getattr(examples, name)

    if isinstance(target, list):
        if options not in (None, {}):
            raise ValueError(f"CFG:{name} does not accept options.")
        return target

    if isinstance(target, type):
        target = target()
    elif is_dataclass(target):
        if options not in (None, {}):
            raise ValueError(
                f"{name} is an instantiated cfg and does not accept overrides. "
                "Use the corresponding *Cfg class name instead."
            )

    if options:
        for key, value in options.items():
            setattr(target, key, value)

    return target
