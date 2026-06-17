from .general import CFGModule
from .. import configs


def module_factory(cfg_name):
    """Create a runtime module class bound to a specific config name."""

    if not configs.is_cfg_name(cfg_name):
        raise ValueError(f"Unknown cfg name: {cfg_name}")

    def __call__(self, default=None, name=None, **kwargs):
        return CFGModule.__call__(self, default=cfg_name, name=cfg_name, **kwargs)

    return type(
        cfg_name,
        (CFGModule,),
        {
            "__call__": __call__,
            "__doc__": f"Runtime module wrapper for config {cfg_name}.",
        },
    )


def build_cfg_modules(cfg_names=None):
    """Build a ``MODULES``-compatible mapping of cfg names to module classes."""
    cfg_names = configs.get_cfg_names() if cfg_names is None else cfg_names
    return {cfg_name: module_factory(cfg_name) for cfg_name in cfg_names}
